# propagation with unity time step
function propagate(A0, A::Vector{<:AbstractMatrix}, u, x0, cache=nothing)

    T = eltype(complex(float(x0)))
    Nt = size(u, 2)

    if cache === nothing
        cache = setup_grape_cache(A0, T.(x0), size(u))
    end

    x, λ, dJdu, Uk_vec = cache
    cache.u .= u # To know the u used for the compuations

    x[1] .= x0

    # Propage forwards
    Threads.@threads for k=1:Nt
        Ak = caches=cache.exp_cache[Threads.threadid()][end]  # temporary storage of Ak = A0 + u[1,k]*A[1] + ...
        Ak .= A0
        for j=1:length(A)
            Ak .+= u[j,k] .* A[j]
        end

        Uk_vec[k] .= ExponentialUtilities._exp!(Ak, caches=cache.exp_cache[Threads.threadid()])# exp(Ak)
    end

    for k=1:Nt
        mul!(x[k+1], Uk_vec[k], x[k])
    end

    return x
end

# Utilizes the cache from the last call to propagate
function grape_sensitivity(A0, A::Vector{<:AbstractMatrix}, dJfinal_dx, u, x0, cache; dUkdp_order=3, dL_dx=nothing)

    if u != cache.u
        error("Cache data from other control signal u") # Not sure if this will ever happen. Currently captured in the optimizer callbacks. Otherwise, one could call propagate here.
    end
    x, λ, dJdu, Uk_vec = cache

    T = eltype(x[1])
    Nt = size(u, 2)
    @assert length(x) == length(λ) == Nt + 1

    λ[Nt+1] .= dJfinal_dx(x[Nt+1])
    if dL_dx !== nothing
        λ[Nt+1] .+= dL_dx(x[Nt+1])
    end

    # Propage co-states backwards
    for k=Nt:-1:1
        mul!(λ[k], Uk_vec[k]', λ[k+1])

        if dL_dx !== nothing
            λ[k] .+= dL_dx(x[k])
        end
    end

    # Compute sensitivity of J with respect to u
    dUkdu = [similar(A0) for k=1:length(A)]
    expm_jac_cache = [similar(A0) for k=1:4]
    #dxduk = similar(x[1])

    for k=Nt:-1:1
        # Compute derivative of exp(A0 + u1*A1 + u2*A2 + ...) wrt u1 = u[1,k], u2 = u[2,k], ...
        expm_jacobian!(dUkdu, A0, A, u[:,k], expm_jac_cache, dUkdp_order)

        # Compute the sensitivity of J wrt u[1,k], ..
        for j=1:length(A)
            # compute sum( λ[k+1] .* (dx[k]/du[j]) ) by doing the sum column wise
            dJdu[j, k] = _compute_u_sensitivity(x[k], λ[k+1], dUkdu[j])
        end
    end

    return dJdu
end

function setup_grape_cache(A0, x0, u_size)
    T = float(eltype(x0)) # Could be both complex and real
    Tc = complex(T)
    Nt = u_size[2]

    if (eltype(x0) <: Real && size(x0, 1) != 2size(A0, 1)) ||
       (eltype(x0) <: Complex && size(x0, 1) != size(A0, 1))
        error("Error when creating cache, A0 and x0 have incompatiable dimensions")
    end

    return (x = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1],
            λ = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1],
            dJdu = Matrix{real(T)}(undef, u_size),
            Uk_vec = Matrix{Tc}[Matrix{Tc}(undef, size(x0,1), size(x0,1)) for k=1:Nt+1],
            exp_cache = Vector{Matrix{Tc}}[Matrix{Tc}[similar(A0) for k=1:(5+1)] for l=1:Threads.nthreads()], # Should be one longer than needed for _exp!
            u = Matrix{real(T)}(undef, u_size) # The u used for computaitons, to avoid reevaluating
            )
end


function wrap_pwc(f)
    function(dx, x, p, t)
        @inbounds @views for j=1:size(x,2)
            f(dx[:,j], x[:,j], p[1], t)
        end
    end
end


function propagate_pwc(f, x0, u, Δt, cache=nothing; dt=0.1Δt)

    if cache === nothing
        n = Int(size(x0,1)/2)
        cache = setup_grape_cache(zeros(n,n), x0, size(u))
    end


    x_cache = cache[1]

    tgate = Δt * size(u, 2)

    function update_u!(integrator)
        k = Int(round(integrator.t/Δt))
        integrator.p[1] .= integrator.p[2][:, k+1]
    end
    pcb=PeriodicCallback(update_u!, Δt, initial_affect=true, save_positions=(true,false))
    prob = ODEProblem{true}(wrap_pwc(f), x0, (0.0, tgate), callback=pcb)

    sol = solve(prob, Tsit5(), x_cache, p=([0.0; 0.0], u), saveat=[0.0,tgate], adaptive=false, dt=dt)
    sol
end

setup_cache(x0, Nt) = ([similar(x0) for k=1:Nt+1], [similar(x0) for k=1:Nt+1], Matrix{float(real(eltype(x0)))}(undef, 2, Nt))
function compute_pwc_gradient(dλdt, dJfinal_dx::Function, u, Δt, A0, A, cache; dUkdp_order=2, dt=0.1Δt)

    x, λ_store, dJdu = cache

    tgate = Δt * size(u, 2)

    # Co-states
    λf_real = dJfinal_dx(r2c(x[end]))
    λf = c2r(λf_real)

    function update_u_bwd!(integrator)
        k = max(Int(round(integrator.t/Δt)) - 1, 0)
        integrator.p[1] .= integrator.p[2][:, k+1]
    end
    pcb_adj = PeriodicCallback(update_u_bwd!, -Δt, initial_affect=true, save_positions=(true,false)) # save_positions should maybe be other way around, but really shouldn't matter?
    prob_adj = ODEProblem{true}(wrap_pwc(dλdt), λf, (tgate, 0.0), callback=pcb_adj)

    sol_adj = solve(prob_adj, Tsit5(), λ_store, p=([0.0; 0.0], u), saveat=[0.0,tgate], adaptive=false, dt=dt)

    # Compute sensitivity of J wrt the control signal u
    λ = sol_adj.u::typeof(λ_store)

    if dUkdp_order == 0; return dJdu; end

    dUkdu = [similar(A0) for k=1:length(A)]
    expm_jac_cache = [similar(A0) for k=1:4]

    Nt = size(u, 2)
    for k=Nt:-1:1
        expm_jacobian!(dUkdu, A0, A, u[:,k], expm_jac_cache, dUkdp_order, Δt=Δt)

        for j=1:2
            dJdu[j, k] = _compute_u_sensitivity(r2c(x[k]), r2c(λ[end-(k+1)+1]), dUkdu[j])
        end
    end

    return dJdu
end

"""
    expm_jacobian!(dFdp, A0, A, p, tmp_data, approx_order=2)

Modifies `dFdp` in-place to contain the of matrix-valued derivatives of `F = exp(A0 + p[1]*A[1] + p[2]*A[2] + ...)`
with respect to p1, p2, ...
"""
function expm_jacobian!(dFdp, A0, A, p, tmp_data, approx_order=2; Δt=1)

    for j=1:length(A) # The first-order approximation
        dFdp[j] .= Δt * A[j]
    end
    approx_order <= 1 && return

    X = tmp_data[1]
    AjX = tmp_data[2]
    XAj = tmp_data[3]

    X .= A0
    for j=1:length(A)
        X .+= p[j] .* A[j]
    end

    for j=1:length(A)
        if approx_order >= 2
            mul!(AjX, A[j], X)
            mul!(XAj, X, A[j])
            dFdp[j] .+= (Δt^2/2) .* (AjX .+ XAj)
        end
        if approx_order >= 3
            mul!(dFdp[j], AjX, X, Δt^3/6, 1)
            mul!(dFdp[j], XAj, X, Δt^3/6, 1)
            mul!(dFdp[j], X, XAj, Δt^3/6, 1)
        end
        if approx_order >= 4
            X2 = tmp_data[4]
            mul!(X2, X, X)
            mul!(dFdp[j], AjX, X2, Δt^4/24, 1)
            mul!(dFdp[j], XAj, X2, Δt^4/24, 1)
            mul!(dFdp[j], X2, AjX, Δt^4/24, 1)
            mul!(dFdp[j], X2, XAj, Δt^4/24, 1)
        end
    end
end


#dJdu[j, k] = sum(real(dot(λi, dUkdu[j], xi)) for (λi, xi) in zip(eachcol(r2c(λ[end-(k+1)+1])), eachcol(r2c(x[k]))))
function _compute_u_sensitivity(xk, λkp1, dU_duj)
    dJ_duj = 0
    @views for l=1:size(xk,2)
        dJ_duj += real(dot(λkp1[:,l], dU_duj, xk[:,l]))
    end
    return dJ_duj
end