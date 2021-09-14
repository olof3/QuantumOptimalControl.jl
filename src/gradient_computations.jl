# propagation with unity time step
function propagate(A0, A::Vector{<:AbstractMatrix}, u, x0, cache=nothing)

    T = eltype(complex(float(x0)))
    Nt = size(u, 2)

    if cache === nothing
        cache = setup_grape_cache(A0, T.(x0), size(u))
    end

    x, λ, dJdu, Uk_vec = cache
    cache.u .= u # To know the u used for the compuations

    x[1] .= copy(x0)

    # Propage forwards
    Ak = similar(A0) # temporary storage of Ak = A0 + u[1,k]*A[1] + ...
    for k=1:Nt
        Ak .= A0
        for j=1:length(A)
            Ak .+= u[j,k] .* A[j]
        end

        Uk_vec[k] .= ExponentialUtilities._exp!(Ak, caches=cache.exp_cache)# exp(Ak)

        mul!(x[k+1], Uk_vec[k], x[k])
    end

    return x
end

# Utilizes the cache from the last call to propagate
function grape_sensitivity(A0, A::Vector{<:AbstractMatrix}, Jfinal, u, x0, cache; dUkdp_order=3)

    if u != cache.u
        error("Cache data from other control signal u")
    end
    x, λ, dJdu, Uk_vec = cache

    T = eltype(x[1])
    Nt = size(u, 2)

    λ[end] .= Zygote.gradient(Jfinal, x[end])[1]

    dUkdu = [similar(A0) for k=1:length(A)]
    tmp = [similar(A0) for k=1:4]
    tmp_prod = [Matrix{T}(undef, size(x0[:,:])) for k=1:2]

    for k=Nt:-1:1
        mul!(λ[k], Uk_vec[k]', λ[k+1]) # Propage co-states backwards
        #λ[k] .+= Zygote(g, 

        #λkp1 = reinterpret(ComplexF64, λ[(Nt+2) - (k+1)])
        #xk = reinterpret(ComplexF64, x[k])

        # Compute derivative of exp(A0 + u1*A1 + u2*A2 + ...) wrt u1 = u[1,k], u2 = u[2,k], ...
        expm_jacobian!(dUkdu, A0, A, u[:,k], tmp, dUkdp_order)

        # Compute the sensitivity of J wrt u[1,k], ..
        for j=1:length(A)
            #dJdu[j, k] = sum(real(λ[k+1]' * dUkdu[j] * x[k]))
            #dJdu[j, k] = sum(real(conj(λ[k+1]) .* (dUkdu[j] * x[k]))) # Possibly λ' instead, (corresponds to trace/inner product, this seems more readable)
            mul!(tmp_prod[1], dUkdu[j], x[k]) # dx[k]/du[j]
            tmp_prod[2] .= real.(conj.(λ[k+1]) .* tmp_prod[1])
            dJdu[j, k] = sum(tmp_prod[2]) # Possibly λ' instead, (corresponds to trace/inner product, this seems more readable)
        end
    end

    return dJdu
end

function grape_naive(A0, A::Vector{<:AbstractMatrix}, Jfinal, u, x0, cache=nothing; dUkdp_order=3)

    T = eltype(complex(float(x0)))
    Nt = size(u, 2)

    if cache === nothing
        cache = setup_grape_cache(A0, T.(x0), size(u))
    end

    x, λ, dJdu, Uk_vec = cache
    cache.u .= u # To know the u used for the compuations

    x[1] .= copy(x0)

    Ak = similar(A0)

    # Propage forwards
    for k=1:Nt
        Ak .= A0
        for j=1:length(A)
            Ak .+= u[j,k] .* A[j]
        end

        Uk_vec[k] .= ExponentialUtilities._exp!(Ak, caches=cache.exp_cache)# exp(Ak)

        mul!(x[k+1], Uk_vec[k], x[k])
    end


    λ[end] .= Zygote.gradient(Jfinal, x[end])[1]
    #J, J_pullback = ChainRulesCore.rrule(Jfinal, x[end])
    #λ[end] .= J_pullback(1)[2]

    dUkdu = [similar(A0) for k=1:length(A)]
    tmp = [similar(A0) for k=1:4]
    tmp_prod = [Matrix{T}(undef, size(x0[:,:])) for k=1:2]

    for k=Nt:-1:1
        mul!(λ[k], Uk_vec[k]', λ[k+1]) # Propage co-states backwards

        λkp1 = reinterpret(ComplexF64, λ[(Nt+2) - (k+1)])
        xk = reinterpret(ComplexF64, x[k])

        # Compute derivative of exp(A0 + u1*A1 + u2*A2) wrt u1 and u2
        expm_jacobian!(dUkdu, A0, A, u[:,k], tmp, dUkdp_order)
        for j=1:length(A)
            #dJdu[j, k] = sum(real(λ[k+1]' * dUkdu[j] * x[k]))
            #dJdu[j, k] = sum(real(conj(λ[k+1]) .* (dUkdu[j] * x[k]))) # Possibly λ' instead, (corresponds to trace/inner product, this seems more readable)
            mul!(tmp_prod[1], dUkdu[j], x[k])
            tmp_prod[2] .= real.(conj.(λ[k+1]) .* tmp_prod[1])
            dJdu[j, k] = sum(tmp_prod[2]) # Possibly λ' instead, (corresponds to trace/inner product, this seems more readable)
        end
    end

    return x, λ, dJdu
end



function setup_grape_cache(A0, x0, u_size)
    T = float(eltype(x0)) # Could be both complex and real
    Tc = complex(T)
    Nt = u_size[2]

    return (x = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1],
            λ = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1],
            dJdu = Matrix{real(T)}(undef, u_size),
            Uk_vec = Matrix{Tc}[Matrix{Tc}(undef, size(x0,1), size(x0,1)) for k=1:Nt+1],
            exp_cache = Matrix{Tc}[similar(A0) for k=1:6],
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


function propagate_pwc(f, x0, u_data, Δt, cache; dt=0.1Δt)

    x_cache = cache[1]

    tgate = Δt * size(u_data, 2)

    function update_u!(integrator)
        k = Int(round(integrator.t/Δt))
        integrator.p[1] .= integrator.p[2][:, k+1]
    end
    pcb=PeriodicCallback(update_u!, 1.0, initial_affect=true, save_positions=(true,false))

    # Might need to worry about type of x0 for forward diff

    prob = ODEProblem{true}(wrap_pwc(f), x0, (0.0, tgate), callback=pcb)
    sol = solve(prob, Tsit5(), x_cache, p=([0.0; 0.0], u_data), saveat=0:Δt:tgate, adaptive=false, dt=dt)

    sol
end

setup_cache(x0, Nt) = ([similar(x0) for k=1:Nt+1], [similar(x0) for k=1:Nt+1], Matrix{float(real(eltype(x0)))}(undef, 2, Nt))
function compute_pwc_gradient(dλdt, Jfinal::Function, u, Δt, A0, A, cache; dUkdp_order=2, dt=0.1Δt)

    x, λ_store, dJdu = cache

    tgate = Δt * size(u, 2)

    # Co-states
    λfreal = Zygote.gradient(Jfinal, r2c(x[end]))[1]
    λf = c2r(λfreal)

    function update_u_bwd!(integrator)
        k = max(Int(round(integrator.t/Δt)) - 1, 0)
        integrator.p[1] .= integrator.p[2][:, k+1]
    end
    pcb_adj = PeriodicCallback(update_u_bwd!, -1.0, initial_affect=true, save_positions=(true,false)) # save_positions should maybe be other way around, but really shouldn't matter?
    prob_adj = ODEProblem{true}(wrap_pwc(dλdt), λf, (tgate, 0.0), callback=pcb_adj)
    sol_adj = solve(prob_adj, Tsit5(), λ_store, p=([0.0; 0.0], u), saveat=0:Δt:tgate, adaptive=false, dt=dt)

    # Compute loss
    λ = sol_adj.u::typeof(λ_store)

    dUkdu = [similar(A0) for k=1:length(A)]
    tmp = [similar(A0) for k=1:length(A)+1]

    Nt = size(u, 2)
    for k=Nt:-1:1
        λkp1 = reinterpret(ComplexF64, λ[(Nt+2) - (k+1)]) # At time-index k+1
        xk = reinterpret(ComplexF64, x[k])

        expm_jacobian!(dUkdu, A0, A, u[:,k], tmp, dUkdp_order)
        for j=1:2
            dJdu[j, k] = sum(real( λkp1' * dUkdu[j] * xk))
        end
    end

    #convert(Array, sol), convert(Array, sol_adj)
    sol_adj, dJdu
end

"""
    expm_jacobian!(dFdp, A0, A, p, tmp_data, approx_order=2)

Modifies `dFdp` in-place to contain the of matrix-valued derivatives of `F = exp(A0 + p[1]*A[1] + p[2]*A[2] + ...)`
with respect to p1, p2, ...
"""
function expm_jacobian!(dFdp, A0, A, p, tmp_data, approx_order=2)

    X = tmp_data[1]

    AjX = tmp_data[2]
    XAj = tmp_data[3]


    X .= A0
    for j=1:length(A)
        X .+= p[j] .* A[j]
    end

    for j=1:length(A)
        dFdp[j] .= A[j]
        if approx_order >= 2
            mul!(AjX, A[j], X)
            mul!(XAj, X, A[j])
            dFdp[j] .+= 1/2 .* (AjX .+ XAj)
        end
        if approx_order >= 3
            mul!(dFdp[j], AjX, X, 1/6, 1)
            mul!(dFdp[j], XAj, X, 1/6, 1)
            mul!(dFdp[j], X, XAj, 1/6, 1)
        end
        if approx_order >= 4
            X2 = tmp_data[4]
            mul!(X2, X, X)
            mul!(dFdp[j], AjX, X2, 1/24, 1)
            mul!(dFdp[j], XAj, X2, 1/24, 1)
            mul!(dFdp[j], X2, AjX, 1/24, 1)
            mul!(dFdp[j], X2, XAj, 1/24, 1)
        end
    end
end