function grape_naive(A0, A::Vector{<:AbstractMatrix}, Jfinal, u, x0; dUkdp_order=3)

    Nt = size(u, 2)
    T = eltype(complex(A0))
    x = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1]
    λ = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1]

    dJdu = Matrix{real(eltype(A0))}(undef, 2, Nt)

    Uk_vec = [similar(complex(A0)) for k=1:Nt+1]

    x[1] .= copy(x0)

    Ak = similar(A0)

    # Propage forwards
    for k=1:Nt
        Ak .= A0
        for j=1:length(A)
            Ak .+= u[j,k] .* A[j]
        end

        Uk_vec[k] .= exp(Ak)

        mul!(x[k+1], Uk_vec[k], x[k])
    end

    λ[end] .= Zygote.gradient(Jfinal, x[end])[1]


    dUkdu = [similar(A0) for k=1:length(A)]
    tmp = [similar(A0) for k=1:3]

    for k=Nt:-1:1
        mul!(λ[k], Uk_vec[k]', λ[k+1]) # Propage co-states backwards

        λkp1 = reinterpret(ComplexF64, λ[(Nt+2) - (k+1)])
        xk = reinterpret(ComplexF64, x[k])

        # Compute derivative of exp(A0 + u1*A1 + u2*A2) wrt u1 and u2
        expm_jacobian!(dUkdu, A0, A, u[:,k], tmp, dUkdp_order)
        for j=1:length(A)
            dJdu[j, k] = sum(real(λ[k+1]' * dUkdu[j] * x[k]))
        end
    end

    return x, λ, dJdu
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

    prob = ODEProblem{true}(wrap_f(f), x0, (0.0, tgate), callback=pcb)
    sol = solve(prob, Tsit5(), x_cache, p=([0.0; 0.0], u_data), saveat=0:Δt:tgate, adaptive=false, dt=dt)

    sol
end

setup_cache(x0, Nt) = ([similar(x0) for k=1:Nt+1], [similar(x0) for k=1:Nt+1], Matrix{float(real(eltype(x0)))}(undef, 2, Nt))
function compute_pwc_gradient(dλdt, Jfinal::Function, u, Δt, A0, A, cache; dUkdp_order=2)

    x, λ_store, dJdu = cache

    tgate = Δt * size(u, 2)

    #x = sol.u[end]::Matrix{T}

    # Co-states
    λf = reinterpret(Float64, Zygote.gradient(Jfinal, reinterpret(ComplexF64, x[end]))[1])

    function update_u_bwd!(integrator)
        k = max(Int(round(integrator.t/Δt)) - 1, 0)
        integrator.p[1] .= integrator.p[2][:, k+1]
    end
    pcb_adj = PeriodicCallback(update_u_bwd!, -1.0, initial_affect=true, save_positions=(true,false)) # save_positions should maybe be other way around, but really shouldn't matter?
    prob_adj = ODEProblem{true}(wrap_f(dλdt), λf, (tgate, 0.0), callback=pcb_adj)
    sol_adj = solve(prob_adj, Tsit5(), λ_store, p=([0.0; 0.0], u), saveat=0:Δt:tgate, adaptive=false, dt=0.1Δt)


    # Compute loss
    λ = sol_adj.u::typeof(λ_store)

    dUkdu = [similar(A0) for k=1:length(A)]
    tmp = [similar(A0) for k=1:length(A)+1]

    Nt = size(u, 2)
    for k=Nt:-1:1
        λkp1 = reinterpret(ComplexF64, λ[(Nt+2) - (k+1)])
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
Differentiate an exponential matrix of the form
    `F = exp(A0 + p[1]*A[1] + p[2]*A[2] + ...)`
with respect to p1 and p2
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
    end
end