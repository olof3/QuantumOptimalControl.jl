function grape_naive(A0, A::Vector{<:AbstractMatrix}, Jfinal, u, x0)

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
        Ak .= A0 .+ u[1,k] .* A[1] .+ u[2,k] .* A[2] # Fixme -> sum
        
        Uk_vec[k] .= exp(Ak)

        mul!(x[k+1], Uk_vec[k], x[k])
    end

    λ[end] .= Zygote.gradient(Jfinal, x[end])[1]

    tmp, X = similar(A0), similar(A0)    

    for k=Nt:-1:1
        mul!(λ[k], Uk_vec[k]', λ[k+1]) # Propage co-states backwards

        # Compute derivative of exp(A0 + u1*A1 + u2*A2) wrt u1 and u2
        X .= A0 .+ u[1,k].*A[1] .+ u[2,k].*A[2]
        for j=1:2
            tmp .= A[j]
            mul!(tmp, A[j], X, 0.5, 1)
            mul!(tmp, X, A[j], 0.5, 1)
            mul!(tmp, A[j], X*X, 1/6, 1)
            mul!(tmp, X*A[j], X, 1/6, 1)
            mul!(tmp, X*X, A[j], 1/6, 1)
            dJdu[j, k] = sum(real(λ[k+1]' * tmp * x[k]))
        end
    end

    return x, λ, dJdu
end






function propagate_pwc(f, x0, u_data, Δt, cache; dt=0.1Δt)

    x_store = cache[1]
    
    tgate = Δt * size(u_data, 2)    

    function update_u!(integrator)    
        k = Int(round(integrator.t/Δt))
        integrator.p[1] .= integrator.p[2][:, k+1]
    end
    pcb=PeriodicCallback(update_u!, 1.0, initial_affect=true, save_positions=(true,false))

    # Might need to worry about type of x0 for forward diff
    
    prob = ODEProblem{true}(wrap_f(f), x0, (0.0, tgate), callback=pcb)
    sol = solve(prob, Tsit5(), x_store, p=([0.0; 0.0], u_data), saveat=0:Δt:tgate, adaptive=false, dt=dt)

    sol
end


function compute_pwc_gradient(dλdt, Jfinal::Function, u_data, Δt, A0, A, cache; dUkdp_order=2)

    x, λ_store, dJdu = cache

    tgate = Δt * size(u_data, 2)
    
    #x = sol.u[end]::Matrix{T}

    # Co-states
    λf = reinterpret(Float64, Zygote.gradient(Jfinal, reinterpret(ComplexF64, x[end]))[1])

    function update_u_bwd!(integrator)
        k = max(Int(round(integrator.t/Δt)) - 1, 0)
        integrator.p[1] .= integrator.p[2][:, k+1]
    end
    pcb_adj = PeriodicCallback(update_u_bwd!, -1.0, initial_affect=true, save_positions=(true,false)) # save_positions should maybe be other way around, but really shouldn't matter?
    prob_adj = ODEProblem{true}(wrap_f(dλdt), λf, (tgate, 0.0), callback=pcb_adj)
    sol_adj = solve(prob_adj, Tsit5(), λ_store, p=([0.0; 0.0], u_data), saveat=0:Δt:tgate, adaptive=false, dt=0.1Δt)


    # Compute loss
    λ = sol_adj.u::typeof(λ_store)

    tmp = similar(A0)
    X = similar(A0)

    X2 = similar(A0)    
    AjX = [similar(A0), similar(A0)]    

    Nt = size(u_data, 2)
    for k=Nt:-1:1      
        X .= A0 .+ u_data[1,k].*A[1] .+ u_data[2,k].*A[2]
        
        if dUkdp_order >= 3; mul!(X2, X, X); end
            
        for j=1:2
            
            tmp .= A[j]
            if dUkdp_order >= 2
                mul!(tmp, A[j], X, 0.5, 1)
                mul!(tmp, X, A[j], 0.5, 1)
            end
            if dUkdp_order >= 3
                #mul!(AjX[j], A[j], X)
                mul!(tmp, A[j], X2, 1/6, 1)
                mul!(tmp, X*A[j], X, 1/6, 1)
                mul!(tmp, X2, A[j], 1/6, 1)
            end
            λk = reinterpret(ComplexF64, λ[(Nt+2) - (k+1)])            
            xk = reinterpret(ComplexF64, x[k])            
            dJdu[j, k] = sum(real( λk' * tmp * xk))
        end
    end

    #convert(Array, sol), convert(Array, sol_adj)
    sol_adj, dJdu
end

