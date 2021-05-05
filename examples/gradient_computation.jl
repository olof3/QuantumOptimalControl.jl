using FiniteDifferences
using Optim
using StaticArrays
using Zygote

x0 = c0
A0 = -im*1e-9*H0
A1 = -im*(Hc + Hc')/2 # Hc <-> not quite Hamiltonian
A2 = -im*(im*(Hc - Hc'))/2

u_data0 = hcat([[reim(u)...] for u in u_ctrl]...)
u_data = u_data0[:, :]

x_target = normalize!(kron([1, 0], exp.(1im*theta)))
Gfinal(x) = 1 - norm(x_target' * x) # Hmm, what is the gradient
#Gfinal(x) = (1 - norm(x_target' * x))^2
#Gfinal(x) = norm(x - x_target)^2

Gfinal2 = U -> abs(tr(subspace_target'*U))

function propagate_diffeq(dxdt, dλdt, x0, u_data, A0, A, cache; dUkdp_order=2)

    x_store, λ_store, dGdu = cache

    function wrap_f(f)
        function(dx, x, p, t)
            @inbounds @views for j=1:size(x,2)
                f(dx[:,j], x[:,j], p[1], t)
            end
        end
    end

    Δt = 1.0
    tgate = Δt * size(u_data, 2)    

    function update_u!(integrator)    
        k = Int(round(integrator.t/Δt))
        integrator.p[1] .= integrator.p[2][:, k+1]
    end
    function update_u2!(integrator)
        k = max(Int(round(integrator.t/Δt)) - 1, 0)
        integrator.p[1] .= integrator.p[2][:, k+1]
    end

    pcb=PeriodicCallback(update_u!, 1.0, initial_affect=true, save_positions=(true,false))

    T = eltype(u_data)    
    
    prob = ODEProblem{true}(wrap_f(dxdt), x0, (0.0, tgate), callback=pcb)
    sol = solve(prob, Tsit5(), x_store, p=([0.0; 0.0], u_data), saveat=0:Δt:550, adaptive=false, dt=0.1Δt)

    xf = sol.u[end]::Matrix{T}

    # Co-states
    λf = reinterpret(Float64, Zygote.gradient(Gfinal, reinterpret(ComplexF64, xf))[1])

    pcb_adj = PeriodicCallback(update_u2!, -1.0, initial_affect=true, save_positions=(true,false)) # save_positions should maybe be other way around, but really shouldn't matter?
    prob_adj = ODEProblem{true}(wrap_f(dλdt), λf, (tgate, 0.0), callback=pcb_adj)
    sol_adj = solve(prob_adj, Tsit5(), λ_store, p=([0.0; 0.0], u_data), saveat=0:Δt:550, adaptive=false, dt=0.1Δt)

    #reverse!(sol_adj.u)
    #reverse!(sol_adj.t)

    x = sol.u::typeof(x_store)
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
            dGdu[j, k] = sum(real( λk' * tmp * xk))
            #dGdu[j, k] = sum(real( real2complex(λ[(Nt+2) - (k+1)])' * tmp * real2complex(x[k])))
            #dGdu[j, k] = sum(real(λ[k+1]' * tmp * x[k]))
        end
    end

    #convert(Array, sol), convert(Array, sol_adj)
    sol, sol_adj, dGdu
end

function propagate(A0, A::Vector{<:AbstractMatrix}, u, c0)

    Nt = size(u, 2)
    T = eltype(complex(A0))
    x = Matrix{T}[Matrix{T}(undef, size(c0[:,:])...) for k=1:Nt+1]
    λ = Matrix{T}[Matrix{T}(undef, size(c0[:,:])...) for k=1:Nt+1]
    
    dGdu = Matrix{real(eltype(A0))}(undef, 2, Nt)

    Uk_vec = [similar(complex(A0)) for k=1:Nt+1]

    x[1] .= copy(c0)

    Ak = similar(A0)

    for k=1:Nt
        Ak .= A0 .+ u[1,k] .* A1 .+ u[2,k] .* A2
        
        Uk_vec[k] .= exp(Ak)

        mul!(x[k+1], Uk_vec[k], x[k])
    end

    λ[end] .= Zygote.gradient(Gfinal, x[end])[1]

    tmp = similar(A0)
    X = similar(A0)

    for k=Nt:-1:1
        mul!(λ[k], Uk_vec[k]', λ[k+1])

        X .= A0 .+ u[1,k].*A1 .+ u[2,k].*A2

        for j=1:2
            tmp .= A[j]
            mul!(tmp, A[j], X, 0.5, 1)
            mul!(tmp, X, A[j], 0.5, 1)
            mul!(tmp, A[j], X*X, 1/6, 1)
            mul!(tmp, X*A[j], X, 1/6, 1)
            mul!(tmp, X*X, A[j], 1/6, 1)
            dGdu[j, k] = sum(real(λ[k+1]' * tmp * x[k]))
        end
    end

    return x, λ, dGdu
    
end


##
#u_data2 = u_data[
@btime x, λ, dGdu = propagate(A0, [A1, A2], u_data, c0)

#@time x, λ, dGdu = propagate(SMatrix{24,24}(A0), [SMatrix{24,24}(A1), SMatrix{24,24}(A2)], u_data[:, 1:10], SVector{24}(c0))

x, λ, dGdu = propagate(A0, [A1, A2], u_data[:,1:10], c0)
display(dGdu)


# DiffEq based version
x0 = complex2real(c0)[:,:]
c0real = float(complex2real(I(24)))

Nt = size(u_data, 2)
cache = ([similar(x0) for k=1:Nt+1], [similar(x0) for k=1:Nt+1], Matrix{real(eltype(A0))}(undef, 2, Nt))

@btime x3, λ3, dGdu2 = propagate_diffeq(dxdt, dλdt, x0, u_data, A0, [A1, A2], cache)

@time x3, λ3, dGdu2 = propagate_diffeq(dxdt, dλdt, x0, u_data, A0, [A1, A2], cache, dUkdp_order=2)

Nt = 10
cache = ([similar(x0) for k=1:Nt+1], [similar(x0) for k=1:Nt+1], Matrix{real(eltype(A0))}(undef, 2, Nt))
@time x3, λ3, dGdu2 = propagate_diffeq(dxdt, dλdt, x0, u_data[:,1:10], A0, [A1, A2], cache, dUkdp_order=3)
display(dGdu2)

# Finite diff
obj = u -> Gfinal(propagate(A0, [A1, A2], u, c0)[1][end])
obj(u_data)
du_grad_fd = grad(central_fdm(12, 1), obj, u_data[:,1:10])[1]
display(du_grad_fd)

## Optim stuff

f, g! = let
    cache = ([similar(x0) for k=1:Nt+1], [similar(x0) for k=1:Nt+1], Matrix{real(eltype(A0))}(undef, 2, Nt))

    function f(u)
        u -> Gfinal(propagate(A0, A1, A2, u, c0)[1][end])
    end
    function g!(gout, u)
        gout .= propagate(A, A1, A2, u, c0)[3]
    end
end

Nt = size(u_data, 2)
cache = ([similar(x0) for k=1:Nt+1], [similar(x0) for k=1:Nt+1], Matrix{real(eltype(A0))}(undef, 2, Nt))

@btime x3, λ3, dGdu2 = propagate_diffeq(dxdt, dλdt, x0, u_data, A0, [A1, A2], cache)

@time f(u_data)

f = u -> Gfinal(real2complex(propagate_diffeq(dxdt, dλdt, x0, u, A0, [A1, A2], cache)[1][end])) # + 1e-5 * norm(u)^2
g! = (storage, u) -> storage .= propagate_diffeq(dxdt, dλdt, x0, u, A0, [A1, A2], cache, dUkdp_order=3)[3] # .+ 1e-5 * 2 * u

fold = u -> Gfinal(propagate(A0, [A1, A2], u, c0)[1][end])# + 1e-5 * norm(u)^2
gold! = (storage, u) -> storage .= propagate(A0, [A1, A2], u, c0)[3]# .+ 1e-5 * 2 * u

@time f(u_data)
@time g!(copy(u_data), u_data)

@time fold(u_data)
@time gold!(copy(u_data), u_data)


u0 = u_data .* (1 .+ randn(size(u_data)))
#u0 = randn(size(u_data))

f(u_data)
f(u0)
fold(u0)
@time res = optimize(f, g!, u0, BFGS(), Optim.Options(f_calls_limit=500))
