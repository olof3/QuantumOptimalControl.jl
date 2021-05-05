using FiniteDifferences
using Optim
using StaticArrays
using Zygote

x0 = c0
A0 = -im*1e-9*H0
A1 = -im*(Hc + Hc')/2
A2 = -im*(im*(Hc - Hc'))/2

u_data0 = hcat([[reim(u)...] for u in u_ctrl]...)
u_data = 1e-9*u_data0[:, :]

x_target = normalize!(kron([1, 0], exp.(1im*theta)))
Gfinal(x) = 1 - norm(x_target' * x) # Hmm, what is the gradient
Gfinal(x) = (1 - norm(x_target' * x))^2
Gfinal(x) = norm(x - x_target)^2



#Gfinal(x) = norm(diag(full_target_operation)' * x)



function propagate(A0, A::Vector{<:AbstractMatrix}, u, c0)

    Nt = size(u, 2)
    T = eltype(complex(A0))
    x = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1]
    λ = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1]
    
    dGdu = Matrix{real(eltype(A0))}(undef, 2, Nt)

    Uk_vec = [similar(complex(A0)) for k=1:Nt+1]

    x[1] .= copy(c0)

    Ak = similar(A0)

    @time for k=1:Nt
        Ak .= A0 .+ u[1,k] .* A1 .+ u[2,k] .* A2
        
        Uk_vec[k] .= exp(Ak)

        mul!(x[k+1], Uk_vec[k], x[k])
    end

    #return x

    #λ[end] .= diag(full_target_operation)[:,:]
    #λ[end] .= -x_target
    λ[end] .= Zygote.gradient(Gfinal, x[end])[1]
    #λ[end] .= -x_target * 2 * (1 - norm(x_target' * x[end]))
    #λ[end] .= 2 * conj(x_target) * (1 - dot(x_target, x[end]))
    #λ[end] .= 2(x[end] - x_target)

    tmp = similar(A0)
    X = similar(A0)
    
    for k=Nt:-1:1
        mul!(λ[k], Uk_vec[k]', λ[k+1])

        X .= A0 .+ u[1,k].*A1 .+ u[2,k].*A2
            
        #dGdu[1, k] = sum(real(λ[k+1]' * (A1 + 0.5(A1*X + X*A1) + (A1*X*X + X*A1*X + X*X*A1)/6) * x[k]))
        #dGdu[2, k] = sum(real(λ[k+1]' * (A2 + 0.5(A2*X + X*A2) + (A2*X*X + X*A2*X + X*X*A2)/6) * x[k]))
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
u_data2 = u_data[:]
@time x, λ, dGdu = propagate(A0, [A1, A2], reshape(u_data2, size(u_data)), c0)

#@time x, λ, dGdu = propagate(SMatrix{24,24}(A0), [SMatrix{24,24}(A1), SMatrix{24,24}(A2)], u_data[:, 1:10], SVector{24}(c0))

@time x, λ, dGdu = propagate(A0, [A1, A2], u_data[:,1:10], c0)
display(dGdu)

obj = u -> Gfinal(propagate(A0, [A1, A2], u, c0)[1][end])
obj(u_data)
du_grad_fd = grad(central_fdm(12, 1), obj, u_data[:,1:10])[1]
display(du_grad_fd)



f = u -> Gfinal(propagate(A0, A1, A2, u, c0)[1][end])# + 1e-5 * norm(u)^2
g! = (storage, u) -> storage .= propagate(A, A1, A2, u, c0)[3]# .+ 1e-5 * 2 * u

f(u_data)
f(u0)

u0 = u_data .* (1 .+ randn(size(u_data)))

f(u0)
@time res = optimize(f, g!, u0, BFGS(), Optim.Options(f_calls_limit=10))

X0 = randn(3,3)
obj = X -> norm(X - I)
@time res = optimize(obj, X0, GradientDescent())

res.minimizer
obj(X0)

@code_warntype propagate(A, A1, A2, u_data, c0)

@btime propagate(A, A1, A2, u_data, c0)




##

prob = ODEProblem{true}(dxdt, complex2real(c0), (0.0, 1e-9))
p0 = [1e7; 1e7]
@btime sol = solve(prob, Tsit5(), p=p0, saveat=[1e-9], adaptive=false, dt=5e-10)

@time sol1 = solve(prob, Tsit5(), p=p0, saveat=0:2e-10:1e-9, adaptive=false, dt=2e-10)
@time sol2 = solve(prob, Tsit5(), prealloc, p=p0, saveat=0:2e-10:1e-9, adaptive=false, dt=2e-10)

prealloc = deepcopy(sol.u)

@time x1 = exp(A + u_data[1, 4]*A1 + u_data[2, 4]*A2)*c0_cavity

[real2complex(sol.u[1]) x1]