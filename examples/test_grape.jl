using FiniteDifferences
using Optim

x0 = c0
A = -im*1e-9*H0
A1 = -im*(Hc + Hc')/2
A2 = -im*(im*(Hc - Hc'))/2

u_data0 = hcat([[reim(u)...] for u in u_ctrl]...)
u_data = 1e-9*u_data0[:, :]

x_target = normalize!(kron([1, 0], exp.(1im*theta)))
Gfinal(x) = 1 - norm(x_target' * x)

Gfinal(x) = norm(x - x_target)^2
#Gfinal(x) = norm(diag(full_target_operation)' * x)
obj = u -> Gfinal(propagate(A, [A1, A2], u, c0)[1][end])

function propagate(A0, A::Vector{<:Matrix}, u, c0)

    Nt = size(u, 2)
    T = eltype(complex(A0))
    x = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1]
    λ = Matrix{T}[Matrix{T}(undef, size(x0[:,:])...) for k=1:Nt+1]
    
    dGdu = Matrix{real(eltype(A0))}(undef, 2, Nt)

    Uk_vec = [similar(complex(A0)) for k=1:Nt+1]

    x[1] = copy(c0[:,:])

    Ak = similar(A0)

    for k=1:Nt
        Ak .= A0 .+ u[1,k] .* A1 .+ u[2,k] .* A2
        
        Uk_vec[k] .= exp(Ak)

        mul!(x[k+1], Uk_vec[k], x[k])
    end

    #return x

    #λ[end] .= diag(full_target_operation)[:,:]
    #λ[end] .= -x_target
    λ[end] .= 2(x[end] - x_target)

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


u_data2 = u_data[:]
@time x, λ, dGdu = propagate(A, [A1, A2], reshape(u_data2, size(u_data)), c0)

@time x, λ, dGdu = propagate(A, [A1, A2], u_data[:,1:10], c0)
display(dGdu)

obj(u_data)
du_grad_fd = grad(central_fdm(12, 1), obj, u_data[:,1:10])[1]
display(du_grad_fd)



f = u -> Gfinal(propagate(A, A1, A2, u, c0)[1][end])# + 1e-5 * norm(u)^2
g! = (storage, u) -> storage .= propagate(A, A1, A2, u, c0)[3]# .+ 1e-5 * 2 * u

f(u_data)

g!(copy(u_data[:, 1:3]), u_data[:, 1:3])
propagate(A, A1, A2, u_data[:,1:3], c0)[3]
grad(central_fdm(12, 1), f, u_data[:,1:3])[1]

u0 = 0.01*randn(2, 100)

f(u_data)

u0 = u_data .* (1 .+ randn(size(u_data)))
f(u0)
optimize(f, g!, u0, GradientDescent(), Optim.Options(f_calls_limit=100))

X0 = randn(3,3)
obj = X -> norm(X - I)
@time res = optimize(obj, X0, GradientDescent())

res.minimizer
obj(X0)

@code_warntype propagate(A, A1, A2, u_data, c0)

@btime propagate(A, A1, A2, u_data, c0)


