using QuantumOptimalControl

include("../examples/cavity_qubit.jl")

A0 = -im*1e-9*H0
A1 = -im*(Tc + Tc')/2
A2 = -im*(im*(Tc - Tc'))/2

u = hcat([[reim(u)...] for u in u_data]...)

x_target = normalize!(kron([1, 0], exp.(1im*theta)))
Jfinal = x -> 1 - norm(x_target' * x) # Hmm, what is the gradient

Nt = 100

@time x, λ, dJdu = QuantumOptimalControl.grape_naive(A0, [A1, A2], Jfinal, u[:,1:Nt], x0, ; dUkdp_order=3)

display(dJdu)

## DiffEq-based version

#c0real = float(complex2real(I(24)))

#Nt = size(u, 2)
cache = ([similar(c2r(x0)) for k=1:Nt+1], [similar(c2r(x0)) for k=1:Nt+1], Matrix{real(eltype(A0))}(undef, 2, Nt))

@time sol = propagate_pwc(dxdt, c2r(x0), u[:,1:Nt], 1.0, cache)
#@time sol_adj, dJdu = QuantumOptimalControl.compute_pwc_gradient(dλdt, Jfinal, u[:,1:Nt], 1.0, A0, A, cache; dUkdp_order=3)
@time sol_adj, dJdu2 = compute_pwc_gradient(dλdt, Jfinal, u[:,1:Nt], 1.0, A0, [A1, A2], cache; dUkdp_order=3)
display(dJdu2)


## Finite diff




using FiniteDiff
obj = u -> Jfinal(r2c(propagate_pwc(dxdt, c2r(x0), u, 1.0, cache).u[end]))

dJdu3 = FiniteDiff.finite_difference_gradient(obj, u[:, 1:100])
display(dJdu3)


obj = u -> Jfinal(QuantumOptimalControl.grape_naive(A0, [A1, A2], Jfinal, u, x0, ; dUkdp_order=3)[1][end])
dJdu4 = FiniteDiff.finite_difference_gradient(obj, u[:, 1:Nt])
display(dJdu4)







#=
Nt = 1

A0 = im*[-1 2; 1 -3]
A1 = [0 1; -1 0]
A2 = [0 1; 1 0]
x0 = 1.0Matrix(I, 2, 2)
x_target = [0 1.0; 1.0 0]
Jfinal = x -> 1 - abs(tr(x_target' * x))/4
x, λ, dJdu = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u_mat[:,1:Nt], x0; dUkdp_order=3)

obj = u -> Jfinal(QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u, x0; dUkdp_order=3)[1][end])
obj2 = x0 -> Jfinal(QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u_mat[:,1:Nt], x0; dUkdp_order=3)[1][end])
[obj(u_mat[:,1:Nt]) Jfinal(x[end])]
dJdu4 = FiniteDiff.finite_difference_gradient(obj, u_mat[:,1:Nt])

FiniteDiff.finite_difference_gradient(Jfinal, x[end])

FiniteDiff.finite_difference_gradient(, x[end])

dJdx0 = FiniteDiff.finite_difference_gradient(obj2, complex(x0))

println(dJdu)
println(dJdu4)

FiniteDiff.finite_difference_gradient(u_mat -> Jfinal(exp((A0 + u_mat[1,1]*A1 + u_mat[2,1]*A2)*Δt)), u_mat[:,1])

FiniteDiff.finite_difference_gradient(u_mat -> Jfinal(exp((A0 + u_mat[1,1]*A1 + u_mat[2,1]*A2)*Δt)), u_mat[:,1])

dU10 = FiniteDiff.finite_difference_derivative(u1 -> exp((A0 + u1*A1 + u_mat[2,1]*A2)*Δt), u_mat[1,1])
dU20 = FiniteDiff.finite_difference_derivative(u2 -> exp((A0 + u_mat[1,1]*A1 + u2*A2)*Δt), u_mat[2,1])
real(sum(conj(λ[end]) .* dU10))
real(sum(conj(λ[end]) .* dU20))

dUkdu = [similar(1.0A0) for k=1:2]
tmp = [similar(1.0A0) for k=1:3]
QuantumOptimalControl.expm_jacobian!(dUkdu, Δt*A0, [Δt*A1, Δt*A2], u_mat[:,1], tmp, 3)
=#

    


#=
using FiniteDifferences

Nt = 30

obj = u -> Jfinal(QuantumOptimalControl.grape_naive(Δt*A0, [Δt*A1, Δt*A2], Jfinal, u, x0, ; dUkdp_order=3)[1][end])

dJdu3 = FiniteDiff.finite_difference_gradient(obj, u_mat[:, 1:Nt])
dJdu4 = FiniteDifferences.grad(central_fdm(7,1,max_range=1e-8), obj, u_mat[:, 1:Nt])[1]

x, λ, dJdu = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u_mat[:,1:Nt], x0; dUkdp_order=3)

r = FiniteDiff.finite_difference_gradient(Jfinal, x[end])

Jfinal(x[end])

lamba_end = Zygote.gradient(Jfinal, x[end])[1]
lamba_end = FiniteDiff.finite_difference_gradient(Jfinal, x[end])

cache = ([similar(c2r(x0)) for k=1:Nt+1], [similar(c2r(x0)) for k=1:Nt+1], Matrix{real(eltype(A0))}(undef, 2, Nt))

@time sol = propagate_pwc(dxdt, c2r(x0), u_mat[:,1:500], Δt, cache)
#@time sol_adj, dJdu = QuantumOptimalControl.compute_pwc_gradient(dλdt, Jfinal, u[:,1:Nt], Δt, A0, A, cache; dUkdp_order=3)
@time sol_adj, dJdu2 = compute_pwc_gradient(dλdt, Jfinal, u_mat[:,1:Nt], Δt, A0, [A1, A2], cache; dUkdp_order=3)
display(dJdu2)

display(dJdu4)



x, λ, dJdu = QuantumOptimalControl.grape_naive(A0, [A1, A2], Jfinal, u_mat[:, 1:Nt], x0; dUkdp_order=3)
display(dJdu)
=#
