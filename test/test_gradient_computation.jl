using QuantumOptimalControl

x0 = c0
A0 = -im*1e-9*H0
A1 = -im*(Tc + Tc')/2
A2 = -im*(im*(Tc - Tc'))/2

u_data0 = hcat([[reim(u)...] for u in u_ctrl]...)
u_data = u_data0[:, :]

x_target = normalize!(kron([1, 0], exp.(1im*theta)))
Jfinal = x -> 1 - norm(x_target' * x) # Hmm, what is the gradient

@time x, λ, dJdu = QuantumOptimalControl.grape_naive(A0, [A1, A2], Jfinal, u_data[:,1:100], c0, ; dUkdp_order=3)

display(dJdu)

## DiffEq-based version
x0 = complex2real(c0)[:,:]
#c0real = float(complex2real(I(24)))

#Nt = size(u_data, 2)
Nt = 100
cache = ([similar(x0) for k=1:Nt+1], [similar(x0) for k=1:Nt+1], Matrix{real(eltype(A0))}(undef, 2, Nt))

@time sol = propagate_pwc(dxdt, x0, u_data[:,1:Nt], 1.0, cache)
#@time sol_adj, dJdu = QuantumOptimalControl.compute_pwc_gradient(dλdt, Jfinal, u_data[:,1:Nt], 1.0, A0, A, cache; dUkdp_order=3)
@time sol_adj, dJdu2 = compute_pwc_gradient(dλdt, Jfinal, u_data[:,1:Nt], 1.0, A0, [A1, A2], cache; dUkdp_order=3)
display(dJdu2)



## Finite diff

using FiniteDiff
obj = u -> Jfinal(real2complex(propagate_pwc(dxdt, x0, u, 1.0, cache).u[end]))
obj(u_data[:, 1:100])

dJdu3 = FiniteDiff.finite_difference_gradient(obj, u_data[:, 1:100])
display(dJdu3)


    