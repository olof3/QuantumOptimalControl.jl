using Zygote, Test

inds_css = [1,2,5,6]
inds_penalty = [7,8,9]


L2 = x -> 0.22*norm(x[inds_penalty, inds_css])^2

L, dL_dx = QuantumOptimalControl.setup_state_penalty(inds_penalty, inds_css, 0.22)

x0 = reshape(1.0:81, 9, 9)

@test L(x0) == L2(x0)

grad1 = dL_dx(x0)
grad2 = Zygote.gradient(L2, x0)[1]

@test grad1 ≈ grad2 rtol=1e-15

x_target = qr(randn(ComplexF64, 9, 8)).Q[:,1:4]
x = randn(ComplexF64, 9, 4)



Jf, dJf_dx = QuantumOptimalControl.setup_infidelity(x_target)

grad1 = dJf_dx(x)
grad2 = Zygote.gradient(Jf, x)[1]

@test grad1 == grad2


J, dJ_dx = QuantumOptimalControl.setup_infidelity_zcalibrated(x_target)

J(x)

grad1 = dJ_dx(x)
grad2 = Zygote.gradient(J, x)[1]

@test grad1 == grad2