using FiniteDiff
using Random
Random.seed!(0)

A0 = 0.05*randn(3,3)
A1 = 0.05*randn(3,3)
A2 = 0.05*randn(3,3)
u = [1.0, 2.0]

dFdp = [similar(A0) for k=1:2]
exp_jac_cache = [similar(A0) for k=1:4]

dFdp_fd = FiniteDiff.finite_difference_jacobian(u -> exp(A0 .+ u[1].*A1 + u[2].*A2)[:], u)


QuantumOptimalControl.expm_jacobian!(dFdp, A0, [A1, A2], u, exp_jac_cache, 3)
dFdp_reshaped = hcat(vec.(dFdp)...)
@test norm(dFdp_reshaped - dFdp_fd) < 4e-4

QuantumOptimalControl.expm_jacobian!(dFdp, A0, [A1, A2], u, exp_jac_cache, 4)
dFdp_reshaped = hcat(vec.(dFdp)...)
@test norm(dFdp_reshaped - dFdp_fd) < 3e-5


# Test with a step Δt of non-unit length
Δt = 0.25
dFdp_fd = FiniteDiff.finite_difference_jacobian(u -> exp(Δt*(A0 .+ u[1].*A1 + u[2].*A2))[:], u)

QuantumOptimalControl.expm_jacobian!(dFdp, A0, [A1, A2], u, exp_jac_cache, 3, Δt=Δt)
dFdp_reshaped = hcat(vec.(dFdp)...)
@test norm(dFdp_reshaped - dFdp_fd) < 2e-6

QuantumOptimalControl.expm_jacobian!(dFdp, A0, [A1, A2], u, exp_jac_cache, 4, Δt=Δt)
dFdp_reshaped = hcat(vec.(dFdp)...)
@test norm(dFdp_reshaped - dFdp_fd) < 3e-8