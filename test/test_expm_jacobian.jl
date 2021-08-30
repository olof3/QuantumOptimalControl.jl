A0 = 0.05*randn(3,3)
A1 = 0.05*randn(3,3)
A2 = 0.05*randn(3,3)

dFdp = [similar(A0) for k=1:2]
tmp = [similar(A0) for k=1:3]


u = [1.0, 2.0]
QuantumOptimalControl.expm_jacobian!(dFdp, A0, [A1, A2], u, tmp, 3)

fexpm = u -> exp(A0 .+ u[1].*A1 + u[2].*A2)

dFdp1_approx = (fexpm(u + [1e-5, 0]) - fexpm(u)) / 1e-5
#display(dFdp[1])
println(norm(dFdp[1] - dFdp1_approx))

dFdp2_approx = (fexpm(u + [0, 1e-5]) - fexpm(u)) / 1e-5

println(norm(dFdp[2] - dFdp2_approx))






using FiniteDiff


