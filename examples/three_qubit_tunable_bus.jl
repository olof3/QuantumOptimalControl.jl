a0, a1, a2, ac_1, ac_2 = QuantumOptimalControl.annihilation_op2(3, 3, 3, 3, 3)

ω0 = 4.0 * 2π
ω1 = 4.5 * 2π
ω2 = 4.2 * 2π
ωc0_1 = 7.5 * 2π
ωc0_2 = 7.5 * 2π
α0 = -0.2 * 2π
α1 = -0.2 * 2π
α2 = -0.2 * 2π
αc_1 = 0.0 * 2π
αc_2 = 0.0 * 2π
g01 = 0.04 * 2π
g11 = 0.04 * 2π
g02 = 0.04 * 2π
g22 = 0.04 * 2π

Hq0 = ω0*a0'*a0 + 0.5*α0*a0'*a0*(a0'*a0 - I)
Hq1 = ω1*a1'*a1 + 0.5*α1*a1'*a1*(a1'*a1 - I)
Hq2 = ω2*a2'*a2 + 0.5*α2*a2'*a2*(a2'*a2 - I)
Hcc = 0.5*αc_1*ac_1'*ac_1*(ac_1'*ac_1 - I) + 0.5*αc_2*ac_2'*ac_2*(ac_2'*ac_2 - I)
Hi1 = (g01*(a0' + a0) + g11*(a1' + a1)) * (ac_1' + ac_1)
Hi2 = (g02*(a0' + a0) + g22*(a2' + a2)) * (ac_2' + ac_2)
Hq = Hq0 + Hq1 + Hq2 + Hcc
Hi = Hi1 + Hi2

H0 = Hq + Hi

Hc = [ωc0_1*ac_1'*ac_1, ωc0_2*ac_2'*ac_2]




Ntot = size(Hq,1)

@parameters t u1 u2
@variables cᵣ[1:Ntot](t) cᵢ[1:Ntot](t)
D = Differential(t)
c = cᵣ + im*cᵢ

Htot = sparse(H0 + u1*Hc[1] + u2*Hc[2])

symrhs = -1im * (Htot * c)
symrhs = simplify.(symrhs)
symlhs = D.(c)

eqs = symlhs .~ symrhs

eqs_vec = vcat(eqs...) # Since complex-valued equations become a Vector of two equations
sys = ODESystem(eqs_vec, t, complex2real(c[:]), [u1, u2])

@time dxdt = generate_function(sys; expression=Val{false})[2]



dx = randn(2*Ntot)
x = randn(2*Ntot)

@btime dxdt(dx, x, (1.4, 1.2), 3)

function wrap_control(dxdt, u_fcn1, u_fcn2)
    return (dx, x, p, t) -> dxdt(dx, x, (u_fcn1(t,p), u_fcn2(t,p)), t)
end
dxdt_wrapped = wrap_control(dxdt, envelope, envelope)



prob = ODEProblem{true}(dxdt_wrapped, [1; zeros(2*Ntot-1)], (0.0, 550))
#@time sol = solve(prob, Tsit5(), p=p0, saveat=tlist, adaptive=false, dt=0.001)

sol = solve(prob, Tsit5(), p=p0, saveat=tlist, adaptive=false, dt=1e-3, atol=1e-10, rtol=1e-10)



# 1.089e-6 * 6 * 1000 * 550
# 6 * 1000 * 550