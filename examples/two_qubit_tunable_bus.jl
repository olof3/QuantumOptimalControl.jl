using QuantumOptimalControl
using Symbolics, DifferentialEquations
using Plots

N = 3

# DEVICE PARAMETERS
ω1 = 4.5 * 2π
ω2 = 4.2 * 2π
ωc0 = 7.5 * 2π
α1 = -0.2 * 2π
α2 = -0.2 * 2π
g1 = 0.04 * 2π
g2 = 0.04 * 2π

# OPERATORS
qb = QuantumOptimalControl.QuantumBasis([3,3,3])
a1, a2, ac = annihilation_ops(qb)

# HAMILTONIAN
Hq1 = ω1 * a1'*a1 + α1 * a1'*a1 * (a1'*a1 - I)
Hq2 = ω2 * a2'*a2 + α2 * a2'*a2 * (a2'*a2 - I) # Factor /2 after anharmonicity missing
Hi1 = g1 * (a1' + a1) * (ac' + ac)
Hi2 = g2 * (a2' + a2) * (ac' + ac)
Hc = ωc0 * ac' * ac

H0 = Hq1 + Hq2 + Hi1 + Hi2

##

# INITIAL AND TARGET STATE, RESONANCE (CZ: 11 -> 20)
x0 = qb[:, "110"]
xtarget = qb[:, "200"]

function envelope(p, t)
    t_plateau, t_rise_fall, θ0, ω_Φ, A = p
    
    δ = QuantumOptimalControl.cos_envelope(t_plateau, t_rise_fall, t)

    Φ = θ0 + A * δ * cos(ω_Φ * t)
    return sqrt(abs(cos(π * Φ)))
    #return √cos(π * θ0) - A*δ*cos(ω_Φ*t)
end

# Pulse parameters

t_plateau = 300.0
t_rise_fall = 50.0

θ0 = 0.25

i1 = qb.state_dict["110"]; i2 = qb.state_dict["200"]
ω_th = abs(H0[i1, i1] - H0[i2, i2])  # true eigenvalues are a better guess
f_offset = -0.002
ω_Φ = ω_th + f_offset * 2π  # ω_offset is given in GHz !!??!! Hmmm

A = 0.13

p0 = [t_plateau, t_rise_fall, θ0, ω_Φ, A]

##

A0 = -im*H0
A1 = -im*Hc

Ntot = N*N*N

@variables xᵣ[1:Ntot], xᵢ[1:Ntot]
@variables u

x = xᵣ + im*xᵢ
rhs = (A0 + u*A1)*x
dxdt = Symbolics.build_function(Symbolics.simplify.(c2r(rhs)), c2r(x), u, expression=Val{false})[2]

##

tlist = LinRange(0.0, t_rise_fall + t_plateau, Int(round(2 * (t_rise_fall + t_plateau))))
plot(tlist, [t -> envelope(p0, t), t -> √cos(π * θ0) - A*cos(ω_Φ*t)])

dxdt_wrapped = QuantumOptimalControl.wrap_envelope(dxdt, envelope)

Tgate = t_rise_fall + t_plateau

prob = ODEProblem{true}(dxdt_wrapped, c2r(x0), (0.0, Tgate))

@time sol = DifferentialEquations.solve(prob, Tsit5(), p=p0, adaptive=false, dt=1e-3, saveat=0:1e-3:Tgate)
GC.gc()

t = sol.t

xfinal = sol.u[end]
cost = abs2(r2c(xfinal)' * (xtarget))
println(cost) # Should be something like 0.937218


# Compute transformation
# Transform initial state, select eigenstates
