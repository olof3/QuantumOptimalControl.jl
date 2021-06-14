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
a1, a2, ac = QuantumOptimalControl.annihilation_op(3, 3, 3)

# HAMILTONIAN
Hq1 = ω1 * a1'*a1 + α1 * a1'*a1 * (a1'*a1 - I)
Hq2 = ω2 * a2'*a2 + α2 * a2'*a2 * (a2'*a2 - I)
Hi1 = g1 * (a1' + a1) * (ac' + ac)
Hi2 = g2 * (a2' + a2) * (ac' + ac)
Hc = ωc0 * ac' * ac

H0 = Hq1 + Hq2 + Hi1 + Hi2

state_labels = map(e -> string("|", reverse(e)..., "⟩"), collect(Iterators.product(0:2, 0:2, 0:2))[:])
sdict = Dict(kron([string.(0:N-1) for N in (3,3,3)]...) .=> 1:27)


##

# Compute transformation
# Transform initial state, select eigenstates

# INITIAL AND TARGET STATE, RESONANCE (CZ: 11 -> 20)
initialstate = complex(float(kron([0, 1, 0], [0, 1, 0], [1, 0, 0]))) # |110>
targetstate = kron([0, 0, 1], [1, 0, 0], [1, 0, 0]) # |200>

#initialstate = kron(qt.basis(N, 1), qt.basis(N, 1), qt.basis(N, 0))
#targetstate = kron(qt.basis(N, 2), qt.basis(N, 0), qt.basis(N, 0))

function envelope(p, t)
    t_plateau, t_rise_fall, θ0, ω_Φ, A = p
    
    δ = QuantumOptimalControl.cos_envelope(t_plateau, t_rise_fall, t)

    Φ = θ0 + A * δ * cos(ω_Φ * t)
    return sqrt(abs(cos(π * Φ))) # θ + δ *cos(ω_Φ * t)   
    #return √cos(π * θ0) - δ*cos(ω_Φ*t)
end

# Pulse parameters

t_plateau = 300.0
t_rise_fall = 50.0

θ0 = 0.25

ω_th = abs(H0[1 * N * N + 1 * N + 1, 1 * N * N + 1 * N + 1] - H0[2 * N * N + 1, 2 * N * N + 1])  # true eigenvalues are a better guess
f_offset = -0.002
ω_Φ = ω_th + f_offset * 2π  # ω_offset is given in GHz !!??!! Hmmm

δ = 0.13

p0 = [t_plateau, t_rise_fall, θ0, ω_Φ, δ]

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
plot(tlist, [t -> envelope(p0, t), t -> √cos(π * θ0) - δ*cos(ω_Φ*t)])

dxdt_wrapped = QuantumOptimalControl.wrap_envelope(dxdt, envelope)

Tgate = t_rise_fall + t_plateau

prob = ODEProblem{true}(dxdt_wrapped, c2r(initialstate), (0.0, Tgate))

@btime sol = DifferentialEquations.solve(prob, Tsit5(), p=p0, adaptive=false, dt=1e-3, saveat=0:1e-3:Tgate)
GC.gc()

t = sol.t
@time solc = hcat(r2c.(sol.u)...)

finalstate = sol.u[end]
cost = abs2(r2c(finalstate)' * (targetstate))
println(cost) # Should be something like 0.937218