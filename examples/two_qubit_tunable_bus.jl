using QuantumOptimalControl
using Symbolics, OrdinaryDiffEq
using Plots

include("models/two_qubit_tunable_bus.jl")
include("../examples/models/setup_diffeq_rhs.jl")

## parameterization of target

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

## Plot the drive signal
# tlist = LinRange(0.0, t_rise_fall + t_plateau, Int(round(2 * (t_rise_fall + t_plateau))))
# plot(tlist, [t -> envelope(p0, t), t -> √cos(π * θ0) - A*cos(ω_Φ*t)])

##

A0 = -im*H0
A1 = -im*Hc

dxdt = setup_dxdt(A0, [A1])


##

# INITIAL AND TARGET STATE, RESONANCE (CZ: 11 -> 20)
x0 = qb[:, "110"]
xtarget = qb[:, "200"]

dxdt_wrapped = QuantumOptimalControl.wrap_envelope(dxdt, envelope)

tgate = t_rise_fall + t_plateau

prob = ODEProblem{true}(dxdt_wrapped, c2r(x0), (0.0, tgate))

@time sol = solve(prob, Tsit5(), p=p0, adaptive=false, dt=1e-3, saveat=0:10e-3:tgate)
GC.gc()

t = sol.t

xfinal = sol.u[end]
cost = abs2(r2c(xfinal)' * (xtarget))
println(cost) # Should be something like 0.937218
