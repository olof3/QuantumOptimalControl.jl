using DelimitedFiles

using OrdinaryDiffEq, DiffEqCallbacks
using Symbolics

using SparseArrays

using QuantumOptimalControl

include("models/cavity_qubit_model.jl")

# Optimization parameters
#max_rabi_rate = 2π* 40e6
#cutoff_frequency = 2π* 50e6

examples_dir = dirname(Base.find_package("QuantumOptimalControl"))
iq_data = DelimitedFiles.readdlm(joinpath(examples_dir, "../examples/cavity_qubit_control.csv"))
u_data = 1e-9 * [Complex(r...) for r in eachrow(iq_data)] # Go over to GHz

##

Ntot = N_qubit * N_cavity
@variables xᵣ[1:Ntot], xᵢ[1:Ntot]
xᵣ, xᵢ = [xᵣ...], [xᵢ...]
@variables uᵣ uᵢ

u = uᵣ + im*uᵢ
x = xᵣ + im*xᵢ

Htot = H0 + u*Tc/2 + conj(u)*Tc'/2

A0 = -im*H0
A1 = -im*(Tc + Tc')
A2 = -im*(im*(Tc - Tc'))
Htot2 = Tc + Tc'

rhs = simplify.(-1im * (Htot * x))
dxdt = Symbolics.build_function(Symbolics.simplify.(c2r(rhs)), c2r(x), c2r(u), expression=Val{false})[2]

@variables λᵣ[1:Ntot] λᵢ[1:Ntot]
λᵣ, λᵢ = [λᵣ...], [λᵢ...]
λ = λᵣ + im*λᵢ

rhs = simplify.(-1im * (Htot' * λ))
dλdt = Symbolics.build_function(Symbolics.simplify.(c2r(rhs)), c2r(λ), c2r(u), expression=Val{false})[2]

##

function wrap_f(f)
    function(dx, x, p, t)        
        f(dx, x, p[1], t)
    end
end


Δt = 1.0
tgate = 550 * Δt

function update_u!(integrator)
    k = Int(round(integrator.t/Δt))
    integrator.p[1][1] = real(integrator.p[2][k+1])
    integrator.p[1][2] = imag(integrator.p[2][k+1])    
end

pcb=PeriodicCallback(update_u!, Δt, initial_affect=true, save_positions=(true,false))

prob = ODEProblem{true}(wrap_f(dxdt), c2r(x0), (0.0, tgate), callback=pcb)

sol = solve(prob, Tsit5(), p=([0.0; 0.0], u_data), saveat=0:Δt:tgate, adaptive=false, dt=0.1Δt)

t = sol.t
x = hcat([r2c(u) for u in sol.u]...)

abs(real2complex(sol.u[end])' * normalize!(diag(subspace_target)))

# using Plots
# plt1 = plot(sol.t, abs2.(x[1:11,:]'))
