using DelimitedFiles

using DifferentialEquations
using Symbolics

using SparseArrays

using QuantumOptimalControl

using Plots

# Optimization parameters
N_qubit = 2 # nuber of transmon levels
N_cavity = 12 # nuber of cavity levels

examples_dir = dirname(Base.find_package("QuantumOptimalControl"))
iq_data = DelimitedFiles.readdlm(joinpath(examples_dir, "../examples/cavity_qubit_control.csv"))
u_data = 1e-9 * [Complex(r...) for r in eachrow(iq_data)] # Going over to GHz

# System parameters
theta= [ 3.6348672 ,  1.1435776 ,  0.        ,  1.7441809 , -0.4598031 ,
-0.37506938, -0.27870846,  0.        ,  0.        ,  0.        ,
0.        ,  0.         ] # SNAP2 gate binomial
xi = 2π * (-2.574749e6) #Hz
delta = -2.574749e6 #Hz drive spacing
K = -2π * 0 #Hz
alpha = 0 # trasmon qubit anharmonicity (not used)
xip = 0 # 2nd order dispersive shift correcion (not used)
wt = 0  # interaction picture
wc = 0 # interaction picture

#max_rabi_rate = 2π* 40e6
#cutoff_frequency = 2π* 50e6
tgate = 550e-9 #s


# Setup Hamiltonian
a = annihilation_op(N_cavity)
b = annihilation_op(N_qubit)

# Specify system Hamiltonians
Hosc = wc*kron(I(N_qubit), a'*a) + K/2*kron(I(N_qubit), a' * a' * a * a)
Htrans = wt*kron(b'*b, I(N_cavity)) + alpha/2*kron(b' * b' * b * b, I(N_cavity))
Hint = xi*kron(b'*b, a'*a) + xip/2*kron(b'*b, a'*a'*a*a)
H0 = Hosc + Htrans + Hint

Tc = kron(b', I(N_cavity))

# Setup initial state
x0_cavity =  normalize!(ones(N_cavity))
x0_qubit = [1; zeros(N_qubit-1)]
x0 = kron(x0_qubit, x0_cavity)

# setup for optimization target
# target operation for the cavity only
cav_target_operation = exp(diagm(1im*theta))

# target operatrion for the full system
full_target_operation = kron(I(N_qubit), cav_target_operation)
# work in the subspace of |0> qubit state 
cavity_subspace_projector = diagm(kron([1.,0.], ones(N_cavity)))
# net system target
subspace_target = full_target_operation * cavity_subspace_projector

##

Ntot = N_qubit * N_cavity
@variables xᵣ[1:Ntot], xᵢ[1:Ntot]
@variables uᵣ uᵢ

u = uᵣ + im*uᵢ
x = xᵣ + im*xᵢ

Htot = sparse(H0 * 1e-9 + u*Tc/2 + conj(u)*Tc'/2)

rhs = simplify.(-1im * (Htot * x))
dxdt = Symbolics.build_function(Symbolics.simplify.(c2r(rhs)), c2r(x), c2r(u), expression=Val{false})[2]

@variables λᵣ[1:Ntot] λᵢ[1:Ntot]
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

# plt1 = plot(sol.t, abs2.(x[1:11,:]'))