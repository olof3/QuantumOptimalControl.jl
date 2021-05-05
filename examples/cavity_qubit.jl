using DelimitedFiles

using DifferentialEquations
using ModelingToolkit

using SparseArrays

using QuantumOptimalControl

using Plots

# Optimization parameters
N_qubit = 2 # nuber of transmon levels
N_cavity = 12 # nuber of cavity levels

examples_dir = dirname(Base.find_package("QuantumOptimalControl"))
iq_data = DelimitedFiles.readdlm(joinpath(examples_dir, "../examples/cavity_qubit_control.csv"))
u_ctrl = 1e-9 * [Complex(r...) for r in eachrow(iq_data)] # Going over to GHz

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
c0_cavity =  normalize!(ones(N_cavity))
c0_qubit = [1; zeros(N_qubit-1)]
c0 = kron(c0_qubit, c0_cavity)

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

Ntot = size(H0,1)
@parameters t uᵣ uᵢ
@variables cᵣ[1:Ntot](t) cᵢ[1:Ntot](t)
D = Differential(t)
c = cᵣ + im*cᵢ
u = uᵣ + im*uᵢ

@parameters λᵣ λᵢ
λ = λᵣ + im*λᵢ

Htot = sparse(H0 * 1e-9 + u*Tc/2 + conj(u)*Tc'/2)

f = simplify.(-1im * (Htot * c))
eqs = D.(c) .~ simplify.(-1im * (Htot * c))
eqs_vec = vcat(eqs...) # Since complex-valued equations become a Vector of two equations
sys = ODESystem(eqs_vec, t, complex2real(c[:]), complex2real([u]))
dxdt = generate_function(sys; expression=Val{false})[2]

eqs = D.(c) .~ simplify.(-1im * (Htot' * c))
eqs_vec = vcat(eqs...) # Since complex-valued equations become a Vector of two equations
sys = ODESystem(eqs_vec, t, complex2real(c[:]), complex2real([u]))
dλdt = generate_function(sys; expression=Val{false})[2]

f_u = λ * Symbolics.jacobian(complex2real(f), complex2real([u]))




##

function wrap_f(f)
    function(dx, x, p, t)        
        f(dx, x, p[1], t)
    end
end


Δt = 1.0 # 1e-9
tgate = 550 * Δt

function update_u!(integrator)
    k = Int(round(integrator.t/Δt))
    integrator.p[1][1] = real(integrator.p[2][k+1])
    integrator.p[1][2] = imag(integrator.p[2][k+1])    
end



pcb=PeriodicCallback(update_u!, Δt, initial_affect=true, save_positions=(true,false))

prob = ODEProblem{true}(wrap_f(dxdt), complex2real(c0), (0.0, tgate), callback=pcb)

sol = solve(prob, Tsit5(), p=([0.0; 0.0], u_ctrl), saveat=0:Δt:tgate, adaptive=false, dt=0.1Δt)

t = sol.t
x = transpose(hcat([real2complex(u) for u in sol.u]...))

plt1 = plot(sol.t, abs2.(x[:,1:11]))


abs(real2complex(sol.u[end])' * normalize!(diag(subspace_target)))

##

# Using very simple propagation

function prop_pwc(H0, Tc, u_ctrl, x0)
    x = Vector{typeof(x0)}(undef, length(u_ctrl)+1)
    x[1] = copy(x0)

    for k=1:length(u_ctrl)
        Ω = u_ctrl[k]
        x[k+1] = exp(-1e-9*im*(H0 + Ω*Tc + conj(Ω)*Tc')) * x[k]
    end
    return x
end

@time x = prop_pwc(H0, Tc/2, u_ctrl, complex(c0))

x = prop_pwc(H0, Tc/2, u_ctrl, complex(c0))

x2 = hcat(x...) 

plt2 = plot( abs2.(x2[1:11,:])')


plot(plt1, plt2)

