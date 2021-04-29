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
u_ctrl = [Complex(r...) for r in eachrow(iq_data)]

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

max_rabi_rate = 2π* 40e6
cutoff_frequency = 2π* 50e6
#gate_duration = 0.5e-6 #s
num_drives = 1


# Setup Hamiltonian
a = annihilation_op(N_cavity)
b = annihilation_op(N_qubit)

# Specify system Hamiltonians
Hosc = wc*kron(I(N_qubit), a'*a) + K/2*kron(I(N_qubit), a' * a' * a * a)
Htrans = wt*kron(b'*b, I(N_cavity)) + alpha/2*kron(b' * b' * b * b, I(N_cavity))
Hint = xi*kron(b'*b, a'*a) + xip/2*kron(b'*b, a'*a'*a*a)
H0 = Hosc + Htrans + Hint

T_control = kron(b', I(N_cavity))




Ntot = size(H0,1)
@parameters t uᵣ uᵢ
@variables cᵣ[1:Ntot](t) cᵢ[1:Ntot](t)
D = Differential(t)
c = cᵣ + im*cᵢ
u = uᵣ + im*uᵢ

Htot = sparse(H0 + u*Hc/2 + conj(u)*Hc'/2)

symrhs = -1im * (Htot * c)
symrhs = simplify.(symrhs)
symlhs = D.(c)

eqs = symlhs .~ symrhs

eqs_vec = vcat(eqs...) # Since complex-valued equations become a Vector of two equations
sys = ODESystem(eqs_vec, t, complex2real(c[:]), complex2real([u]))

dxdt = generate_function(sys; expression=Val{false})[2]


function wrap_f(f)
    function(dx, x, p, t)        
        f(dx, x, p[1], t)
    end
end

function update_u!(integrator)
    k = Int(round(integrator.t/1e-9))
    integrator.p[1][1] = real(integrator.p[2][k+1])
    integrator.p[1][2] = imag(integrator.p[2][k+1])    
end


# Setup initial state
c0_cavity =  normalize!(ones(N_cavity))
c0_qubit = [1; zeros(N_qubit-1)]
c0 = kron(c0_qubit, c0_cavity)


pcb=PeriodicCallback(update_u!, 1.0e-9, initial_affect=true, save_positions=(true,false))

prob = ODEProblem{true}(wrap_f(dxdt), complex2real(c0), (0.0, 550e-9), callback=pcb)

sol = solve(prob, Tsit5(), p=([0.0; 0.0], u_ctrl), saveat=0:1e-9:550e-9, adaptive=false, dt=1e-10)

t = sol.t
x = transpose(hcat([real2complex(u) for u in sol.u]...))

plot(sol.t, abs2.(x[:,1:11]))