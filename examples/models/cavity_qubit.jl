using QuantumOptimalControl

# System parameters
N_qubit = 2 # nuber of transmon levels
N_cavity = 12 # nuber of cavity levels


xi = 2π * (-2.574749e-3) #GHz
delta = -2.574749e-3 #GHz drive spacing
K = -2π * 0 #GHz
alpha = 0 # trasmon qubit anharmonicity (not used)
xip = 0 # 2nd order dispersive shift correcion (not used)
wt = 0  # interaction picture
wc = 0 # interaction picture

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

Ntot = size(x0, 1)

# setup for optimization target

theta= [ 3.6348672 ,  1.1435776 ,  0.        ,  1.7441809 , -0.4598031 ,
-0.37506938, -0.27870846,  0.        ,  0.        ,  0.        ,
0.        ,  0.         ] # SNAP2 gate binomial
# target operation for the cavity only
cav_target_operation = exp(diagm(1im*theta))

# target operatrion for the full system
full_target_operation = kron(I(N_qubit), cav_target_operation)
# work in the subspace of |0> qubit state 
cavity_subspace_projector = diagm(kron([1.,0.], ones(N_cavity)))
# net system target
subspace_target = full_target_operation * cavity_subspace_projector
