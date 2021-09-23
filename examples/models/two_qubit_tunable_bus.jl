using QuantumOptimalControl

# Set up the Hamiltonian for two qubits coupled with a tunable because
# The model and parameters are from Jorge

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
Hq2 = ω2 * a2'*a2 + α2 * a2'*a2 * (a2'*a2 - I) # Factor /2 after anharmonicity missing ?! due to no RWA?
Hi1 = g1 * (a1' + a1) * (ac' + ac)
Hi2 = g2 * (a2' + a2) * (ac' + ac)
Hc = ωc0 * ac' * ac

H0 = Hq1 + Hq2 + Hi1 + Hi2

Ntot = prod(qb.dims)