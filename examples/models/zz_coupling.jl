# Model used by Tahere to study the impact of ZZ interaction with a spectator qubit when performing single qubit gates

# System parameters
dimq = 3
dims = 3
Ntot = dimq * dims
α_q = 2π * 0.2 
α_s = 2π * 0.2
X_dispersive = 2π  * 1e-4

a_q = QuantumOptimalControl.annihilation_op(dimq)
a_s = QuantumOptimalControl.annihilation_op(dims)

# Specify system Hamiltonians
Hq = - α_q/2 * kron(a_q'*a_q'*a_q*a_q, I(dims)) # + wq * kron(dot(q',q), eye(dims))
Hspectator = - α_s/2 * kron(I(dimq), a_s'*a_s'*a_s*a_s) # + ws * kron(eye(dimq), dot(a',a))
Hint = -X_dispersive * kron(a_q'*a_q, a_s'*a_s)

# Define drive control operators
Tc = kron(a_q', I(dims))

H0 = Hq + Hspectator + Hint