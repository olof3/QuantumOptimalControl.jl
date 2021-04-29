a = annihilation_op(3)

α = -0.229
ωr = 3.82


qubit_hamiltonian(1, 0, 5) == diagm([k for k=0:4])
qubit_hamiltonian(0, 1, 5) == diagm([(k-1)*k/2 for k=0:4])

