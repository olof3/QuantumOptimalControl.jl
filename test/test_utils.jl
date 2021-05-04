a = annihilation_op(3)

α = -0.229
ωr = 3.82


qubit_hamiltonian(1, 0, 5) == diagm([k for k=0:4])
qubit_hamiltonian(0, 1, 5) == diagm([(k-1)*k/2 for k=0:4])




c = [1; 0.5]
@test real2complex(c) == [1 + 0.5im]
@test complex2real([1 + 0.5im]) == c

Ac = [1 0; 0 im]
Ar = [1 0; 0 0; 0 0; 0 1]
@test complex2real(Ac) == Ar
@test real2complex(Ar) == Ac