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


v_compress = ((1:2:27, [1,4]), ((2:2:26), [2,3]))
x0 = collect(reshape(1:27*4, 27, 4))
x0[1:2:27, [2,3]] .= 0
x0[2:2:26, [1,4]] .= 0
x1 = compress_states(x0, v_compress)
@test size(x1, 2) == 2
@test decompress_states(x1, v_compress) == x0


v_compress = ((1:2:27, [1,4,5]), ((2:2:26), [2,3]))
x0 = collect(reshape(1:27*5, 27, 5))
x0[1:2:27, [2,3]] .= 0
x0[2:2:26, [1,4,5]] .= 0
x1 = compress_states(x0, v_compress)
@test size(x1, 2) == 3
@test decompress_states(x1, v_compress) == x0