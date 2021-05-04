# shouldn't need matrix conversions (vectorize vectors first, then apply these)
function complex2real(A::AbstractVector)
    Ar = Vector{real(eltype(A))}(undef, 2 * length(A))
    Ar[1:2:end] .= real.(A)
    Ar[2:2:end] .= imag.(A)
    return Ar
end
function real2complex(A::AbstractVector)
    if !iseven(length(A))
        error("Must have even length")
    end
    A[1:2:end] + im*A[2:2:end]
end

# Column-wise conversion, i.e., not the normal version
function complex2real(A::AbstractMatrix)
    Ar = Matrix{real(eltype(A))}(undef, 2 * size(A,1), size(A,2))
    Ar[1:2:end, :] .= real.(A)
    Ar[2:2:end, :] .= imag.(A)
    return Ar
end
function real2complex(A::AbstractMatrix)
    if !iseven(size(A,1))
        error("A must have an even number of rows")
    end
    A[1:2:end, :] + im*A[2:2:end, :]
end

annihilation_op(dim::Int) = diagm(1 => [sqrt(k) for k=1:dim-1])
function annihilation_op(dims...)
    a_vec = [annihilation_op(dim) for dim in dims]
    return [kron([k==j ? a_vec[k] : I(dims[k]) for k=1:length(dims)]...)
            for j=1:length(dims)]
end

qubit_hamiltonian(ωr, α, n) = diagm([k*ωr + α*(k-1)*k/2 for k=0:n-1])
