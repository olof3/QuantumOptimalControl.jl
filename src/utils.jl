# shouldn't need matrix conversions (vectorize vectors first, then apply these)
function complex2real(A::Union{Number,AbstractVector})
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



struct QuantumBasis
    dims::Vector{Int}
    state_dict::Dict{String, Int}
    state_labels::Vector{String}

    function QuantumBasis(dims)
        state_labels = map(e -> string("|", reverse(e)..., "⟩"), collect(Iterators.product([0:n-1 for n in dims]...))[:])
        state_dict = Dict(kron([string.(0:n-1) for n in dims]...) .=> 1:prod(dims))
        new(dims, state_dict, state_labels)
    end
end

function Base.getindex(qb::QuantumBasis, rows::Union{String,Vector{String},Colon}, cols::Union{String,Vector{String},Colon})
    row_inds = rows isa Union{String,Vector{String}} ? Base.getindex.(Ref(qb.state_dict), rows) : rows
    col_inds = cols isa Union{String,Vector{String}} ? Base.getindex.(Ref(qb.state_dict), cols) : cols
    I[1:qb.Ntot,1:qb.Ntot][row_inds, col_inds]
end


function Base.getproperty(qb::QuantumBasis, d::Symbol)
    if d == :Ntot
        return prod(getfield(qb, :dims))
    else
        return getfield(qb, d)
    end
end


annihilation_op(dim::Int) = diagm(1 => [sqrt(k) for k=1:dim-1])
function annihilation_ops(dims::Int...)
    a_vec = [annihilation_op(n) for n in dims]
    return [kron([k==j ? a_vec[k] : I(dims[k]) for k=1:length(dims)]...)
            for j=1:length(dims)]
end
annihilation_ops(qb::QuantumBasis) = annihilation_ops(qb.dims...)

qubit_hamiltonian(ωr, α, n) = diagm([k*ωr + α*(k-1)*k/2 for k=0:n-1])




function compress_states(x, v)
    n1, n2 = length(v[1][2]), length(v[2][2])
    x_compr = zeros(eltype(x), size(x,1), max(n1,n2))
    x_compr[v[1][1], 1:n1] .= x[v[1][1], v[1][2]]
    x_compr[v[2][1], 1:n2] .= x[v[2][1], v[2][2]]
    x_compr
end
function decompress_states(x_compr, v)
    n1, n2 = length(v[1][2]), length(v[2][2])
    x = zeros(eltype(x_compr), size(x_compr,1), n1 + n2)
    x[v[1][1], v[1][2]] .= x_compr[v[1][1], 1:n1]
    x[v[2][1], v[2][2]] .= x_compr[v[2][1], 1:n2]
    x
end