using Symbolics

function setup_dxdt(A0, A::Vector{<:Matrix})
    N = size(A0, 1)
    @variables xᵣ[1:N], xᵢ[1:N]
    xᵣ, xᵢ = [xᵣ...], [xᵢ...]
    x = xᵣ + im*xᵢ    

    @variables u[1:length(A)]
    u = [u...]
    
    rhs = (A0 + sum(u[k]*A[k] for k=1:length(A)))*x
    dxdt = Symbolics.build_function(Symbolics.simplify.(c2r(rhs)), c2r(x), u, expression=Val{false})[2]
    return dxdt
end

# Currently no support for when adjoint eq. depends on x
function setup_dλdt(A0, A::Vector{<:Matrix})
    N = size(A0, 1)

    @variables λᵣ[1:N] λᵢ[1:N]
    λᵣ, λᵢ = [λᵣ...], [λᵢ...]
    λ = λᵣ + im*λᵢ

    @variables u[1:length(A)]
    u = [u...]
    
    #rhs = simplify.(-1im * (Htot' * λ))
    rhs = -(A0' + sum(u[k]*A[k]' for k=1:length(A)))*λ
    dλdt = Symbolics.build_function(Symbolics.simplify.(c2r(rhs)), c2r(λ), u, expression=Val{false})[2]    
    return dλdt
end


function setup_pwc_sensitivity_fcn(A0, A::Vector{<:Matrix}; order=1)
    N = size(A0, 1)

    @variables xᵣ[1:N], xᵢ[1:N]
    xᵣ, xᵢ = [xᵣ...], [xᵢ...]
    x = xᵣ + im*xᵢ    

    @variables λᵣ[1:N] λᵢ[1:N]
    λᵣ, λᵢ = [λᵣ...], [λᵢ...]
    λ = λᵣ + im*λᵢ

    @variables u[1:length(A)]
    u = [u...]

    G_expr = if order == 1
        [real(λ'*Ak*x) for Ak in A]
    elseif order == 2
        @warn "not sure if this is correct.."
        X = (A0 + sum(u[k]*A[k] for k=1:length(A)))
        [real(λ'*Ak*x) + 1//2*real(λ'*X*Ak*x) + 1//2*real(λ'*Ak*X*x) for Ak in A]
    else
        error("order > 2 not supported")
    end

    G = Symbolics.build_function(Symbolics.simplify.(G_expr), c2r(x), c2r(λ), u, expression=Val{false})[2]
    
    return G
end