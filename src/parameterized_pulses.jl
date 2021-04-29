function u_drag(p, t)
    tgate = p[1]
    σ = p[2]
    A = p[3]
    ξ = p[4] # -(λ^2/4Δ)

    x = t - tgate/2
    
    tmp = exp(-x^2/(2σ^2))
    Ωx = tmp - exp(-tgate^2/(8σ^2))
    Ωy =  - ξ * x/σ^2 * tmp #-(λ^2/4Δ)*Ωx'(t)
    return Complex(A*Ωx, A*Ωy)
end

function u_sinebasis(p, t)#::Tuple{Float64,Float64}
    Tgate = p[1]
    num_controls = length(p) ÷ 2
    Ax = Ay = zero(eltype(p))
    for k=1:num_controls
        bkt = sinpi(k*t/Tgate)
        Ax += p[2k]*bkt
        Ay += p[2k+1]*bkt
    end
    return Complex(Ax, Ay)
end

