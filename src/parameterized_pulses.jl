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
    Ωx = Ωy = zero(eltype(p))
    for k=1:num_controls
        bkt = sinpi(k*t/Tgate)
        Ωx += p[2k]*bkt
        Ωy += p[2k+1]*bkt
    end
    return Complex(Ωx, Ωy)
end

function cos_envelope(t_plateau, t_rise_fall, t)
    if t > t_rise_fall / 2 && t <= t_rise_fall / 2 + t_plateau
        1
    elseif t <= t_rise_fall / 2
        1 / 2 * (1 - cos(2π * t / t_rise_fall))
    elseif t > t_rise_fall / 2 + t_plateau
        1 / 2 * (1 - cos(2π * (t - t_plateau) / t_rise_fall))
    end
end