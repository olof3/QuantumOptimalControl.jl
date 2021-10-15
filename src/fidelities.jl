abs_trace_phase_calibrated(M, calibration=:optimal) = abs_sum_phase_calibrated(diag(M), calibration)

function abs_sum_phase_calibrated(m, calibration=:optimal)
    if calibration === :optimal
        return optimal_calibration(m)[1]
    elseif calibration === :approx_optimal
        return optimal_calibration(m, exact_optimum=false)[1]
    elseif calibration === :basic
        return basic_calibration(m)[1]
    elseif calibration === :none
        return abs(sum(m))
    elseif calibration === :circular_mean1
        circular_mean_calibration(m, :version1)[1]
    elseif calibration === :circular_mean2
        circular_mean_calibration(m, :version2)[1]
    elseif calibration === :circular_mean3
        circular_mean_calibration(m, :version3)[1]
    elseif calibration === :circular_mean4
        circular_mean_calibration(m, :version4)[1]
    elseif calibration === :grid
        return grid_calibration(m)[1]
    else
        error("Unknown calibration algorithm")
    end
end

function abs_sum_phase_calibrated_grad(m, θ1_opt)
    v1 = (m[1] + cis(θ1_opt)*m[2])
    v2 = (m[3] + cis(θ1_opt)*m[4])
    dF_dm = 2*(abs(v1) + abs(v2)) .* [v1 / abs(v1), v1 / abs(v1) * cis(-θ1_opt), v2/abs(v2),  v2/abs(v2) * cis(-θ1_opt)]
end

function ChainRulesCore.rrule(::typeof(abs_sum_phase_calibrated), m)
    y, θ_opt = optimal_calibration(m)

    v1 = (m[1] + cis(θ_opt[1])*m[2])
    v2 = (m[3] + cis(θ_opt[1])*m[4])
    #dF_dm = 2*(abs(v1) + abs(v2)) .* [v1 / abs(v1), v1 / abs(v1) * cis(-θ_opt[1]), v2/abs(v2),  v2/abs(v2) * cis(-θ_opt[1])]
    dF_dm = [v1 / abs(v1), v1 / abs(v1) * cis(-θ_opt[1]), v2/abs(v2),  v2/abs(v2) * cis(-θ_opt[1])]
    return y, ybar -> (NoTangent(), dF_dm * ybar)
end


function target_gate_infildelity_pc(U_target, U)
    1 - abs_sum_phase_calibrated(tr(U_target'*U))^2 / 16
end




# Current approach
function basic_calibration(m)
    θ0 = angle(m[1])
    θ = [-(angle(m[2]) - θ0), -(angle(m[3]) - θ0)]
    return abs(m[1] + m[2]*cis(θ[1]) + m[3]*cis(θ[2]) + m[4]*cis(θ[1] + θ[2])), θ
end

# To avoid dependening on an optimization package. This is just a rather course grid search.
function grid_calibration(m)
    J = θ -> abs(m[1] + m[2]*cis(θ)) + abs(m[3] + m[4]*cis(θ))

    θ_vec = LinRange(0,2π,100)
    J_best, k_best = findmax(J, θ_vec)

    return J_best, θ_vec[k_best]
end


function optimal_calibration(m, θ_tol=1e-9; exact_optimum=true)
    a1 = abs2(m[1]) + abs2(m[2]); b1 = 2*abs(m[1])*abs(m[2])
    a2 = abs2(m[3]) + abs2(m[4]); b2 = 2*abs(m[3])*abs(m[4])

    ϕ1 = mod2pi(angle(m[1]) - angle(m[2]))
    ϕ2 = mod2pi(angle(m[3]) - angle(m[4]))
    ϕ_mean, Δ, α = if abs(ϕ2 - ϕ1) <= π
        (ϕ1 + ϕ2)/2, abs(ϕ2 - ϕ1)/2, ϕ1 < ϕ2 ? 1 : -1
    else
        (2π + ϕ1 + ϕ2)/2, π - abs(ϕ2 - ϕ1)/2, ϕ1 < ϕ2 ? -1 : 1
    end

    f = δ -> sqrt(a1 + b1*cos(δ + Δ)) + sqrt(a2 + b2*cos(δ - Δ)) # δ is deviation from ϕ_mean

    minus_f, δ = if exact_optimum
        _golden_section_search(δ -> -f(δ), (-Δ, Δ), θ_tol)
    else
        δ_opt_approx = _abs_trace_δ_approximation(a1, a2, b1, b2, Δ) # Compute first-order approximation
        -f(δ_opt_approx), δ_opt_approx
    end

    θ2_opt = ϕ_mean + α*δ
    θ1_opt = angle(m[1] + m[2]*cis(θ2_opt)) - angle(m[3] + m[4]*cis(θ2_opt))

    return -minus_f, [θ1_opt, θ2_opt]
end

function _abs_trace_δ_approximation(a1, a2, b1, b2, Δ)
    s1 = sqrt(a1 + cos(Δ)*b1)
    s2 = sqrt(a2 + cos(Δ)*b2)
    if b1 == 0 || s1 == 0
        δ_opt_approx = Δ
    elseif b2 == 0 || s2 == 0
        δ_opt_approx = -Δ
    else
        δ_opt_approx = -2sin(Δ)*(b1/s1 - b2/s2)/((b1*(b1 + 2*a1*cos(Δ) + b1*cos(Δ)^2))/s1^3 + (b2*(b2 + 2*a2*cos(Δ) + b2*cos(Δ)^2))/s2^3)
    end
    δ_opt_approx
end


# The golden section search is included here to avoid dependence on Optim
function _golden_section_search(f, (x_lower, x_upper), x_tol)
    x_lower > x_upper && error("x_lower must be less than x_upper ($x_lower, $x_upper)")

    golden_ratio = 0.5 * (3.0 - sqrt(5.0))

    new_minimizer = x_lower + golden_ratio*(x_upper-x_lower)
    new_minimum = f(new_minimizer)

    while x_upper - x_lower >= x_tol
        if x_upper - new_minimizer > new_minimizer - x_lower
            new_x = new_minimizer + golden_ratio*(x_upper - new_minimizer)
            new_f = f(new_x)
            if new_f < new_minimum
                x_lower = new_minimizer
                new_minimizer = new_x
                new_minimum = new_f
            else
                x_upper = new_x
            end
        else
            new_x = new_minimizer - golden_ratio*(new_minimizer - x_lower)
            new_f = f(new_x)
            if new_f < new_minimum
                x_upper = new_minimizer
                new_minimizer = new_x
                new_minimum = new_f
            else
                x_lower = new_x
            end
        end
    end
    return new_minimum, new_minimizer
end


function circular_mean_calibration(m, version=:version3)
    if version === :version1
        # https://en.wikipedia.org/wiki/Mean_of_circular_quantities
        θ2 = -angle(conj(m[1])*m[2] + conj(m[3])*m[4]) # divide by (abs(m[1]) + abs(m[2]))
    elseif version === :version2
        x1, x2 = sqrt(abs(m[1]*m[2])), sqrt(abs(m[3]*m[4]))
        if x1 < eps() || x2 < eps()
            return abs(m[1]) + abs(m[2]) + abs(m[3]) + abs(m[4])
        end
        θ2 = -angle(conj(m[1])*m[2] / x1 + conj(m[3])*m[4] / x2)
    elseif version === :version3
        x1, x2 = (abs(m[1]) + abs(m[2])), (abs(m[3]) + abs(m[4]))
        θ2 = -angle(conj(m[1])*m[2] / x1 + conj(m[3])*m[4] / x2)
    elseif version === :version4
        x1, x2 = abs(m[1]*m[2]), abs(m[3]*m[4])
        if x1 < eps() || x2 < eps()
            return abs(m[1]) + abs(m[2]) + abs(m[3]) + abs(m[4])
        end
        θ2 = -angle(conj(m[1])*m[2] / x1 + conj(m[3])*m[4] / x2)
    else
        error("Unkown algorithm name")
    end

    θ1 = angle(m[1] + m[2]*cis(θ2)) - angle(m[3] + m[4]*cis(θ2))

    return abs(m[1] + m[2]*cis(θ2)) + abs(m[3] + m[4]*cis(θ2)), [θ1, θ2]
end