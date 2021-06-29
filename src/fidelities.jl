function infidelity(U_target, Uf, calibration=:lms_phase)
    if size(U_target) == (4,4)
        return 1 - abs_trace_phase_calibrated(U_target' * Uf, calibration) / 4
    else
        error("Not supported yet")
    end
end

function abs_trace_phase_calibrated(M, calibration=:lms_phase)
    m = diag(M)

    normalize(x) = x / abs(x)

    if calibration === :lms_phase
        θ1 = -angle(conj(m[1])*m[2] + conj(m[3])*m[4]) # divide by (abs(m[1]) + abs(m[2]))
        return abs(m[1] + m[2]*cis(θ1)) + abs(m[3] + m[4]*cis(θ1))
    end

    if calibration === :lms_phase2
        x1, x2 = sqrt(abs(m[1]*m[2])), sqrt(abs(m[3]*m[4]))
        θ11 = -angle(conj(m[1])*m[2] / x1 + conj(m[3])*m[4] / x2) # divide by (abs(m[1]) + abs(m[2]))
        return abs(m[1] + m[2]*cis(θ11)) + abs(m[3] + m[4]*cis(θ11))
    end

    if calibration === :lms_phase3
        #r = abs.(m)
        v1, v2 = sqrt(abs(m[1]*m[2])), sqrt(abs(m[3]*m[4])) #abs(m[1]) + abs(m[2]), abs(m[3]) + abs(m[4])

        x = normalize(m[1]*conj(m[2]) / v1 + m[3]*conj(m[4]) / v2) # divide by (abs(m[1]) + abs(m[2]))
        v1, v2 = abs(m[1] + x*m[2]), abs(m[3] + x*m[4])

        x = normalize(m[1]*conj(m[2]) / v1 + m[3]*conj(m[4]) / v2) # divide by (abs(m[1]) + abs(m[2]))
        v1, v2 = abs(m[1] + x*m[2]), abs(m[3] + x*m[4])

        #x = normalize(m[1]*conj(m[2]) / v1 + m[3]*conj(m[4]) / v2) # divide by (abs(m[1]) + abs(m[2]))
        #v1, v2 = abs(m[1] + x*m[2]), abs(m[3] + x*m[4])

        #x = normalize(m[1]*conj(m[2]) / v1 + m[3]*conj(m[4]) / v2) # divide by (abs(m[1]) + abs(m[2]))
        #v1, v2 = abs(m[1] + x*m[2]), abs(m[3] + x*m[4])

        return v1 + v2
    end

    θ = if calibration === :lms_phase_semiold
        lms_phase_calibration_semiold(m)[1]
    elseif calibration === :lms_phase_old
        return lms_phase_calibration_old(m)[1]
    elseif calibration === :basic
        return basic_calibration(m)[1]
    elseif calibration === :grid
        return grid_calibration(m)[1]
    end
end

# Current approach
function basic_calibration(m)
    θ0 = angle(m[1])
    θ = [-(angle(m[2]) - θ0), -(angle(m[3]) - θ0)]
    return abs(m[1] + m[2]*cis(θ[1]) + m[3]*cis(θ[2]) + m[4]*cis(θ[1] + θ[2])), θ
end
# Seems to be very close to optimal, at least for overlap fidelity
function lms_phase_calibration_semiold(m)
    θ = [angle(m[1]*conj(m[2]) + m[3]*conj(m[4])),
         angle(m[1]*conj(m[3]) + m[2]*conj(m[4]))]

    return abs(m[1] + m[2]*cis(θ[1]) + m[3]*cis(θ[2]) + m[4]*cis(θ[1] + θ[2])), θ

end
function lms_phase_calibration_old(m) # Has problems with wrap-around
    ϕ = angle.(m)

    #θ0 = [-1 0 0; -1 1 0; -1 0 1; -1 1 1] \ ϕ; θ = θ0[2:3]
    # pinv([-1 0 0; -1 1 0; -1 0 1; -1 1 1])[2:3, :]

    R = [-0.5    0.5   -0.5   0.5;
         -0.5   -0.5    0.5   0.5]
    θ = R * ϕ

    θ[1] += mod(θ[1] - (ϕ[2]-ϕ[1]) + π, 2π) - π < 1.5 ? 0 : π
    θ[2] += mod(θ[2] - (ϕ[3]-ϕ[1]) + π, 2π) - π < 1.5 ? 0 : π

    return -θ
end


# https://en.wikipedia.org/wiki/Mean_of_circular_quantities
# Slightly worse version. Probably also has wrapping problems
#ϕ = angle(m[2:4]) .- angle(m[1])
#R = [2/3 -1/3 1/3;
#    -1/3  2/3 1/3] # [1 0; 0 1; 1 1] \ I(3)
#θ = R * ϕ

# To avoid dependening on an optimization package this is only a rather course grid search.
function grid_calibration(m)
    J = θ -> abs(m[1] + m[2]*cis(θ)) + abs(m[3] + m[4]*cis(θ))

    J_best, θ_best = findmax(J, LinRange(0,2π,100))

    return J_best, θ_best
end

function optimal_calibration(m, θ_tol=1e-15)
    J = θ -> abs(m[1] + m[2]*cis(θ)) + abs(m[3] + m[4]*cis(θ))

    θa, θb = minmax(angle(m[1]) - angle(m[2]), angle(m[3]) - angle(m[4]))

    minusJ1, θ1 = _golden_section_search(θ -> -J(θ), (θa, θb), θ_tol)
    minusJ2, θ2 = _golden_section_search(θ -> -J(θ), (θb, θa + 2π), θ_tol)

    return minusJ1 < minusJ2 ? (-minusJ1, θ1) : (-minusJ2, θ2)
end

# To avoid dependence on Optim
function _golden_section_search(f, (x_lower, x_upper), x_tol)
    x_lower > x_upper && error("x_lower must be less than x_upper")

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
