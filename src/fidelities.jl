

function abs_trace_phase_calibrated(M, calibration=:lms_phase)
    m = diag(M)

    θ = if calibration === :lms_phase
        lms_phase_calibration(m)
    elseif calibration === :lms_phase_old
        lms_phase_calibration_old(m)
    elseif calibration === :basic
        basic_calibration(m)
    elseif calibration === :grid
        grid_calibration(m)
    end
    
    return abs(sum(m[1] + m[2]*cis(θ[1]) + m[3]*cis(θ[2]) + m[4]*cis(θ[1] + θ[2])))
end

# Current approach
function basic_calibration(m)
    θ0 = angle(m[1])
    θ1 = angle(m[2]) - θ0
    θ2 = angle(m[3]) - θ0    
    return [-θ1, -θ2]
end
# Seems to be very close to optimal, at least for overlap fidelity
function lms_phase_calibration(m)
    θ = [angle(conj(m[1])*m[2] + conj(m[3])*m[4]),
         angle(conj(m[1])*m[3] + conj(m[2])*m[4])]

    return -θ
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
    J = θ -> abs(sum(m[1] + m[2]*cis(θ[1]) + m[3]*cis(θ[2]) + m[4]*cis(θ[1] + θ[2])))

    θ_grid = Iterators.product(LinRange(0,2π,100), LinRange(0,2π,100))
    θbest = argmax(J, θ_grid)

    #res = optimize(p -> -J(m, p), θopt_grid, x_abstol=1e-12)
    θbest
end
