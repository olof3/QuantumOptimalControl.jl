# The functions return penality functions together with their gradiends (if it is desired to avoid AD tools)

function setup_state_penalty(inds_penalty::AbstractVector, inds_css::AbstractVector, μ::Real)
    L = function(x)
        @views μ*sum(abs2, x[inds_penalty, inds_css])
    end
    dL_dx = function(x)
        dL_dx = zeros(eltype(x), size(x))    
        dL_dx[inds_penalty, inds_css] .= 2 .* μ .* x[inds_penalty, inds_css]
        dL_dx
    end
    return L, dL_dx
end


# Some rows of U_target could be zero, i.e., if full propagator is computed, but only subspace is considered
function setup_infidelity(x_target, n=size(x_target, 2))    
    F = function(x)
        1 - abs2(tr(x_target'*x))/n^2
    end    
    dF_dx = function(x)
        Ω = tr(x_target'*x)
        (-2Ω/n^2) * x_target
    end
    F, dF_dx
end

# Here, U and U_target need to have 4 columns (corresponding to 2 qubits)
# The gradient computation assumes that the optimum is found,
# so should really only allow :optimal, only used for experiments
function setup_infidelity_zcalibrated(x_target, calibration=:optimal)
    if size(x_target,2) != 4
        error("Only works for two-qubit gates, x_target must have four columns") # 1 qubit should also be okay to implement
    end
    F = function(x)
        v = diag(x_target'*x)
        f = abs_sum_phase_calibrated(v, calibration)
        1 - f^2/4^2
    end    
    dF_dx = function(x)
        v = diag(x_target'*x)
        f, pullback = ChainRulesCore.rrule(abs_sum_phase_calibrated, v)
        grad_F = pullback(1)[2]        
        (-2f/4^2) * x_target * Diagonal(grad_f)
    end
    F, dF_dx
end


