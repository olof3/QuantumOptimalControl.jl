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
    J = function(x)
        1 - abs2(tr(x_target'*x))/n^2
    end    
    dJ_dx = function(x)
        Ω = tr(x_target'*x)
        (-2Ω/n^2) * x_target
    end
    J, dJ_dx
end

# Here, U and U_target need to have 4 columns (corresponding to 2 qubits)
function setup_infidelity_zcalibrated(x_target)
    if size(x_target,2) != 4
        error("Only works for two-qubit gates, x_target must have four columns") # 1 qubit should also be okay to implement
    end
    J = function(x)
        m = diag(x_target'*x)
        1 - abs_sum_phase_calibrated(m)^2/4^2
    end    
    dJ_dx = function(x)
        m = diag(x_target'*x)
        J, pullback = ChainRulesCore.rrule(abs_sum_phase_calibrated, m)
        grad_F = pullback(1)[2]        
        (-2J/4^2) * x_target * Diagonal(grad_F)
    end
    J, dJ_dx
end


