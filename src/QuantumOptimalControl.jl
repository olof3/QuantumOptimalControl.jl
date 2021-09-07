module QuantumOptimalControl

using LinearAlgebra: getindex
using LinearAlgebra

using ChainRulesCore

using OrdinaryDiffEq, DiffEqCallbacks

import Base.getproperty, Base.getindex

using Zygote

using ExponentialUtilities

include("utils.jl")
include("fidelities.jl")
include("parameterized_pulses.jl")
include("gradient_computations.jl")



export propagator, real2complex, complex2real, c2r, r2c

export compress_states, decompress_states

export abs_trace_phase_calibrated, abs_sum_phase_calibrated, infidelity

const c2r = complex2real
const r2c = real2complex

export u_drag, u_sinebasis, annihilation_op, annihilation_ops

export compute_pwc_gradient, propagate_pwc

function wrap_controls(dxdt, u_fcn)
    function(dx, x, p, t)
        dxdt(dx, x, [reim(u_fcn(p,t))...], t)
    end
end

function wrap_envelope(f, u_fcn)
    function(dx, x, p, t)
        u = u_fcn(p, t) # Preferably tuple if not a scalar
        if x isa Vector # https://github.com/JuliaLang/julia/issues/41221
            f(dx, x, u, t)
        else
            @inbounds @views for j=1:size(x,2)
                f(dx[:,j], x[:,j], u, t)
            end
        end
    end
end
function wrap_f_old(f, u_fcn)
    function(dx, x, p, t)
        u = u_fcn(p, t) # Preferably tuple if not a scalar
        @inbounds @views for j=1:size(x,2)
            f(dx[:,j], x[:,j], u, t)
        end
    end
end

#=
function propagator(dUdt, U0, p, t)
    # Parameters are wrapped in real due to issues with reverse diff
    U0_vec = complex2real(U0[:])
    prob = ODEProblem{true}(dUdt, U0_vec, (0.0,t[end]))
    if t isa Real
        sol = solve(prob, Tsit5(), p=real(p), abstol=1e-8, reltol=1e-8, save_start=false, saveat=t[end], sensealg=ForwardDiffSensitivity())
        
        #return reshape(real2complex(sol.u[end]), size(U0))
        return reshape(real2complex(sol.u[end]), size(U0))
    else
        sol = solve(prob, Tsit5(), p=real(p), abstol=1e-8, reltol=1e-8)
        return sol.t, [reshape(real2complex(x), size(U0)) for x in sol.u]
    end
end
=#



end