module QuantumOptimalControl

using LinearAlgebra

using DifferentialEquations
using DiffEqSensitivity: ForwardDiffSensitivity

using Plots

include("utils.jl")
include("parameterized_pulses.jl")

export propagator, real2complex, complex2real

export u_drag, u_sinebasis, annihilation_op

function wrap_controls(dxdt, u_fcn)
    function(dx, x, p, t)
        dxdt(dx, x, [reim(u_fcn(p,t))...], t)
    end
end

function plot_propagation(t, Ut_vec)
    n = size(Ut_vec[1], 1)
    plt = [plot() for k=1:n]
    for j=1:n 
        for i=1:n
            # Use abs2 or abs?
            plot!(plt[j], t, [abs2(Ut[i,j]) for Ut in Ut_vec], label=(j==1 ? "|$(i-1)⟩" : nothing), title="Evolution from |$(j-1)⟩")
        end
    end
    if n == 2
        layout = @layout [a b]
    else
        layout = @layout [a b c]
    end
    plt_all = plot(plt..., layout=layout, size=(1700,600))
    display(plt_all)
end


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




end