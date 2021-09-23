using QuantumOptimalControl
using DelimitedFiles
using Ipopt, Zygote
using MKL

include("models/zz_coupling.jl")

iq_data = 1e-9*DelimitedFiles.readdlm(joinpath(dirname(Base.find_package("QuantumOptimalControl")), "../examples/virtual_zz_pulse_tahereh210823.csv"))
u = copy(transpose(iq_data))
##
function setup_ipopt_callbacks(A0Δt, A1Δt, A2Δt, x0, u_prototype, (Jfinal,dJfinal_dx), (L,dL_dx), B)
    nu = size(u_prototype,1)
    ng = 2
    nx = length(x0)
    nsplines = size(B,2)
    nc = nu * nsplines

    c_prev = Vector{Float64}(undef, nc)
    cache = QuantumOptimalControl.setup_grape_cache(A0Δt, complex(x0), (2, segment_count))

    f = function(c::Vector{Float64})
        c_prev .= c
        c = reshape(c, nsplines, nu)
        u = transpose(B*c)

        # xdot = A0*x + (A1*u[1] + A2*u[2])*x
        x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0, cache)

        Jfinal(x[end]) + sum(L, x)
    end

    f_grad = function(c, f_grad_out)
        if c_prev != c
            println("gradient not computed") # Note sure if IPOPT will ever ask for the gradient before the fcn value
            f(c) # Use the side effects in f that puts the necessary stuff into cache
        end
        
        dJdu = QuantumOptimalControl.grape_sensitivity(A0Δt, [A1Δt, A2Δt], dJfinal_dx, cache.u, x0, cache; dUkdp_order=3, dL_dx=dL_dx) # use cache.u
        dJdc = B'*transpose(dJdu)

        f_grad_out .= dJdc[:]
    end
    
    g_oop = function(c)
        c = reshape(c, nsplines, nu)
        [norm(c);
        norm(diff(c, dims=1))]        
    end

    g = function(c, g_out)
        g_out .= g_oop(c)
    end

    function g_jac(c, mode, rows, cols, g_jac_out)    
        if mode == :Structure
            cols .= kron(ones(ng), 1:nc)
            rows .= kron(1:ng, ones(nc))    
        else
            g_jac_tmp = Zygote.jacobian(g_oop, c)[1]::Matrix{Float64}
            g_jac_out .= transpose(g_jac_tmp)[:]
        end        
    end

    f, g, f_grad, g_jac, nu, ng, nx, nc, cache
end

## Could try test_ippot_fcns.jl to make sure that gradients/jacobians work

qb = QuantumOptimalControl.QuantumBasis([3,3])
Q_css = qb[:, ["00", "01", "10", "11"]]

#x0 = 1.0I[1:9, 1:9]
x0 = 1.0Q_css



x_target = kron([0 1; 1 0], I(2)) # NOT
#x_target = kron([1 0; 0 1], I(2)) # Identity

#F = Q_css * x_target' * Q_css' # Can shift the order of the matrices inside trace
F = Q_css*x_target
Jfinal, dJfinal_dx = QuantumOptimalControl.setup_infidelity(F, 4)
#Jfinal, dJfinal_dx = QuantumOptimalControl.setup_infidelity_zcalibrated(Q_css*x_target)

##

tgate = 10 # ns
segment_count = 100

t = LinRange(0, tgate, segment_count + 1)

Δt = tgate / segment_count

A0Δt, A1Δt, A2Δt = QuantumOptimalControl.setup_bilinear_matrices(H0, Tc, Δt)


using BSplines

nsplines = 10
spline_basis = BSplineBasis(4, LinRange(0,tgate,nsplines+4))

Bpre = zeros(length(t)-1, nsplines+6)
t_midpoints = t[1:end-1] .+ Δt/2
for (k, t) in enumerate(t_midpoints)
    spl = bsplines(spline_basis, t)
    Bpre[k, collect(eachindex(spl))] .= spl.parent     
end
B = Bpre[:, 4:end-3]

##



#inds_css = qb(["00", "01", "10", "11"])
inds_css = 1:4
inds_penalty = qb(["20", "21", "22"])

μ_state = 1e-9 * Δt/tgate
#L, dL_dx = QuantumOptimalControl.setup_state_penalty(inds_penalty, inds_css, μ_state)
L, dL_dx = Returns(0), x -> 0*x

f, g, f_grad, g_jac, nu, ng, nx, nc, cache = setup_ipopt_callbacks(A0Δt, A1Δt, A2Δt, x0, u, (Jfinal,dJfinal_dx), (L,dL_dx), B)


# Not quite the rabi rate since its on the coefficients
max_rabi_rate = 2π * 0.060
c_L = -max_rabi_rate*ones(nc)
c_U = max_rabi_rate*ones(nc)

m = 2
g_L = [-Inf, -Inf]
g_U = [2.0, 1]
#g_U = [1, 1.0]

c0 = [0.01*ones(nsplines); zeros(nsplines)][:]# + 0.001randn(nsplines,2)[:]


prob = createProblem(nc, c_L, c_U, ng, g_L, g_U, ng*nc, 0, f, g, f_grad, g_jac)
prob.x .= c0 #c_opt[:]

addOption( prob, "hessian_approximation", "limited-memory");
addOption( prob, "max_iter", 150);
addOption( prob, "print_level", 5);
#addOption( prob, "accept_after_max_steps", 10);



#@profview status = solveProblem(prob)
@time status = solveProblem(prob)

println(Ipopt.ApplicationReturnStatus[status])
#println(prob.x)

c_opt = reshape(prob.x, nsplines, nu)
u = transpose(B*c_opt)


#cache = QuantumOptimalControl.setup_grape_cache(A0Δt, complex(x0), (2, segment_count))
x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)

println([sum(L.(x)), Jfinal(x[end])])
println([sum(L.(x)) / μ_state, Jfinal(x[end])])
println(prob.obj_val) # 17.01401714517915


@time f(vec(c_opt))
@time f_grad(vec(c_opt), copy(vec(c_opt)))
@time g(vec(c_opt), zeros(2))


x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)

v = [[x[k][ij_ind] for k=1:length(x)] for ij_ind = CartesianIndices(x[1])]


using Plots
plt1 = plot();
plot!(plt1, t, abs2.(v[2]))


plts = []


css_labels = ["00", "01", "10", "11"]
css_dict = Dict(css_labels .=> 1:4)


comp_basis_inds = [qb.state_dict[s] for s in css_labels]
#comp_basis_inds = qb(["20", "21"])

for i=0:1, j=0:1

    l = css_dict[string(i, j)]

    plt = plot(legend=(i==j==0), title="From state |$(css_labels[l])⟩")
    for k in comp_basis_inds
        plot!(plt, t, abs2.(v[k,l]), label=qb.state_labels[k])
    end
    push!(plts, plt)
end

plt_u = plot(t[1:end-1], transpose(u), linetype=:steppost, title="Control signal")
plot!(plt_u, [t[1], t[end]], max_rabi_rate*[-1 1; -1 1], c="black", label=nothing)

layout = @layout [a b; c d; e]
plot(plts..., plt_u, layout=layout)