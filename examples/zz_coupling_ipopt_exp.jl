using QuantumOptimalControl
using Ipopt, Zygote, MKL, DelimitedFiles
using BSplines

include("models/zz_coupling.jl")
include("ipopt_callbacks_exp.jl")

Q_css = qb[:, ["00", "01", "10", "11"]] # Orthogonal basis for the computational subspace

x0 = float(Q_css)

# Setup the fidelity function

css_target = kron([0 1; 1 0], I(2)) # NOT
#css_target = kron([1 0; 0 1], I(2)) # Identity
Jfinal, dJfinal_dx = QuantumOptimalControl.setup_infidelity(Q_css*css_target, 4)
#Jfinal, dJfinal_dx = QuantumOptimalControl.setup_infidelity_zcalibrated(Q_css*x_target)

##
tgate = 10 # ns
segment_count = 100
Δt = tgate / segment_count

t = LinRange(0, tgate, segment_count + 1) # t = 0:Δt:tgate

A0Δt, A1Δt, A2Δt = QuantumOptimalControl.setup_bilinear_matrices(H0, Tc, Δt)

# Setup the basis functions
nsplines = 10
spline_basis = BSplineBasis(4, LinRange(0,tgate,nsplines+4))

Bpre = zeros(length(t)-1, nsplines+6)
t_midpoints = t[1:end-1] .+ Δt/2
for (k, t) in enumerate(t_midpoints)
    spl = bsplines(spline_basis, t)
    Bpre[k, collect(eachindex(spl))] .= spl.parent     
end
B = Bpre[:, 4:end-3]

# Setup penalty for guard state population

#inds_css = 1:4
#inds_penalty = qb(["20", "21", "22"])
#μ_state = 1e-9 * Δt/tgate # weighting factors
#L, dL_dx = QuantumOptimalControl.setup_state_penalty(inds_penalty, inds_css, μ_state)
L, dL_dx = Returns(0), x -> 0*x


## Optimization

f, g, f_grad, g_jac, nu, ng, nx, nc, cache = setup_ipopt_callbacks(A0Δt, A1Δt, A2Δt, x0, zeros(2,segment_count), (Jfinal,dJfinal_dx), (L,dL_dx), B)

# Not quite the rabi rate since its on the coefficients
max_rabi_rate = 2π * 0.060
c_L = -max_rabi_rate*ones(nc)
c_U = max_rabi_rate*ones(nc)

# Constraints (control signal penalties)
g_L = [-Inf, -Inf]
g_U = [2.0, 1]

c0 = [0.01*ones(nsplines); zeros(nsplines)][:]# + 0.001randn(nsplines,2)[:]

prob = createProblem(nc, c_L, c_U, ng, g_L, g_U, ng*nc, 0, f, g, f_grad, g_jac)
prob.x .= c0 #c_opt[:]

addOption(prob, "hessian_approximation", "limited-memory")
addOption(prob, "max_iter", 150)
addOption(prob, "print_level", 4)
#addOption(prob, "accept_after_max_steps", 10);

@time status = solveProblem(prob)

c_opt = reshape(prob.x, nsplines, nu)
u_opt = transpose(B*c_opt)

x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u_opt, x0)

println("IPOPT result: ", round(prob.obj_val, sigdigits=5), " ($(Ipopt.ApplicationReturnStatus[status]))")
println("Final infidelity: $(round(Jfinal(x[end]),sigdigits=5)), Guard state population: $(sum(L.(x)))")
println("Constrained quantities: $(round.(g(c_opt[:],zeros(2)),sigdigits=3))")


# Plot results
include("plot_fcns.jl")

plot_2qubit_evolution(qb, t, x, u_opt)

## Plot the guard state population
plot_2qubit_evolution(qb, t, x, u_opt, to_states=["20", "21", "22"])

