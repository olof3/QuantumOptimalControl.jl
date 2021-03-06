using QuantumOptimalControl
using DelimitedFiles
using Ipopt, Zygote
using MKL

include("models/zz_coupling.jl")
include("../examples/models/setup_diffeq_rhs.jl")

##

function setup_ipopt_callbacks(A0, A_bl, x0, u_prototype, (Jfinal,dJfinal_dx), (L,dL_dx), B)

    nx = length(x0)    
    nu = length(A_bl)
    segment_count = size(B, 1)
    nsplines = size(B,2)    
    nc = nu * nsplines
    ng = 2
    Δt = 10 / segment_count    
    
    dxdt = setup_dxdt(A0, A_bl)
    dλdt = setup_dλdt(A0, A_bl)

    cache = QuantumOptimalControl.setup_grape_cache(A0, c2r(x0[:,:]), (nu, segment_count))
     
    c_prev = Vector{Float64}(undef, nc)
    #cache = QuantumOptimalControl.setup_grape_cache(A0Δt, complex(x0), (2, segment_count))

    f = function(c::Vector{Float64})
        c_prev .= c
        c = reshape(c, nsplines, nu)
        u = transpose(B*c)

        propagate_pwc(dxdt, c2r(x0[:,:]), u, Δt, cache; dt=0.2Δt)
        x = cache[1]

        Jfinal(reinterpret(ComplexF64, x[end])) + sum(x -> L(r2c(x)), x)
    end   

    f_grad = function(c, f_grad_out)
        if c_prev != c
            println("gradient not computed") # Note sure if IPOPT will ever ask for the gradient before the fcn value
            f(c) # Use the side effects in f that puts the necessary stuff into cache
        end
        
        #compute_pwc_gradient(dλdt, Jfinal, cache.u, Δt, A0, A_bl, cache; dUkdp_order=0, dt=0.2Δt)                                  
        #compute_sensitivity!(cache.dJdu, A_bl, cache.x, cache.λ, cache.u)
        
        c = reshape(c, nsplines, nu)
        u = transpose(B*c)        
        
        dJdu = compute_pwc_gradient(dλdt, dJfinal_dx, u, Δt, A0, A_bl, cache; dUkdp_order=3)

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


# The target gate unitary in the computational subspace
U_target_css = kron([0.0 1; 1 0], I(2)) # NOT
#U_target_css = kron([1.0 0; 0 1], I(2)) # Identity

#F = Q_css * x_target' * Q_css' # Can shift the order of the matrices inside trace
x_target = Q_css*U_target_css
Jfinal, dJfinal_dx = QuantumOptimalControl.setup_infidelity(x_target, 4)
#Jfinal, dJfinal_dx = QuantumOptimalControl.setup_infidelity_zcalibrated(Q_css*x_target)

##

tgate = 10 # ns
segment_count = 200

t = LinRange(0, tgate, segment_count + 1)
Δt = tgate / segment_count

A0, A1, A2 = QuantumOptimalControl.setup_bilinear_matrices(H0, Tc)
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
L, dL_dx = QuantumOptimalControl.setup_state_penalty(inds_penalty, inds_css, μ_state)
L, dL_dx = Returns(0), x -> 0*x

f, g, f_grad, g_jac, nu, ng, nx, nc, cache2 = setup_ipopt_callbacks(A0, [A1, A2], x0, zeros(2,segment_count), (Jfinal,dJfinal_dx), (L,dL_dx), B)

# Not quite the rabi rate since its on the coefficients
max_rabi_rate = 2π * 0.060
c_L = -max_rabi_rate*ones(nc)
c_U = max_rabi_rate*ones(nc)

m = 2
g_L = [-Inf, -Inf]
g_U = [2.0, 1]
#g_U = [1, 1.0]

c0 = [0.01*ones(nsplines); zeros(nsplines)][:]# + randn(nsplines,2)[:]


prob = createProblem(nc, c_L, c_U, ng, g_L, g_U, ng*nc, 0, f, g, f_grad, g_jac)
prob.x .= c0 #c_opt[:]

addOption( prob, "hessian_approximation", "limited-memory");
addOption( prob, "max_iter", 150);
addOption( prob, "print_level", 3);
#addOption( prob, "accept_after_max_steps", 10);

@time status = solveProblem(prob)

c_opt = reshape(prob.x, nsplines, nu)
u_opt = transpose(B*c_opt)

x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u_opt, x0)

println("IPOPT result: ", round(prob.obj_val, sigdigits=5), " ($(Ipopt.ApplicationReturnStatus[status]))")
println("Final infidelity: $(round(Jfinal(x[end]),sigdigits=5)), Guard state population: $(sum(L.(x)))")
println("Constrained quantities: $(round.(g(c_opt[:],zeros(2)),sigdigits=3))")


# Plot results
using Plots
include("plot_fcns.jl")

plot_2qubit_evolution(qb, t, x, u_opt)

## Plot the guard state population
plot_2qubit_evolution(qb, t, x, u_opt, to_states=["20", "21", "22"])