using QuantumOptimalControl
using DelimitedFiles
using Ipopt, Zygote
using MKL

include("models/zz_coupling.jl")

iq_data = 1e-9*DelimitedFiles.readdlm(joinpath(dirname(Base.find_package("QuantumOptimalControl")), "../examples/pulse_210823.csv"))

u = iq_data[:,1] + im*iq_data[:,2]
u_mat = transpose(iq_data)

u = copy(transpose(iq_data))

x0 = 1.0I[1:9, 1:9]

# Overly constrained x_target (full state space) (used in Tahere's original Q-CTRL code)
#x_target_qubit1 = [0 1 0; 1 0 0; 0 0 1]
#x_target_full = kron(x_target_qubit1, I(3))

qb = QuantumOptimalControl.QuantumBasis([3,3])
Q_css = qb[:, ["00", "01", "10", "11"]]

x_target = kron([0 1; 1 0], I(2)) # NOT
#x_target = kron([1 0; 0 1], I(2)) # Id

#Jfinal = x -> 1 - abs(tr(Q_css * x_target' * Q_css'*x))/4
#Jfinal = x -> 1 - abs_sum_phase_calibrated(diag(x_target' * Q_css'*x*Q_css))/4
#dJfinal_dx = x -> Zygote.gradient(Jfinal, x)[1]

F = Q_css * x_target' * Q_css' # Can shift the order of the matrices inside trace

Jfinal, dJfinal_dx = QuantumOptimalControl.setup_infidelity(F, 4)
#Jfinal, dJfinal_dx = QuantumOptimalControl.setup_infidelity_zcalibrated(Q_css*x_target)

#- test1

##

tgate = 10 # ns
segment_count = 100

t = LinRange(0, tgate, segment_count + 1)

Δt = tgate / segment_count

#=A0 = -im*H0
A1 = -im*(Tc + Tc')
A2 = -im*(im*(Tc - Tc'))
A0Δt, A1Δt, A2Δt = A0 * Δt, A1 * Δt, A2 * Δt=#
A0Δt, A1Δt, A2Δt = QuantumOptimalControl.setup_bilinear_matrices(H0, Tc, Δt)


using BSplines
using Plots

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
Q_penalty = qb[:, ["20", "21", "22"]]
#L2 = x -> Δt*0.0002*norm(Q_penalty' * x * Q_css)

#L = x -> norm(Q_penalty' * x * Q_css)^2
#dL_dx = x -> Zygote.gradient(L2, x)[1]

inds_css = qb(["00", "01", "10", "11"])
inds_penalty = qb(["20", "21", "22"])

μ_state = 1e-9 * Δt/tgate
L, dL_dx = QuantumOptimalControl.setup_state_penalty(inds_penalty, inds_css, μ_state)

@time f, g, f_grad, g_jac, nu, ng, nx, nc = let x0=x0, L=L    

    nu = 2    
    ng = 2
    nx = length(x0)
    nc = nu * nsplines

    c_prev = Vector{Float64}(undef, nc)
    cache = QuantumOptimalControl.setup_grape_cache(A0Δt, complex(x0), (2, segment_count))

    f = function(c)
        c_prev .= c
        c = reshape(c, nsplines, nu)
        u = transpose(B*c)

        # xdot = A0*x + (A1*u[1] + A2*u[2])*x
        x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0, cache)

        Jfinal(x[end]) + sum(L.(x))
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

    f, g, f_grad, g_jac, nu, ng, nx, nc
end
## Could try test_ippot_fcns.jl to make sure that it makes sense



# Not quite the rabi rate since its on the coefficients
max_rabi_rate = 2π * 0.060
c_L = -max_rabi_rate*ones(nc)
c_U = max_rabi_rate*ones(nc)

m = 2
g_L = [-Inf, -Inf]
g_U = [2.0, 1]
#g_U = [1, 1.0]

prob = createProblem(nc, c_L, c_U, ng, g_L, g_U, ng*nc, 0, f, g, f_grad, g_jac)

addOption( prob, "hessian_approximation", "limited-memory");
addOption( prob, "max_iter", 150);
addOption( prob, "print_level", 5);
#addOption( prob, "accept_after_max_steps", 10);



c0 = [0.01*ones(nsplines); zeros(nsplines)][:]# + 0.001randn(nsplines,2)[:]
prob.x .= c0 #c_opt[:]
#@time status = solveProblem(prob)
@time status = solveProblem(prob)

println(Ipopt.ApplicationReturnStatus[status])
#println(prob.x)

c_opt = reshape(prob.x, nsplines, nu)
u = transpose(B*c_opt)


cache = QuantumOptimalControl.setup_grape_cache(A0Δt, complex(x0), (2, segment_count))
x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0, cache)

println([sum(L.(x)), Jfinal(x[end])])
println([sum(L.(x)) / μ_state, Jfinal(x[end])])
println(prob.obj_val) # 17.01401714517915


@time f(vec(c_opt))
@time f_grad(vec(c_opt), copy(vec(c_opt)))
@time g(vec(c_opt), zeros(2))




x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)

v = [[x[k][i, j] for k=1:length(x)] for i=1:9, j=1:9]

using Plots
plt1 = plot();
plot!(plt1, t, abs2.(v[2]))


plts = []

#comp_basis_inds = [qb.state_dict[s] for s in ["00", "01", "10", "11"]]
comp_basis_inds = qb(["20", "21"])

for i=0:1, j=0:1

    l = qb.state_dict[string(i, j)]

    plt = plot(legend=(i==j==0), title="From state $(qb.state_labels[l])")
    for k in comp_basis_inds
        plot!(plt, t, abs2.(v[k,l]), label=qb.state_labels[k])
    end
    push!(plts, plt)
end

plt_u = plot(t[1:end-1], transpose(u), linetype=:steppost, title="Control signal")
plot!(plt_u, [t[1], t[end]], max_rabi_rate*[-1 1; -1 1], c="black", label=nothing)

layout = @layout [a b; c d; e]
plot(plts..., plt_u, layout=layout)


##

using AbstractFFTs, FFTW
u_cplx = u[1,:] + im*u[2,:]

Ω = fftfreq(length(u_cplx), 1/Δt)
U = fft(u_cplx)

Ω = -20:0.01:20
fft_mat = [exp(-im*ω*t) for ω in Ω, t in t[1:end-1]]
U = fft_mat * u_cplx



plot(Ω[Ω .> 0]/2π, abs.(U[Ω .> 0]), xscale=:log, yscale=:log);
plot!(-Ω[Ω .< 0]/2π, abs.(U[Ω .< 0]), xscale=:log, yscale=:log, c=:red)



# Cryptic error message?
# number of non-zero elemeents in hessian, cannot be negative
# why does nc need to be specified if c_L and c_U are given?
