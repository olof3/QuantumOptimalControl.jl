using QuantumOptimalControl
using DelimitedFiles

iq_data = 1e-9*DelimitedFiles.readdlm(joinpath(dirname(Base.find_package("QuantumOptimalControl")), "../examples/pulse_210823.csv"))

u = iq_data[:,1] + im*iq_data[:,2]
u_mat = transpose(iq_data)

u = copy(transpose(iq_data))

# System parameters

dimq = 3
dims = 3
Ntot = dimq * dims
α_q = 2π * 0.2 
α_s = 2π * 0.2
X_dispersive = 2π  * 1e-4

a_q = QuantumOptimalControl.annihilation_op(dimq)
a_s = QuantumOptimalControl.annihilation_op(dims)

# Specify system Hamiltonians
Hq = - α_q/2 * kron(a_q'*a_q'*a_q*a_q, I(dims)) # + wq * kron(dot(q',q), eye(dims))
Hspectator = - α_s/2 * kron(I(dimq), a_s'*a_s'*a_s*a_s) # + ws * kron(eye(dimq), dot(a',a))
Hint = -X_dispersive * kron(a_q'*a_q, a_s'*a_s)

# Define drive control operators
Tc = kron(a_q', I(dims))

H0 = Hq + Hspectator + Hint

x0 = 1.0I[1:9, 1:9]

# Get the projection on the full 
#x_target_qubit1 = [0 1 0; 1 0 0; 0 0 1]
#x_target_full = kron(x_target_qubit1, I(3))

Q_css = I(9)[:, [1,2,4,5]] # Should be possible to use just I[:, [1,2,4,5]] in julia 1.8
x_target = kron([0 1; 1 0], I(2)) # NOT
#x_target = kron([1 0; 0 1], I(2)) # Id

#Jfinal = x -> 1 - abs(tr(x_target' * Q_css'*x*Q_css))/4 # Hmm, what is the gradient
Jfinal = x -> 1 - abs_sum_phase_calibrated(diag(x_target' * Q_css'*x*Q_css))/4

# Optimization parameters
max_rabi_rate = 2π * 0.060
#cutoff_freq = 2π* 300 * 1e6

##

tgate = 20 # ns
segment_count = 200

t = LinRange(0, tgate, segment_count + 1)

Δt = tgate / segment_count

A0 = -im*H0
A1 = -im*(Tc + Tc')
A2 = -im*(im*(Tc - Tc'))

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

f, g, f_grad, g_jac = let x0=x0

    nu = 2    
    ng = 2
    nx = length(x0)
    nc = nu * nsplines

    c_prev = Vector{Float64}(undef, nc)
    cache = QuantumOptimalControl.setup_grape_cache(A0, complex(x0), (2, segment_count))

    f = function(c)
        c_prev .= c
        c = reshape(c, nsplines, nu)
        u = transpose(B*c)
        x, λ, dJdu = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u, x0, cache; dUkdp_order=4)
        
        Jfinal(x[end])
    end

    f_grad = function(c, f_grad_out)
        # Todo: check cache
        if c_prev != c
            println("gradient not computed", c[1:2])            
            f(c) # Use side effects in f
        end
        dJdc = B'*transpose(cache.dJdu)

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
            g_jac_tmp = Zygote.jacobian(g_oop, c)[1]            
            g_jac_out .= transpose(g_jac_tmp)[:]
        end        
    end

    f, g, f_grad, g_jac
end

##

using FiniteDiff

nu = 2
nc = nu * nsplines

c0 = randn(nsplines, 2)
df = similar(vec(c0))

c = reshape(c0, nsplines, nu)
u = transpose(B*c)
cache = QuantumOptimalControl.setup_grape_cache(A0, complex(x0), (2, segment_count))
x, λ, dJdu = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u, x0, cache; dUkdp_order=4)
vec(B'*transpose(dJdu))

# Compare gradient of f
f(vec(c0))
f_grad1 = f_grad(vec(c0), df)
f_grad2 = FiniteDiff.finite_difference_gradient(f, vec(c0))[:]
df - f_grad2

gout = zeros(2)
g(c0, dg)

# Compare jacobian of g
rows = zeros(Int32, ng * nc)
cols = zeros(Int32, ng * nc)
g_jac(c0, :Structure, rows, cols, values)

values = zeros(Float64, ng * nc)

g_jac(c0, :None, rows, cols, values)
g_jac1 = zeros(ng, nc)
for k=1:length(rows); g_jac1[rows[k], cols[k]] = values[k]; end

g_jac2 = FiniteDiff.finite_difference_jacobian(c -> g(vec(c), zeros(2)), vec(c0))

g_jac1 - g_jac2

##

c_L = -0.3ones(nc)
c_U = 0.3ones(nc)

m = 2
g_L = [-Inf, -Inf]
g_U = [1, 0.2]

@time prob = createProblem(nc, c_L, c_U, ng, g_L, g_U, ng*nc, 0, f, g, f_grad, g_jac)

addOption( prob, "hessian_approximation", "limited-memory");
addOption( prob, "max_iter", 100);

#prob.x = 0.01*randn(nc)
prob.x = [0.01*ones(nsplines); zeros(nsplines)][:]
@time status = solveProblem(prob)

println(Ipopt.ApplicationReturnStatus[status])
println(prob.x)


c_opt = reshape(prob.x, nsplines, nu)
u = transpose(B*c_opt)

f(vec(c_opt))
g(vec(c_opt), zeros(2))

println(prob.obj_val) # 17.01401714517915


##
u_cplx = u[1,:] + im*u[2,:]

Ω = fftfreq(200, 1)
U = fft(u_cplx)

plot(Ω, abs.(U))






##

#x, λ, dJdu = @btime QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u, x0; dUkdp_order=3)

v = [[x[k][i, 2] for k=1:length(x)] for i=1:9]

using Plots
plt1 = plot();
plot!(plt1, t, abs2.(v[2]))
for k=1:9
    plot!(plt1, t, abs2.(v[k]), legend=false)
end

plt2 = plot(t[1:end-1], transpose(u))
plot!(plt2, [t[1], t[end]], max_rabi_rate*[-1 1; -1 1], c="black")

layout = @layout [a; b]
plot(plt1, plt2, layout=layout)



# Cryptic error message?
# number of non-zero elemeents in hessian, cannot be negative
# why does nc need to be specified if c_L and c_U are given?
