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


x_target1 = [0 1 0; 1 0 0; 0 0 1]
x_target9 = kron(x_target1, I(3))

Q_css = I(9)[:, [1,2,4,5]]
x_target = kron([0 1; 1 0], I(2)) # NOT
#x_target = kron([1 0; 0 1], I(2)) # Id


#Jfinal = x -> 1 - abs(tr(x_target' * Q_css'*x*Q_css))/4 # Hmm, what is the gradient
Jfinal = x -> 1 - abs_sum_phase_calibrated(diag(x_target' * Q_css'*x*Q_css))/4

#=function ChainRulesCore.rrule(::typeof(Jfinal2), x)
    Jfinal_internal =  x -> 1 - abs_sum_phase_calibrated(diag(x_target' * Q_css'*x*Q_css))/4
    println(Jfinal_internal(x), ":", diag(x_target' * Q_css'*x*Q_css))
    return Jfinal_internal(x), xbar -> (NoTangent(), Zygote.gradient(Jfinal_internal, x)[1]::typeof(x) * xbar)
end=#


# Optimization parameters
max_rabi_rate = 2π * 0.060
#cutoff_freq = 2π* 300 * 1e6
num_drives = 1 # according to DRAG
#optimization_count = 10

##

Tgate = 20 # ns
segment_count = 200

t = LinRange(0, Tgate, segment_count + 1)

Δt = Tgate / segment_count

A0 = -im*H0
A1 = -im*(Tc + Tc')
A2 = -im*(im*(Tc - Tc'))

using BSplines
using Plots

nsplines = 20
spline_basis = BSplineBasis(4, LinRange(0,Tgate,nsplines+4))

Bpre = zeros(length(t)-1, nsplines+6)
t_midpoints = t[1:end-1] .+ Δt/2
for (k, t) in enumerate(t_midpoints)
    spl = bsplines(spline_basis, t)
    Bpre[k, collect(eachindex(spl))] .= spl.parent     
end
B = Bpre[:, 4:end-3]

#c0 = B \ iq_data

#plot(t[1:end-1], iq_data)
#plot!(t[1:end-1], B*c0)


#cache = QuantumOptimalControl.setup_grape_cache(A0, x0, (2, segment_count))
##

μ  = 100e-3
fg! = let

    cache = QuantumOptimalControl.setup_grape_cache(A0, x0, (2, segment_count))

    function (F,dc,c)
        
        L = c -> μ*(norm(c) + norm(diff(c, dims=1)))#1e-7*norm(c)

        u = transpose(B*c)
        x, λ, dJdu = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u, x0, cache; dUkdp_order=4)
        #println(dJdu)
        dJdc = B'*transpose(dJdu)

        if dc != nothing
            dc .= dJdc + Zygote.gradient(L, c)[1]
        end
        if F != nothing      
        return Jfinal(x[end]) + L(c)
        end
    end

end
  
#@btime x, λ, dJdu = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u, x0, cache; dUkdp_order=4)
#@time fg!(1, nothing, c0)

using Optim, Zygote


#@time opt_result =  Optim.optimize(Optim.only_fg!(fg!), c0 + 0.02*randn(nsplines,2), f_calls_limit=150)
@time opt_result =  Optim.optimize(Optim.only_fg!(fg!), 0.01*randn(nsplines,2), f_calls_limit=150)

#fg!(1, nothing, c_opt)- 1e-7norm(c_opt)

reg_terms = [μ * norm(opt_result.minimizer) μ*norm(diff(opt_result.minimizer, dims=1))]
println(opt_result.minimum)
println(fg!(1, nothing, opt_result.minimizer) - sum(reg_terms), ":", reg_terms[1],":", reg_terms[2])




c_opt = opt_result.minimizer
#c_opt = [0.03472746926963958 -0.038537642961308606; 0.1337770048392767 0.022329042032806157; 0.11518342934209276 0.04751345523848279; 0.09022634618326703 0.003401908326480719; 0.1916328866679226 -0.03756712862809127; 0.15004196460458463 -0.0281578091880935]

#x2, λ2, dJdu2 = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, , x0; dUkdp_order=3)


u = transpose(B*c_opt)
#u = u_mat[:,1:500]
x, λ, dJdu = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u, x0; dUkdp_order=3)
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


