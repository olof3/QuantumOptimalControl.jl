using QuantumOptimalControl, DelimitedFiles
using Zygote

include("../examples/models/cavity_qubit.jl")

examples_dir = dirname(Base.find_package("QuantumOptimalControl"))
iq_data = DelimitedFiles.readdlm(joinpath(examples_dir, "../examples/cavity_qubit_pulse_marina.csv"))
u_data = 1e-9 * [Complex(r...) for r in eachrow(iq_data)] # Going over to GHz

Δt = 1.0
u = hcat([[reim(u)...] for u in u_data]...)

#Δt = 0.5
#u = kron(u, [1 1])

A0, A1, A2 = QuantumOptimalControl.setup_bilinear_matrices(H0, Tc/2) # Note factor 1/2!
A0Δt, A1Δt, A2Δt = QuantumOptimalControl.setup_bilinear_matrices(H0, Tc/2, Δt)

#x_target = normalize!(kron([1, 0], exp.(1im*theta)))[:,:]

x0 = [normalize!([ones(N_cavity); zeros(N_cavity)]) normalize!([zeros(N_cavity); ones(N_cavity)])]
x_target = [normalize!(kron([1,1], exp.(1im*theta))) normalize!([zeros(N_cavity); ones(N_cavity)])]

Jfinal = x -> 1 - abs(tr(x_target' * x))
dJfinal_dx = x -> Zygote.gradient(Jfinal, x)[1]

Nt = 100
##
A0Δt, A1Δt, A2Δt = A0*Δt, A1*Δt, A2*Δt
cache = QuantumOptimalControl.setup_grape_cache(A0, complex(x0[:,:]), (2, Nt))

x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u[:,1:Nt], x0[:,:], cache)
println(Jfinal(x[end]))

@time dJdu = QuantumOptimalControl.grape_sensitivity(A0Δt, [A1Δt, A2Δt], dJfinal_dx, u[:,1:Nt], x0, cache; dUkdp_order=3)

#@btime QuantumOptimalControl.propagate($A0, [$A1, $A2], $u[:,1:$Nt], $x0, $cache)
#@btime QuantumOptimalControl.grape_sensitivity($A0, [$A1, $A2], $Jfinal, $u[:,1:$Nt], $x0, $cache; dUkdp_order=3)

display(dJdu)

## DiffEq-based version

include("../examples/models/setup_diffeq_rhs.jl")
dxdt = setup_dxdt(A0, [A1, A2])
dλdt = setup_dλdt(A0, [A1, A2])

cache2 = QuantumOptimalControl.setup_grape_cache(A0, c2r(x0[:,:]), (2, Nt))

@time sol = propagate_pwc(dxdt, c2r(x0[:,:]), u[:,1:Nt], Δt, cache2)
@btime dJdu2 = compute_pwc_gradient(dλdt, dJfinal_dx, u[:,1:Nt], Δt, A0, [A1, A2], cache2; dUkdp_order=3)

println(Jfinal(r2c(sol.u[end])))
display(dJdu2)


## Finer grid but 1st order approximation of d(exp(H(t)Δt))/duk

m = 10
Δt2 = Δt / m
u2 = kron(u[:,1:Nt], ones(1,m))
Nt2 = Nt * m


#G = setup_pwc_sensitivity_fcn(A0, [A1, A2])

cache = QuantumOptimalControl.setup_grape_cache(A0, c2r(x0[:,:]), (2, Nt2))

sol = propagate_pwc(dxdt, c2r(x0[:,:]), u2, Δt2, cache; dt=0.5Δt2)
@time dJdu = compute_pwc_gradient(dλdt, dJfinal_dx, u2, Δt2, A0, [A1, A2], cache; dUkdp_order=1, dt=0.5Δt2)

println(Jfinal(r2c(sol.u[end])))


function compute_sensitivity!(out, A_bl, x, λ)
    for k=1:size(out,2)
        for j=1:length(A_bl)
            #@views out[j,k] = Δt * sum(real(dot(r2c(λ[end-k+1])[:,l], A_bl[j], r2c(x[k])[:,l])) for l=1:size(x[1],2))
            out[j,k] = QuantumOptimalControl._compute_u_sensitivity(r2c(x[k]), r2c(λ[end-(k+1)+1]), A_bl[j])
        end
    end
    out
end
@time compute_sensitivity!(cache2.dJdu, [A1, A2], cache2.x, cache2.λ)

@btime QuantumOptimalControl._compute_u_sensitivity(r2c($cache2.x[2]), r2c($cache2.λ[3]), $A1)
display(sum(cache.dJdu[:, k:m:end] for k=1:m) ./ m)


## Finite diff

using FiniteDiff
obj = u -> Jfinal(r2c(propagate_pwc(dxdt, c2r(x0[:,:]), u, Δt).u[end]))
dJdu3 = FiniteDiff.finite_difference_gradient(obj, u[:, 1:Nt])
display(dJdu3)

obj = u -> Jfinal(QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)[end])
dJdu4 = FiniteDiff.finite_difference_gradient(obj, u[:, 1:Nt])
display(dJdu4)



## Test gradient with state penalty

include("../examples/models/zz_coupling.jl")
A0Δt, A1Δt, A2Δt = QuantumOptimalControl.setup_bilinear_matrices(H0, Tc, Δt)

Nt = 50

Jfinal = x -> 0.0*norm(x)#1 - abs_sum_phase_calibrated(diag(x_target' * Q_css'*x*Q_css))/4
#Jfinal = x -> 1 - norm(x_target' * x)
qb = QuantumOptimalControl.QuantumBasis([3,3])

Q_css = qb[:, ["00", "01", "10", "11"]]
Q_penalty = qb[:, ["20", "21", "22"]]

inds_css = getindex.(Ref(qb.state_dict), ["00", "01", "10", "11"])
inds_penalty = getindex.(Ref(qb.state_dict), ["20", "21", "22"])

L, dL_dx = QuantumOptimalControl.setup_state_penalty(inds_penalty, inds_css)



cache = QuantumOptimalControl.setup_grape_cache(A0, complex(x0), (2, Nt))
@time x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u[:,1:Nt], x0, cache)
@time dJdu = QuantumOptimalControl.grape_sensitivity(A0Δt, [A1Δt, A2Δt], dJfinal_dx, u[:,1:Nt], x0, cache; dUkdp_order=4, dL_dx=dL_dx)
display(dJdu)

#obj = u -> Jfinal(QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)[end])
obj = u -> sum(L.(QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)))
dJdu4 = FiniteDiff.finite_difference_gradient(obj, u[:, 1:Nt])
display(dJdu2)
