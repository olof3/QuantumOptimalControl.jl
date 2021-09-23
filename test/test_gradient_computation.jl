using QuantumOptimalControl, DelimitedFiles
using Zygote

include("../examples/models/cavity_qubit.jl")

examples_dir = dirname(Base.find_package("QuantumOptimalControl"))
iq_data = DelimitedFiles.readdlm(joinpath(examples_dir, "../examples/cavity_qubit_control.csv"))
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

dJdu = QuantumOptimalControl.grape_sensitivity(A0Δt, [A1Δt, A2Δt], dJfinal_dx, u[:,1:Nt], x0, cache; dUkdp_order=3)

#@btime QuantumOptimalControl.propagate($A0, [$A1, $A2], $u[:,1:$Nt], $x0, $cache)
#@btime QuantumOptimalControl.grape_sensitivity($A0, [$A1, $A2], $Jfinal, $u[:,1:$Nt], $x0, $cache; dUkdp_order=3)

display(dJdu)

## DiffEq-based version

include("../examples/models/setup_diffeq_rhs.jl")
dxdt = setup_dxdt(A0, [A1, A2])
dλdt = setup_dλdt(A0, [A1, A2])

cache2 = QuantumOptimalControl.setup_grape_cache(A0, c2r(x0[:,:]), (2, Nt))

@time sol = propagate_pwc(dxdt, c2r(x0[:,:]), u[:,1:Nt], Δt, cache2)
#@time sol_adj, dJdu = QuantumOptimalControl.compute_pwc_gradient(dλdt, Jfinal, u[:,1:Nt], Δt, A0, A, cache; dUkdp_order=3)
@time dJdu2 = compute_pwc_gradient(dλdt, Jfinal, u[:,1:Nt], Δt, A0, [A1, A2], cache2; dUkdp_order=3)

println(Jfinal(r2c(sol.u[end])))
display(dJdu2)


##

m = 10
Δt2 = Δt / m
u2 = kron(u[:,1:Nt], ones(1,m))
Nt2 = Nt * m


#G = setup_pwc_sensitivity_fcn(A0, [A1, A2])

cache = QuantumOptimalControl.setup_grape_cache(A0, c2r(x0[:,:]), (2, Nt2))

sol = propagate_pwc(dxdt, c2r(x0[:,:]), u2, Δt2, cache; dt=0.5Δt2)
sol_adj, _ = compute_pwc_gradient(dλdt, Jfinal, u2, Δt2, A0, [A1, A2], cache; dUkdp_order=0, dt=0.5Δt2)

println(Jfinal(r2c(sol.u[end])))


function compute_sensitivity!(out, A_bl, x, λ, u)
    for k=1:size(u,2)
        for j=1:length(A_bl)
            @views out[j,k] = sum(real(dot(r2c(λ[end-k+1])[:,l], A_bl[j], r2c(x[k])[:,l])) for l=1:size(x[1],2))
        end
    end
    out
end
@time compute_sensitivity!(cache.dJdu, [A1, A2], cache.x, cache.λ, u2)
display(Δt*sum(cache.dJdu[:, k:m:end] for k=1:m) ./ m)


## Finite diff




using FiniteDiff
obj = u -> Jfinal(r2c(propagate_pwc(dxdt, c2r(x0[:,:]), u, Δt).u[end]))
dJdu3 = FiniteDiff.finite_difference_gradient(obj, u[:, 1:Nt])
display(dJdu3)

# 0.0172903  0.0130303  0.00783313  0.00181776  -0.00486551  …  -0.0908296  -0.0940295   -0.0950277   -0.0939159
# 0.0475966  0.0520151  0.0557292   0.058564     0.0603646       0.0234938   0.00832458  -0.00675527  -0.0214246

obj = u -> Jfinal(QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)[end])
dJdu4 = FiniteDiff.finite_difference_gradient(obj, u[:, 1:Nt])
display(dJdu4)



##

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



#dJfinaldx = Zygote.gradient(Jfinal, x[20])[1]
##


cache = QuantumOptimalControl.setup_grape_cache(A0, complex(x0), (2, Nt))
@time x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u[:,1:Nt], x0, cache)
@time dJdu = QuantumOptimalControl.grape_sensitivity(A0Δt, [A1Δt, A2Δt], dJfinal_dx, u[:,1:Nt], x0, cache; dUkdp_order=4, dL_dx=dL_dx)
display(dJdu)

#obj = u -> Jfinal(QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)[end])
obj = u -> sum(L.(QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)))
dJdu4 = FiniteDiff.finite_difference_gradient(obj, u[:, 1:Nt])
display(dJdu4)



#=
Nt = 1

A0 = im*[-1 2; 1 -3]
A1 = [0 1; -1 0]
A2 = [0 1; 1 0]
x0 = 1.0Matrix(I, 2, 2)
x_target = [0 1.0; 1.0 0]
Jfinal = x -> 1 - abs(tr(x_target' * x))/4
x, λ, dJdu = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u_mat[:,1:Nt], x0; dUkdp_order=3)

obj = u -> Jfinal(QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u, x0; dUkdp_order=3)[1][end])
obj2 = x0 -> Jfinal(QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u_mat[:,1:Nt], x0; dUkdp_order=3)[1][end])
[obj(u_mat[:,1:Nt]) Jfinal(x[end])]
dJdu4 = FiniteDiff.finite_difference_gradient(obj, u_mat[:,1:Nt])

FiniteDiff.finite_difference_gradient(Jfinal, x[end])

FiniteDiff.finite_difference_gradient(, x[end])

dJdx0 = FiniteDiff.finite_difference_gradient(obj2, complex(x0))

println(dJdu)
println(dJdu4)

FiniteDiff.finite_difference_gradient(u_mat -> Jfinal(exp((A0 + u_mat[1,1]*A1 + u_mat[2,1]*A2)*Δt)), u_mat[:,1])

FiniteDiff.finite_difference_gradient(u_mat -> Jfinal(exp((A0 + u_mat[1,1]*A1 + u_mat[2,1]*A2)*Δt)), u_mat[:,1])

dU10 = FiniteDiff.finite_difference_derivative(u1 -> exp((A0 + u1*A1 + u_mat[2,1]*A2)*Δt), u_mat[1,1])
dU20 = FiniteDiff.finite_difference_derivative(u2 -> exp((A0 + u_mat[1,1]*A1 + u2*A2)*Δt), u_mat[2,1])
real(sum(conj(λ[end]) .* dU10))
real(sum(conj(λ[end]) .* dU20))

dUkdu = [similar(1.0A0) for k=1:2]
tmp = [similar(1.0A0) for k=1:3]
QuantumOptimalControl.expm_jacobian!(dUkdu, Δt*A0, [Δt*A1, Δt*A2], u_mat[:,1], tmp, 3)
=#




#=
using FiniteDifferences

Nt = 30

obj = u -> Jfinal(QuantumOptimalControl.grape_naive(Δt*A0, [Δt*A1, Δt*A2], Jfinal, u, x0, ; dUkdp_order=3)[1][end])

dJdu3 = FiniteDiff.finite_difference_gradient(obj, u_mat[:, 1:Nt])
dJdu4 = FiniteDifferences.grad(central_fdm(7,1,max_range=1e-8), obj, u_mat[:, 1:Nt])[1]

x, λ, dJdu = QuantumOptimalControl.grape_naive(A0 * Δt, [A1 * Δt, A2 * Δt], Jfinal, u_mat[:,1:Nt], x0; dUkdp_order=3)

r = FiniteDiff.finite_difference_gradient(Jfinal, x[end])

Jfinal(x[end])

lamba_end = Zygote.gradient(Jfinal, x[end])[1]
lamba_end = FiniteDiff.finite_difference_gradient(Jfinal, x[end])

cache = ([similar(c2r(x0)) for k=1:Nt+1], [similar(c2r(x0)) for k=1:Nt+1], Matrix{real(eltype(A0))}(undef, 2, Nt))

@time sol = propagate_pwc(dxdt, c2r(x0), u_mat[:,1:500], Δt, cache)
#@time sol_adj, dJdu = QuantumOptimalControl.compute_pwc_gradient(dλdt, Jfinal, u[:,1:Nt], Δt, A0, A, cache; dUkdp_order=3)
@time sol_adj, dJdu2 = compute_pwc_gradient(dλdt, Jfinal, u_mat[:,1:Nt], Δt, A0, [A1, A2], cache; dUkdp_order=3)
display(dJdu2)

display(dJdu4)



x, λ, dJdu = QuantumOptimalControl.grape_naive(A0, [A1, A2], Jfinal, u_mat[:, 1:Nt], x0; dUkdp_order=3)
display(dJdu)
=#
