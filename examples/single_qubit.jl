using LinearAlgebra: I
using SparseArrays: sparse

using DifferentialEquations
using DiffEqFlux: sciml_train
using DiffEqSensitivity: ForwardDiffSensitivity
using ModelingToolkit
using Optim
using Zygote

using QuantumOptimalControl

# natural units of GHz and ns

using Plots
plotly()

Tgate = 30

@parameters t Δ Ax Ay 
@variables Uᵣ[1:3,1:3](t) Uᵢ[1:3,1:3](t)
D = Differential(t)

U = Uᵣ + im*Uᵢ

H0 = diagm([0, 0, Δ])

# quadrature control Hamiltonians
Hx = [0 1 0; 1 0 √2; 0 √2 0]
Hy = [0 -1im 0; 1im 0 -1im*√2; 0 1im*√2 0]

Htot =  sparse(H0 + Ax*Hx + Ay*Hy)
Htot =  sparse(H0 + Ax*Hx + Ay*Hy)

rhs = -1im * (Htot * U)
rhs = simplify.(rhs)
lhs = D.(U)

eqs = lhs .~ rhs

eqs_vec = vcat(eqs...) # Since complex-valued equations become a Vector of two equations

#
# now use ModelingToolkit to build a specialized function for these equations
# ODESystem expects 1D arrays so vec both sides
# use explicit form to get consistent order of parameters
# sys = ODESystem(vec(eqs))

sys = ODESystem(eqs_vec, t, complex2real(U[:]), [Δ, Ax, Ay])

# second return value is in-place version and we'll go head and evaluate the expression
#display(generate_function(sys; expression=Val{true})[2])
dudt = generate_function(sys; expression=Val{false})[2]


##

# build the DE function with time dependent controls

# basis of sinusoids to ensure the pulse goes to zero and the start and finish
# the parameters `p` give the N basis coefficients: with 1:N for the X controls and N+1:2N for the Y controls
function controls(t, p)
    num_controls = length(p) ÷ 2
    Ax = sum(a*sinpi(k*t/Tgate) for (k,a) in enumerate(p[1:num_controls]))
    Ay = sum(a*sinpi(k*t/Tgate) for (k,a) in enumerate(p[(num_controls+1):end]))
    Ax, Ay
end

# wrapper function evaluates the time dependent controls and then passes them on
function dudt_sinebasis(du, u, p, t)
    Ax, Ay = controls(t, p[2:end])
    dudt(du, u, [p[1], Ax, Ay], t)
end


H_drag = (dx, x, p, t) -> dudt(dx, x, u_drag(p,t), t)
H_sinebasis = (dx, x, p, t) -> dudt(dx, x, u_sinebasis(p,t), t)


# x captures the unitary or the state (column-wise to capture both?)
# f(x) gives the evolution of unitary or state..


function propagate(f, tfinal, x0, p)

    prob = ODEProblem{true}(f, U₀, (0.0,t[end]), p, saveat=t)
    sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)
    
    sol.u
    # return vector or scalar
end


U₀ = complex2real(Matrix{ComplexF64}(I, 3, 3)[:])

##
# look at the populations under something a little over a π pulse as a sanity check
##

prob = ODEProblem{true}(dudt_sinebasis, U₀, (0.0,Tgate), vcat(2pi*-0.2, [0.1, 0.0]))
@time sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)


t = sol.t
Ut_vec = [reshape(real2complex(x), 3, 3) for x in sol.u]

p = [plot(), plot(), plot()]
for j=1:3    
    for i=1:3
        # Use abs2 or abs?
        plot!(p[j], t, [abs2(Ut[i,j]) for Ut in Ut_vec], label=(j==1 ? "|$(i-1)⟩" : nothing), title="Evolution from |$(j-1)⟩")
    end
end
layout = @layout [a b c]
ptot = plot(p..., layout=layout, size=(1700,600))
display(ptot)

##

# lock Δ to -200 MHz for now
# docs say ForwardDiffSensitivity is best for small systems

# need to throw in real for  on parameters..

prob = ODEProblem{true}(dudt_sinebasis, U₀, (0.0,Tgate))
function compute_U(p)    
    sol = solve(prob, Tsit5(), p=[2π*-0.2; real(p)], saveat=Tgate, save_start=false, abstol=1e-8, reltol=1e-8, sensealg=ForwardDiffSensitivity())
    Uv = sol.u[end]
    reshape(real2complex(Uv), 3, 3)    
end



Ugoal = zeros(3, 3)
Ugoal[1:2, 1:2] = [0 1; 1 0]  # X180
Ugoal_adjoint = copy(Ugoal')
infidelity(U) = 1 - abs2(tr(Ugoal_adjoint*U))/4

function cost_fcn(p)
    U = compute_U(p)
    infidelity(U)
end

cost_fcn([0.1, 0.0])
##

# double check that we can take a gradient on either side of the minimum
gradient(cost_fcn, [0.05, 0.0])
display(gradient(cost_fcn, [0.11, 0.0]))

# lets look at the optimized controls for varying numbers of basis functions
opt_runs = []
for n=1:3
    p = 0.01*randn(2n)
    res = sciml_train(cost_fcn, p, BFGS(initial_stepnorm=1e-6))
    push!(opt_runs, res)
end

##
# plot the resulting optimized controls
#=  =#using Printf: @sprintf
ts = range(0, Tgate; length=51)
for k=1:length(opt_runs)
    opt_p = opt_runs[k].minimizer
    ax = plot()
    plot!(ax, ts, [controls(t, opt_p)[1]/2π/1e-3 for t in ts], label="Ωx")
    plot!(ax, ts, [controls(t, opt_p)[2]/2π/1e-3 for t in ts], label="Ωy")
    xlabel!(ax, "Time (ns)")
    ylabel!(ax, "Pulse Amplitude (MHz)")
    title!(ax, @sprintf("%d Quadrature Controls with Error = %.3e", k, opt_runs[k].minimum))
    display(ax)
end

