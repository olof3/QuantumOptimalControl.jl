N = 3
θ0 = 0.25
t_plateau = 300.0
t_rise_fall = 50.0


# DEVICE PARAMETERS
ω1 = 4.5 * 2π
ω2 = 4.2 * 2π
ωc0 = 7.5 * 2π
α1 = -0.2 * 2π
α2 = -0.2 * 2π
g1 = 0.04 * 2π
g2 = 0.04 * 2π

# OPERATORS
a1, a2, ac = QuantumOptimalControl.annihilation_op(3, 3, 3)

# HAMILTONIAN
Hq1 = ω1 * a1'*a1 + α1 * a1'*a1 * (a1'*a1 - I)
Hq2 = ω2 * a2'*a2 + α2 * a2'*a2 * (a2'*a2 - I)
Hi1 = g1 * (a1' + a1) * (ac' + ac) #g1 * kron(a + a', I3, a + a')
Hi2 = g2 * (a2' + a2) * (ac' + ac) #g2 * kron(I3, a + a', a + a')
Hc = ωc0 * ac' * ac

H0 = Hq1 + Hq2 + Hi1 + Hi2

state_labels = map(e -> string("|", reverse(e)..., "⟩"), collect(Iterators.product(0:2, 0:2, 0:2))[:])

N_quanta = Int.(round.(diag(a1' * a1) + diag(a2' * a2) + diag(ac' * ac)))
to_keep = N_quanta .< 7

red_map = I(N^3)[to_keep, :]
N_red = count(to_keep)
#classes = [findall(N_quanta .== k) for k=0:6]

@parameters t u
@variables Uᵣ[1:N_red](t) Uᵢ[1:N_red](t)
D = Differential(t)
U = Uᵣ + im*Uᵢ

Htot = sparse(H0 + u*Hc)

symrhs = -1im * (red_map * Htot * red_map' * U)
symrhs = simplify.(symrhs)
symlhs = D.(U)

eqs = symlhs .~ symrhs

eqs_vec = vcat(eqs...) # Since complex-valued equations become a Vector of two equations
sys = ODESystem(eqs_vec, t, complex2real(U[:]), [u])

dxdt = generate_function(sys; expression=Val{false})[2]


# INITIAL AND TARGET STATE, RESONANCE (CZ: 11 -> 20)
initialstate = complex(float(kron([0, 1, 0], [0, 1, 0], [1, 0, 0]))) # |110>
targetstate = kron([0, 0, 1], [1, 0, 0], [1, 0, 0]) # |200>

#initialstate = kron(qt.basis(N, 1), qt.basis(N, 1), qt.basis(N, 0))
#targetstate = kron(qt.basis(N, 2), qt.basis(N, 0), qt.basis(N, 0))

ω_th = abs(H0[1 * N * N + 1 * N + 1, 1 * N * N + 1 * N + 1] - H0[2 * N * N + 1, 2 * N * N + 1])  # true eigenvalues are a better guess

function envelope(t, p)
    t_plateau = p[1]
    ω_Φ = p[2]
    A = p[3]
    θ = p[4]
    t_rise_fall = p[5]
    δ = if t > t_rise_fall / 2 && t <= t_rise_fall / 2 + t_plateau
        A
    elseif t <= t_rise_fall / 2
        A / 2 * (1 - cos(2π * t / t_rise_fall))
    elseif t > t_rise_fall / 2 + t_plateau
        A / 2 * (1 - cos(2π * (t - t_plateau) / t_rise_fall))
    end
    Φ = θ + δ *cos(ω_Φ * t)
    return sqrt(abs(cos(π * Φ)))
end


δ = 0.13
f_offset = -0.002

# ONE SHOT
ω_Φ = ω_th + f_offset * 2π  # ω_offset is given in GHz !!??!! Hmmm

p0 = [t_plateau, ω_Φ, δ, θ0, t_rise_fall]

tlist = LinRange(0.0, t_rise_fall + t_plateau, Int(round(2 * (t_rise_fall + t_plateau))))
plot(tlist, t -> envelope(t, p0))


function wrap_control(dxdt, u_fcn)
    return (dx, x, p, t) -> dxdt(dx, x, u_fcn(t,p), t)
end
dxdt_wrapped = wrap_control(dxdt, envelope)

Tgate = t_rise_fall + t_plateau

prob = ODEProblem{true}(dxdt_wrapped, (complex2real(red_map * initialstate)), (0.0, Tgate))

@time sol = solve(prob, Tsit5(), p=p0, saveat=[Tgate], adaptive=false, dt=0.1e-3)

finalstate = sol.u[end]
cost = abs2(real2complex(finalstate)' * (red_map*targetstate))
println(cost) # Should be something like 0.937218