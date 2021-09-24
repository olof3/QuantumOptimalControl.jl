using QuantumOptimalControl, DelimitedFiles

iq_data = 1e-9*DelimitedFiles.readdlm(joinpath(dirname(Base.find_package("QuantumOptimalControl")), "../examples/zz_coupling_pulse_tahereh210823.csv"))
u = copy(transpose(iq_data))

Q_css = qb[:, ["00", "01", "10", "11"]] # Orthogonal basis for the computational subspace
x0 = float(Q_css)

tgate = 20
Δt = tgate/500
A0Δt, A1Δt, A2Δt = QuantumOptimalControl.setup_bilinear_matrices(H0, Tc, Δt)

x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0)

include("plot_fcns.jl")
plot_2qubit_evolution(qb, 0:Δt:tgate, x, u)