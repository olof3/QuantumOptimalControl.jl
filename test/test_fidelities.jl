using Optim, QuantumOptimalControl, Test

J = (m, θ) -> abs(sum(m[1] + m[2]*cis(θ[1]) + m[3]*cis(θ[2]) + m[4]*cis(θ[1] + θ[2])))


function abs_trace_optimal(M)
    m = diag(M)
    J = θ -> abs(sum(m[1] + m[2]*cis(θ[1]) + m[3]*cis(θ[2]) + m[4]*cis(θ[1] + θ[2])))

    θ_grid = Iterators.product(LinRange(0,2π,100), LinRange(0,2π,100))
    θbest_grid = collect(argmax(J, θ_grid))

    res = optimize(θ -> -J(θ), (θbest_grid), x_abstol=1e-15)

    -res.minimum, res.minimizer
end


M = Diagonal([1, 1im, 1im, 1])

@test abs_trace_phase_calibrated(M) ≈ 2.8284271
@test abs_trace_phase_calibrated(M, :basic) ≈ 2.0
@test abs_trace_phase_calibrated(M, :grid) ≈ 2.8284271
@test abs_trace_optimal(M)[1] ≈ 2.8284271

θ_opt = QuantumOptimalControl.optimal_calibration(diag(M))[2]
@test J(diag(M), θ_opt) ≈ abs_trace_phase_calibrated(M) atol=1e-12


M = Diagonal([1, 0.1im, 0.1im, 1])

@test abs_trace_phase_calibrated(M) ≈ 2.0099751
@test abs_trace_phase_calibrated(M, :basic) ≈ 0.2 
@test abs_trace_phase_calibrated(M, :grid) ≈ 2.0099751
@test abs_trace_optimal(M)[1] ≈ 2.0099751

θ_opt = QuantumOptimalControl.optimal_calibration(diag(M))[2]
@test J(diag(M), θ_opt) ≈ abs_trace_phase_calibrated(M) atol=1e-12


M = Diagonal([cis(1), cis(2), cis(3), cis(4)])
@test abs_trace_phase_calibrated(M) ≈ 4
@test abs_trace_phase_calibrated(M, :basic) ≈ 4
@test abs_trace_phase_calibrated(M, :grid) ≈ 4 atol=1e-3
@test abs_trace_optimal(M)[1] ≈ 4
abs_trace_phase_calibrated(M, :lms_phase_semiold)

θ_opt = QuantumOptimalControl.optimal_calibration(diag(M))[2]
@test J(diag(M), θ_opt) ≈ abs_trace_phase_calibrated(M) atol=1e-12

##

M = Diagonal(cis.([1, 2, -2.5, -1.7]))

@test abs_trace_phase_calibrated(M) ≈ 3.995001
@test abs_trace_phase_calibrated(M, :basic) ≈ 3.995001 atol=0.01
@test abs_trace_phase_calibrated(M, :grid) ≈ 3.995001 atol=1e-3
@test abs_trace_optimal(M)[1] ≈ 3.995001

θ_opt = QuantumOptimalControl.optimal_calibration(diag(M))[2]

@test J(diag(M), θ_opt) ≈ abs_trace_phase_calibrated(M) atol=1e-8
@test QuantumOptimalControl.optimal_calibration(diag(M))[2] ≈ [5.383258515112539, 3.6000220820575084] atol=1e-4


##

M = Diagonal(cis.([2.5, 2.5, 1.5, -2.5]))




J_opt, θ_opt = QuantumOptimalControl.optimal_calibration(diag(M))
@test J_opt ≈ 3.365883939061934
@test J(diag(M), θ_opt) ≈ 3.365883939061934 
@test abs_trace_phase_calibrated(M) ≈ 3.365883939061934

##

M = Diagonal([0.65 - 0.75im, -0.4 + 0.8im, -0.4 + 0.1im, 0.7 - 0.0im])

@test abs_trace_phase_calibrated(M) ≈ 2.9787244710195484

J_opt, θ_opt = QuantumOptimalControl.optimal_calibration(diag(M), 1e-15)
@test J_opt ≈ 2.9787244710195484
@test J(diag(M), θ_opt) ≈ 2.9787244710195484


##


M = Diagonal([im, 1, -im, 1])
abs_trace_phase_calibrated(M) ≈ abs_trace_optimal(M)[1]


M = Diagonal([im, 1, im, -1]) # Problematic case
@test abs_trace_phase_calibrated(M, :lms_phase) ≈ abs_trace_optimal(M)[1]
#abs_trace_phase_calibrated(M, :lms_phase_semiold) # fails

M = Diagonal([im, 0.1, 0.1*im, -1])
@test abs_trace_phase_calibrated(M, :lms_phase) ≈ abs_trace_optimal(M)[1]

M = Diagonal([cis(2), 1, cis(-2), 1])
abs_trace_phase_calibrated(M, :lms_phase) ≈ abs_trace_optimal(M)[1]



#
# Test that the opitmal version is not that much worse than the gridded version
using Random
Random.seed!(0)
M_data = [Diagonal(rand(4) .* cis.(2π*rand(4))) for k=1:500]

F_optimal_test = [1 - abs_trace_optimal(M)[1]/4 for M in M_data]
F_optimal = [1 - abs_trace_phase_calibrated(M, :optimal)/4 for M in M_data]
F_grid = [1 - abs_trace_phase_calibrated(M, :grid)/4 for M in M_data]

@test all(F_grid - F_optimal .> -eps()) # :optimal should at least be better than :grid
@test all(F_grid - F_optimal .< 1e-3) # :grid should not be that much worse than :optimal

@test all(F_optimal_test - F_optimal  .< 1e-6)


println(mean(F_optimal_test - F_optimal))
@test all(F_optimal_test - F_optimal .>= 0)


## Test gradient computation
Jfun = m -> abs_trace_phase_calibrated(Diagonal(m), :optimal)^2

Random.seed!(100)

m_vec = [rand(4) .* cis.(2π*rand(4)) for k=1:2000]

for (k,m) in enumerate(m_vec)

    #println(k)
    J_opt, θ_opt = QuantumOptimalControl.optimal_calibration(m, 1e-12)
    @test J(m, θ_opt) ≈ J_opt atol=1e-14    

    grad_fd = FiniteDifferences.grad(central_fdm(9,1,max_range=0.001), Jfun2, m)[1]
    grad_analytic = QuantumOptimalControl.abs_trace_phase_calibrated_grad(m, θ_opt[1])

    #w1 = g_derivatives(m[1:2], θ_opt[1])
    #w2 = g_derivatives(m[3:4], θ_opt[1])
    #println(w1[1] + w2[1]) # Check that derivative is zero

    @test grad_fd ≈ grad_analytic rtol=1e-6
end


