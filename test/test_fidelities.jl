using QuantumOptimalControl
using Test, Random, Optim

f = (m, θ) -> abs(m[1] + m[2]*cis(θ[2]) + m[3]*cis(θ[1]) + m[4]*cis(θ[1] + θ[2]))

function abs_sum_optimal(m)
    f = θ -> abs(sum(m[1] + m[2]*cis(θ[2]) + m[3]*cis(θ[1]) + m[4]*cis(θ[1] + θ[2])))

    θ_grid = Iterators.product(LinRange(0,2π,100), LinRange(0,2π,100))
    θbest_grid = collect(argmax(J, θ_grid))

    res = optimize(θ -> -f(θ), (θbest_grid), x_abstol=1e-15)

    -res.minimum, res.minimizer
end


m = [1, 1im, 1im, 1]

@test abs_sum_phase_calibrated(m) ≈ 2.8284271
@test abs_sum_phase_calibrated(m, :basic) ≈ 2.0
@test abs_sum_phase_calibrated(m, :grid) ≈ 2.8284271
@test abs_sum_optimal(m)[1] ≈ 2.8284271

θ_opt = QuantumOptimalControl.optimal_calibration(m)[2]
@test f(m, θ_opt) ≈ abs_sum_phase_calibrated(m) atol=1e-12


m = [1, 0.1im, 0.1im, 1]

@test abs_sum_phase_calibrated(m) ≈ 2.0099751
@test abs_sum_phase_calibrated(m, :basic) ≈ 0.2
@test abs_sum_phase_calibrated(m, :grid) ≈ 2.0099751
@test abs_sum_optimal(m)[1] ≈ 2.0099751

θ_opt = QuantumOptimalControl.optimal_calibration(m)[2]
@test f(m, θ_opt) ≈ abs_sum_phase_calibrated(m) atol=1e-12


m = [cis(1), cis(2), cis(3), cis(4)]
@test abs_sum_phase_calibrated(m) ≈ 4
@test abs_sum_phase_calibrated(m, :basic) ≈ 4
@test abs_sum_phase_calibrated(m, :grid) ≈ 4 atol=1e-3
@test abs_sum_optimal(m)[1] ≈ 4

θ_opt = QuantumOptimalControl.optimal_calibration(m)[2]
@test f(m, θ_opt) ≈ abs_sum_phase_calibrated(m) atol=1e-12

##

m = cis.([1, 2, -2.5, -1.7])

@test abs_sum_phase_calibrated(m) ≈ 3.995001
@test abs_sum_phase_calibrated(m, :basic) ≈ 3.995001 atol=0.01
@test abs_sum_phase_calibrated(m, :grid) ≈ 3.995001 atol=1e-3
@test abs_sum_optimal(m)[1] ≈ 3.995001

θ_opt = QuantumOptimalControl.optimal_calibration(m)[2]

@test f(m, θ_opt) ≈ abs_sum_phase_calibrated(m) atol=1e-8
@test QuantumOptimalControl.optimal_calibration(m)[2] ≈ [5.383258515112539, 3.6000220820575084] atol=1e-4


##

m = cis.([2.5, 2.5, 1.5, -2.5])


J_opt, θ_opt = QuantumOptimalControl.optimal_calibration(m)
@test J_opt ≈ 3.365883939061934
@test f(m, θ_opt) ≈ 3.365883939061934
@test abs_sum_phase_calibrated(m) ≈ 3.365883939061934

##

m = [0.65 - 0.75im, -0.4 + 0.8im, -0.4 + 0.1im, 0.7 - 0.0im]

@test abs_sum_phase_calibrated(m) ≈ 2.9787244710195484

J_opt, θ_opt = QuantumOptimalControl.optimal_calibration(m, 1e-15)
@test J_opt ≈ 2.9787244710195484
@test f(m, θ_opt) ≈ 2.9787244710195484


##


m = [im, 1, -im, 1]
abs_sum_phase_calibrated(m) ≈ abs_sum_optimal(m)[1]


m = [im, 1, im, -1] # Problematic case
@test abs_sum_phase_calibrated(m, :lms_phase) ≈ abs_sum_optimal(m)[1]
#abs_sum_phase_calibrated(m, :lms_phase_semiold) # fails

m = [im, 0.1, 0.1*im, -1]
@test abs_sum_phase_calibrated(m, :lms_phase) ≈ abs_sum_optimal(m)[1]

m = [cis(2), 1, cis(-2), 1]
abs_sum_phase_calibrated(m, :lms_phase) ≈ abs_sum_optimal(m)[1]



#
# Test that the opitmal version is not that much worse than the gridded veraasion
Random.seed!(0)
m_data = [rand(4) .* cis.(2π*rand(4)) for k=1:500]

F_optimal_test = [1 - abs_sum_optimal(m)[1]/4 for m in m_data]
F_optimal = [1 - abs_sum_phase_calibrated(m, :optimal)/4 for m in m_data]
F_grid = [1 - abs_sum_phase_calibrated(m, :grid)/4 for m in m_data]

F_approx = [1 - QuantumOptimalControl.optimal_calibration(m, approximate_optimum=true)[1]/4 for m in m_data]

@test all(F_grid - F_optimal .> -eps()) # :optimal should at least be better than :grid
@test all(F_grid - F_optimal .< 1e-3) # :grid should not be that much worse than :optimal

@test all(F_optimal_test - F_optimal  .< 1e-6)


println(mean(F_optimal_test - F_optimal))
@test all(F_optimal_test - F_optimal .>= 0)


## Test gradient computation
using FiniteDifferences
Jfun = m -> abs_sum_phase_calibrated(m, :optimal)^2

Random.seed!(100)

m_vec = [rand(4) .* cis.(2π*rand(4)) for k=1:2000]

for (k,m) in enumerate(m_vec)

    #println(k)
    J_opt, θ_opt = QuantumOptimalControl.optimal_calibration(m, 1e-12)
    @test f(m, θ_opt) ≈ J_opt atol=1e-14

    grad_fd = FiniteDifferences.grad(central_fdm(9,1,max_range=0.001), Jfun, m)[1]
    grad_analytic = QuantumOptimalControl.abs_sum_phase_calibrated_grad(m, θ_opt[1])

    #w1 = g_derivatives(m[1:2], θ_opt[1])
    #w2 = g_derivatives(m[3:4], θ_opt[1])
    #println(w1[1] + w2[1]) # Check that derivative wrt θ is zero

    @test grad_fd ≈ grad_analytic rtol=1e-6
end



using ChainRulesTestUtils

m = [0.65 - 0.75im, -0.4 + 0.8im, -0.4 + 0.1im, 0.7 - 0.0im]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, reltol=)

m = [im, 1, -im, 1]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, reltol=1e-4)

m = [im, 1, im, -1]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, rtol=1e-7)

m = [im, 0.1, 0.1*im, -1]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, rtol=1e-7)

m = [cis(2), 1, cis(-2), 1]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, rtol=1e-7)


m = [0.65 - 0.75im, -0.4 + 0.8im, -0.4 + 0.1im, 0.7 - 0.0im]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m)
