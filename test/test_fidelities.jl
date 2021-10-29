using QuantumOptimalControl
using Test, Random, Optim

f = (v, θ) -> abs(v[1] + v[2]*cis(θ[2]) + v[3]*cis(θ[1]) + v[4]*cis(θ[1] + θ[2]))

function abs_sum_optimal(v)
    f = θ -> abs(sum(v[1] + v[2]*cis(θ[2]) + v[3]*cis(θ[1]) + v[4]*cis(θ[1] + θ[2])))

    θ_grid = Iterators.product(LinRange(0,2π,100), LinRange(0,2π,100))
    θbest_grid = collect(argmax(f, θ_grid))

    res = optimize(θ -> -f(θ), (θbest_grid), x_abstol=1e-15)

    -res.minimum, res.minimizer
end


v =  [1, 1im, 1im, 1]

@test abs_sum_phase_calibrated(v) ≈ 2.8284271
@test abs_sum_phase_calibrated(v, :basic) ≈ 2.0
@test abs_sum_phase_calibrated(v, :grid) ≈ 2.8284271
@test abs_sum_optimal(v)[1] ≈ 2.8284271

θ_opt = QuantumOptimalControl.optimal_calibration(v)[2]
@test f(v, θ_opt) ≈ abs_sum_phase_calibrated(v) atol=1e-12


v =  [1, 0.1im, 0.1im, 1]

@test abs_sum_phase_calibrated(v) ≈ 2.0099751
@test abs_sum_phase_calibrated(v, :basic) ≈ 0.2
@test abs_sum_phase_calibrated(v, :grid) ≈ 2.0099751
@test abs_sum_optimal(v)[1] ≈ 2.0099751

θ_opt = QuantumOptimalControl.optimal_calibration(v)[2]
@test f(v, θ_opt) ≈ abs_sum_phase_calibrated(v) atol=1e-12


v =  [cis(1), cis(2), cis(3), cis(4)]
@test abs_sum_phase_calibrated(v) ≈ 4
@test abs_sum_phase_calibrated(v, :basic) ≈ 4
@test abs_sum_phase_calibrated(v, :grid) ≈ 4 atol=1e-3
@test abs_sum_optimal(v)[1] ≈ 4

θ_opt = QuantumOptimalControl.optimal_calibration(v)[2]
@test f(v, θ_opt) ≈ abs_sum_phase_calibrated(v) atol=1e-12

##

v =  cis.([1, 2, -2.5, -1.7])

@test abs_sum_phase_calibrated(v) ≈ 3.995001
@test abs_sum_phase_calibrated(v, :basic) ≈ 3.995001 atol=0.01
@test abs_sum_phase_calibrated(v, :grid) ≈ 3.995001 atol=1e-3
@test abs_sum_optimal(v)[1] ≈ 3.995001

θ_opt = QuantumOptimalControl.optimal_calibration(v)[2]

@test f(v, θ_opt) ≈ abs_sum_phase_calibrated(v) atol=1e-8
@test QuantumOptimalControl.optimal_calibration(v)[2] ≈ [3.6000220820575084, 5.383258515112539] atol=1e-4


##

v =  cis.([2.5, 2.5, 1.5, -2.5])


f_opt, θ_opt = QuantumOptimalControl.optimal_calibration(v)
@test f_opt ≈ 3.365883939061934
@test f(v, θ_opt) ≈ 3.365883939061934
@test abs_sum_phase_calibrated(v) ≈ 3.365883939061934

##

v =  [0.65 - 0.75im, -0.4 + 0.8im, -0.4 + 0.1im, 0.7 - 0.0im]

@test abs_sum_phase_calibrated(v) ≈ 2.9787244710195484

f_opt, θ_opt = QuantumOptimalControl.optimal_calibration(v, 1e-15)
@test f_opt ≈ 2.9787244710195484
@test f(v, θ_opt) ≈ 2.9787244710195484


##

v =  [im, 1, -im, 1]
abs_sum_phase_calibrated(v) ≈ abs_sum_optimal(v)[1]

v =  [im, 1, im, -1] # Problematic case
@test abs_sum_phase_calibrated(v, :circular_mean) ≈ abs_sum_optimal(v)[1]

v =  [im, 0.1, 0.1*im, -1]
@test abs_sum_phase_calibrated(v, :circular_mean) ≈ abs_sum_optimal(v)[1]

v =  [cis(2), 1, cis(-2), 1]
abs_sum_phase_calibrated(v, :circular_mean) ≈ abs_sum_optimal(v)[1]



#
# Test that the opitmal version is not that much worse than the gridded veraasion
Random.seed!(0)
v_data = [rand(4) .* cis.(2π*rand(4)) for k=1:500]

F_optimal_test = [1 - abs_sum_optimal(v[1]/4 for v in v_data]
F_optimal = [1 - abs_sum_phase_calibrated(v, :optimal)/4 for v in v_data]
F_grid = [1 - abs_sum_phase_calibrated(v, :grid)/4 for v in v_data]

F_approx = [1 - QuantumOptimalControl.optimal_calibration(v, approximate_optimum=true)[1]/4 for m in v_data]

@test all(F_grid - F_optimal .> -eps()) # :optimal should at least be better than :grid
@test all(F_grid - F_optimal .< 1e-3) # :grid should not be that much worse than :optimal

@test all(F_optimal_test - F_optimal  .< 1e-6)


println(vean(F_optimal_test - F_optimal))
@test all(F_optimal_test - F_optimal .>= 0)


## Test gradient computation
using FiniteDifferences
Jfun = v -> abs_sum_phase_calibrated(v, :optimal)^2

Random.seed!(100)

m_vec = [rand(4) .* cis.(2π*rand(4)) for k=1:2000]

for (k,m) in enumerate(m_vec)

    #println(k)
    f_opt, θ_opt = QuantumOptimalControl.optimal_calibration(v, 1e-12)
    @test f(v, θ_opt) ≈ f_opt atol=1e-14

    grad_fd = FiniteDifferences.grad(central_fdm(9,1,max_range=0.001), Jfun, v)[1]
    grad_analytic = QuantumOptimalControl.abs_sum_phase_calibrated_grad(v, θ_opt[1])

    #w1 = g_derivatives(v[1:2], θ_opt[1])
    #w2 = g_derivatives(v[3:4], θ_opt[1])
    #println(w1[1] + w2[1]) # Check that derivative wrt θ is zero

    @test grad_fd ≈ grad_analytic rtol=1e-6
end



using ChainRulesTestUtils

v =  [0.65 - 0.75im, -0.4 + 0.8im, -0.4 + 0.1im, 0.7 - 0.0im]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, reltol=)

v =  [im, 1, -im, 1]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, reltol=1e-4)

v =  [im, 1, im, -1]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, rtol=1e-7)

v =  [im, 0.1, 0.1*im, -1]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, rtol=1e-7)

v =  [cis(2), 1, cis(-2), 1]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m, rtol=1e-7)


v =  [0.65 - 0.75im, -0.4 + 0.8im, -0.4 + 0.1im, 0.7 - 0.0im]
ChainRulesTestUtils.test_rrule(abs_sum_phase_calibrated, m)
