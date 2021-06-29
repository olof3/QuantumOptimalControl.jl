using Optim, QuantumOptimalControl, Test

function abs_trace_optimal(M)
    m = diag(M)
    J = θ -> abs(sum(m[1] + m[2]*cis(θ[1]) + m[3]*cis(θ[2]) + m[4]*cis(θ[1] + θ[2])))

    θ_grid = Iterators.product(LinRange(0,2π,100), LinRange(0,2π,100))
    θbest_grid = collect(argmax(J, θ_grid))

    res = optimize(θ -> -J(θ), θbest_grid, x_abstol=1e-15)

    -res.minimum
end


M = Diagonal([1, 1im, 1im, 1])

@test abs_trace_phase_calibrated(M) ≈ 2.8284271
@test abs_trace_phase_calibrated(M, :basic) ≈ 2.0
@test abs_trace_phase_calibrated(M, :grid) ≈ 2.8284271
@test abs_trace_optimal(M) ≈ 2.8284271
@test abs_trace_phase_calibrated(M, :lms_phase_semiold) ≈ 2.8284271



M = Diagonal([1, 0.1im, 0.1im, 1])

@test abs_trace_phase_calibrated(M) ≈ 2.0099751
@test abs_trace_phase_calibrated(M, :basic) ≈ 0.2 
@test abs_trace_phase_calibrated(M, :grid) ≈ 2.0099751
@test abs_trace_optimal(M) ≈ 2.0099751


M = Diagonal([cis(1), cis(2), cis(3), cis(4)])
@test abs_trace_phase_calibrated(M) ≈ 4
@test abs_trace_phase_calibrated(M, :basic) ≈ 4
@test abs_trace_phase_calibrated(M, :grid) ≈ 4 atol=1e-3
@test abs_trace_optimal(M) ≈ 4
abs_trace_phase_calibrated(M, :lms_phase_semiold)

##

M = Diagonal(cis.([1, 2, -2.5, -1.7]))

@test abs_trace_phase_calibrated(M) ≈ 3.995001
@test abs_trace_phase_calibrated(M, :basic) ≈ 3.995001 atol=0.01
@test abs_trace_phase_calibrated(M, :grid) ≈ 3.995001 atol=1e-3
@test abs_trace_optimal(M) ≈ 3.995001

##

M = Diagonal(cis.([2.5, 2.5, 1.5, -2.5]))

abs_trace_phase_calibrated(M) ≈ 3.365883939061934  
abs_trace_optimal(M) ≈ 3.365883939061934

##


M = Diagonal([im, 1, -im, 1])
abs_trace_phase_calibrated(M) ≈ abs_trace_optimal(M)


M = Diagonal([im, 1, im, -1]) # Problematic case
@test abs_trace_phase_calibrated(M, :lms_phase) ≈ abs_trace_optimal(M)
#abs_trace_phase_calibrated(M, :lms_phase_semiold) # fails

M = Diagonal([im, 0.1, 0.1*im, -1])
@test abs_trace_phase_calibrated(M, :lms_phase) ≈ abs_trace_optimal(M)

M = Diagonal([cis(2), 1, cis(-2), 1])
abs_trace_phase_calibrated(M, :lms_phase) ≈ abs_trace_optimal(M)

