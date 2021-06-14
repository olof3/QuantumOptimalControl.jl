using Optim, QuantumOptimalControl, Test

function abs_trace_optimal(M)
    m = diag(M)
    J = θ -> abs(sum(m[1] + m[2]*cis(θ[1]) + m[3]*cis(θ[2]) + m[4]*cis(θ[1] + θ[2])))

    θ_grid = Iterators.product(LinRange(0,2π,100), LinRange(0,2π,100))
    θbest_grid = collect(argmax(J, θ_grid))

    res = optimize(θ -> -J(θ), θbest_grid, x_abstol=1e-12)

    -res.minimum
end


M = Diagonal([1, 1im, 1im, 1])

@test abs_trace_phase_calibrated(M) ≈ 2.8284271
@test abs_trace_phase_calibrated(M, :basic) ≈ 2.0
@test abs_trace_phase_calibrated(M, :grid) ≈ 2.8284271
@test abs_trace_optimal(M) ≈ 2.8284271
abs_trace_phase_calibrated(M, :lms_phase2)


M = Diagonal([1, 0.1im, 0.1im, 1])

@test abs_trace_phase_calibrated(M) ≈ 2.0099751
@test abs_trace_phase_calibrated(M, :basic) ≈ 0.2 
@test abs_trace_phase_calibrated(M, :grid) ≈ 2.0099751
@test abs_trace_optimal(M) ≈ 2.0099751


M = Diagonal([cis(1), cis(2), cis(3), cis(4)])
@test abs_trace_phase_calibrated(M) == 4
@test abs_trace_phase_calibrated(M, :basic) ≈ 4
@test abs_trace_phase_calibrated(M, :grid) ≈ 4 atol=1e-3
@test abs_trace_optimal(M) ≈ 4
abs_trace_phase_calibrated(M, :lms_phase2)

M = Diagonal([1, cis(-1.5), cis(2), cis(3)])
@test abs_trace_phase_calibrated(M) ≈ 3.24385247
@test abs_trace_optimal(M) ≈ 3.24385247

##

M = Diagonal(cis.([1, 2, -2.5, -1.7]))

@test abs_trace_phase_calibrated(M) ≈ 3.995001
@test abs_trace_phase_calibrated(M, :basic) ≈ 3.995001 atol=0.01
@test abs_trace_phase_calibrated(M, :grid) ≈ 3.995001 atol=1e-3
@test abs_trace_optimal(M) ≈ 3.995001
abs_trace_phase_calibrated(M, :lms_phase2)

##

M = Diagonal(cis.([2.5, 2.5, 1.5, -2.5]))

abs_trace_phase_calibrated(M)
abs_trace_phase_calibrated(M, :basic)
abs_trace_phase_calibrated(M, :grid)
abs_trace_optimal(M)
abs_trace_phase_calibrated(M, :lms_phase2)

##

M = Diagonal([1, im, -im, 1])
@test abs_trace_phase_calibrated(M, :lms_phase2) ≈ 4



M = Diagonal([im, 1, -im, 1]) # Problematic case
abs_trace_phase_calibrated(M, :lms_phase)
abs_trace_phase_calibrated(M, :lms_phase2)
abs_trace_phase_calibrated(M, :basic)
abs_trace_optimal(M)



M = Diagonal([cis(2), 1, cis(-2), 1])
abs_trace_phase_calibrated(M, :lms_phase)
abs_trace_phase_calibrated(M, :lms_phase2)
abs_trace_phase_calibrated(M, :basic)
abs_trace_optimal(M)


M = Diagonal(im*[0.7*cis(2), 0.5, 0.5*cis(-2), 0.7])
abs_trace_phase_calibrated(M, :lms_phase)
abs_trace_phase_calibrated(M, :lms_phase2)
abs_trace_phase_calibrated(M, :basic)
abs_trace_optimal(M)



#Zygote.gradient(m -> abs_trace_phase_calibrated(Diagonal(m), :lms_phase), [im, 1, -im, 1])

QuantumOptimalControl.lms_phase_calibration(diag(M))
QuantumOptimalControl.lms_phase_calibration2(diag(M))

m = diag(M)


θ = [angle(conj(m[1])*m[2] + conj(m[3])*m[4]),
     angle(conj(m[1])*m[3] + conj(m[2])*m[4])]

##
QuantumOptimalControl.lms_phase_calibration2(diag(M))
QuantumOptimalControl.grid_calibration(diag(M))

##
perf_optimal = Float64[]
perf_basic = Float64[]
perf_lms = Float64[]
perf_lms2 = Float64[]
for k=1:5000
    M = Diagonal((1 .- 0.2rand(4)) .* cis.(2π * rand(4)))
    push!(perf_optimal, 1 - abs_trace_optimal(M)/4)
    push!(perf_lms, 1 - abs_trace_phase_calibrated(M, :lms_phase) / 4)
    #push!(perf_lms2, 1 - abs_trace_phase_calibrated(M, :lms_phase2) / 4)
    push!(perf_basic, 1 - abs_trace_phase_calibrated(M, :basic)/4)
end


v = sortperm(perf_optimal, rev=true)

#plot([perf_optimal[v] perf_lms[v]], yscale=:log10)

#plot([perf_optimal[v] perf_lms[v] perf_basic[v]], yscale=:log10)

plot([perf_optimal[v] perf_lms[v] perf_basic[v]] ./ (perf_optimal[v] * ones(3)'))