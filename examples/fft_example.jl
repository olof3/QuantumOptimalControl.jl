using AbstractFFTs, FFTW

# Simple example of getting the FFT of a control signal

u_cplx = u[1,:] + im*u[2,:]

Ω = fftfreq(length(u_cplx), 1/Δt)
U = fft(u_cplx)

Ω = -20:0.01:20
fft_mat = [exp(-im*ω*t) for ω in Ω, t in t[1:end-1]]
U = fft_mat * u_cplx

plot(Ω[Ω .> 0]/2π, abs.(U[Ω .> 0]), xscale=:log, yscale=:log, label="ω > 0");
plot!(-Ω[Ω .< 0]/2π, abs.(U[Ω .< 0]), xscale=:log, yscale=:log, c=:red, label="ω < 0")

