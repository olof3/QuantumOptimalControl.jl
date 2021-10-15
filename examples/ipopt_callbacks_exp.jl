function setup_ipopt_callbacks(A0Δt, A1Δt, A2Δt, x0, u_prototype, (Jfinal,dJfinal_dx), (L,dL_dx), B)
    nu, nsegments = size(u_prototype)
    ng = 2
    nx = length(x0)
    nsplines = size(B,2)
    nc = nu * nsplines # The number of spline coefficients, i.e., optimization parameters

    c_prev = Vector{Float64}(undef, nc) # To check if the f has been evaluated for the provided c
    cache = QuantumOptimalControl.setup_grape_cache(A0Δt, complex(x0), (2, nsegments))

    f = function(c::Vector{Float64})
        c_prev .= c
        c = reshape(c, nsplines, nu)
        u = transpose(B*c)

        x = QuantumOptimalControl.propagate(A0Δt, [A1Δt, A2Δt], u, x0, cache)

        Jfinal(x[end]) + sum(L, x)
    end

    f_grad = function(c, f_grad_out)
        if c_prev != c
            println("gradient not computed") # Note sure if IPOPT will ever ask for the gradient before the fcn value
            f(c) # Use the side effects in f that puts the necessary stuff into cache
        end
        
        dJdu = QuantumOptimalControl.grape_sensitivity(A0Δt, [A1Δt, A2Δt], dJfinal_dx, cache.u, x0, cache; dUkdp_order=3, dL_dx=dL_dx) # use cache.u
        dJdc = B'*transpose(dJdu)

        f_grad_out .= dJdc[:]
    end
    
    g_oop = function(c)
        c = reshape(c, nsplines, nu)
        [norm(c);
        norm(diff(c, dims=1))]        
    end

    g = function(c, g_out)
        g_out .= g_oop(c)
    end

    function g_jac(c, mode, rows, cols, g_jac_out)    
        if mode == :Structure
            cols .= kron(ones(ng), 1:nc)
            rows .= kron(1:ng, ones(nc))    
        else
            g_jac_tmp = Zygote.jacobian(g_oop, c)[1]::Matrix{Float64}
            g_jac_out .= transpose(g_jac_tmp)[:]
        end        
    end

    f, g, f_grad, g_jac, nu, ng, nx, nc, cache
end


## Could use the code in test_ippot_fcns.jl to make sure that gradients/jacobians work