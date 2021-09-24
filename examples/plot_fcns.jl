using Plots

function plot_2qubit_evolution(qb, t, x, u=nothing; to_states=["00", "01", "10", "11"], max_rabi_rate=nothing)
    # reshape the trajectory for easier plotting
    x_reshaped = [[x[k][ij_ind] for k=1:length(x)] for ij_ind = CartesianIndices(x[1])]

    css_labels = ["00", "01", "10", "11"]
    css_dict = Dict(css_labels .=> 1:4)    

    to_state_inds = [qb.state_dict[s] for s in to_states]
    
    plts = []
    for i=0:1, j=0:1

        from_state_idx = css_dict[string(i, j)]
    
        plt = plot(legend=(i==j==0), title="From state |$(css_labels[from_state_idx])⟩")
        for to_state_idx in to_state_inds
            plot!(plt, t, abs2.(x_reshaped[to_state_idx,from_state_idx]), label=qb.state_labels[to_state_idx])
        end
        push!(plts, plt)
    end
    

    if !isnothing(u)
        plt_u = plot(t[1:end-1], transpose(u), linetype=:steppost, title="Control signal", label=["I" "Q"])
        
        if max_rabi_rate !== nothing
            plot!(plt_u, [t[1], t[end]], max_rabi_rate*[-1 1; -1 1], c="black", label=nothing)
        end
        
        layout = @layout [a b; c d; e]
        plot(plts..., plt_u, layout=layout)
    else
        layout = @layout [a b; c d]
        plot(plts..., layout=layout)
    end
end