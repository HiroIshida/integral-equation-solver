function make_grid_pts(N_grid, b_min, b_max)
    ndim = length(b_min)

    if ndim == 1
        make_grid_pts_1(N_grid, b_min, b_max)
    elseif ndim == 2
        make_grid_pts_2(N_grid, b_min, b_max)
    else
        #TODO metaprogramming and such
        error("under construction")
    end
end

function make_grid_pts_1(N_grid, b_min, b_max)
    dx = (b_max - b_min)/N_grid
    pts = Float64[]
    for i in 1:N_grid
        push!(pts, b_min + (i - 0.5) * dx)
    end
    return pts
end

function make_grid_pts_2(N_grid, b_min, b_max)
    dx = (b_max - b_min)/N_grid
    pts = Vector{Float64}[]
    for i in 1:N_grid, j in 1:N_grid
        pt = b_min + [i - 0.5, j - 0.5].*dx
        push!(pts, pt)
    end
    return pts
end


