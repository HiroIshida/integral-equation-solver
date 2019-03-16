include("Grid2d.jl")
include("utils.jl")

struct MonteCarlo
    func
    x_min
    x_max
end

function monte_carlo_inversion(mc::MonteCarlo)
    # func_mc : takes x as an argument and return random y following P(y|x) 

    eps = 1e-3
    N_trial = 10000
    Nx = 20
    Ny = 20

    x_grid_lst = make_grid_pts(Nx, mc.x_min, mc.x_max)
    pt_result_lst = Vector{Float64}[]
    y_min = Inf; y_max = -Inf
    for x_grid in x_grid_lst
        for _ in 1:N_trial
            y = mc.func(x_grid)
            y < y_min && (y_min = y - eps)
            y > y_max && (y_max = y + eps)
            push!(pt_result_lst, [x_grid, y])
        end
    end

    mesh = Grid2d([Nx, Ny], [mc.x_min, y_min], [mc.x_max, y_max])
    data = collect_pts(mesh, pt_result_lst)
end

function test()
    npdf(x) = x + randn()
    mc = MonteCarlo(npdf, -3, 3)
    monte_carlo_inversion(mc)
end
test()
