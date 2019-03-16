include("Grid2d.jl")
include("utils.jl")

using LinearAlgebra

eye(T::Type, m, n) = Matrix{T}(I, m, n)
eye(m, n) = eye(Float64, m, n)
eye(m) = eye(m, m)

struct MonteCarlo
    func
    x_min
    x_max
end

function build_matrix(mc::MonteCarlo, pdf_target, Nx, Ny)
    # func_mc : takes x as an argument and return random y following P(y|x) 

    eps = 1e-3
    N_trial = 10000

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
    data = transpose(collect_pts(mesh, pt_result_lst))
    data_normalized = zeros(Nx, Ny)
    for i in 1:Nx
        data_normalized[i, :] = data[i, :]/sum(data[i, :])
    end

    # see https://www.cs.ubc.ca/~schmidtm/Documents/2005_Notes_Lasso.pdf
    # note that M doesn't have to be square matrix
    λ = 10
    y_vec = [pdf_target(y) for y in make_grid_pts(Ny, y_min, y_max)]
    X = data_normalized'
    w_vec = inv(X'*X + λ*eye(size(X, 2)))*X'*y_vec
    return w_vec
end

function test()
    npdf(x) = x + randn()
    mc = MonteCarlo(npdf, -4, 4)

    sigma = 3
    pdf_t(x) = 1/sqrt(2*pi*sigma^2)*exp(-(x-0)^2/(2*sigma^2))

    Nx = 20
    Ny = 20

    w_vec = build_matrix(mc, pdf_t, Nx, Ny)
end
w = test()
