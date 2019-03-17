include("Grid2d.jl")
include("utils.jl")
using Convex, SCS
using LinearAlgebra
using PyPlot

eye(T::Type, m, n) = Matrix{T}(I, m, n)
eye(m, n) = eye(Float64, m, n)
eye(m) = eye(m, m)

@inline function normpdf(x, mu, sigma)
    p = 1/sqrt(2*pi*sigma^2)*exp(-(x-mu)^2/(2*sigma^2))
    return p
end

struct MonteCarlo
    func
    x_min
    x_max
end

function build_matrix(mc::MonteCarlo, pdf_target, Nx, Ny)
    # func_mc : takes x as an argument and return random y following P(y|x) 

    eps = 1e-3
    N_trial = 20000

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
    data_normalized = zeros(Nx, Ny)
    dx = (mc.x_max - mc.x_min)/Nx
    for i in 1:Nx
        data_normalized[i, :] = data[i, :]/sum(data[i, :])/dx
    end
    #return data_normalized

    # then we can formulate integral-equation solving as 
    # least squares optimization with equality constraints
    A = data_normalized'*dx
    b = [pdf_target(y) for y in make_grid_pts(Ny, y_min, y_max)]
    C = ones(Nx)
    w = Variable(Nx)
    objective = sumsquares(A*w-b)
    #constraints = [C'*w-1 == 0; w>0]
    constraints = [w>0; sum(w*dx) == 1]
    problem = minimize(objective, constraints)
    solve!(problem, SCSSolver())
    w_opt = w.value
    #w_opt = [normpdf(x, 0, 2*sqrt(2)) for x in x_grid_lst]
    plot(make_grid_pts(Ny, y_min, y_max), b)
    plot(x_grid_lst, w_opt)
end

function test()
    npdf(x) = x + randn()
    mc = MonteCarlo(npdf, -10, 10)
    sigma = 3
    pdf_t(x) = 1/sqrt(2*pi*sigma^2)*exp(-(x-0)^2/(2*sigma^2))
    Nx = 50
    Ny = 50
    mat = build_matrix(mc, pdf_t, Nx, Ny)
end
mat = test()
