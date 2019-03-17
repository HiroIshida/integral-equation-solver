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

function solve_integral_equation(func_mc, pdf_target, x_min, x_max, Nx, Ny)

    eps = 1e-3
    N_trial = 30000

    println("start mc")
    x_grid_lst = make_grid_pts(Nx, x_min, x_max)
    pt_result_lst = Vector{Float64}[]
    y_min = Inf; y_max = -Inf
    for x_grid in x_grid_lst
        for _ in 1:N_trial
            y = func_mc(x_grid)
            y < y_min && (y_min = y - eps)
            y > y_max && (y_max = y + eps)
            push!(pt_result_lst, [x_grid, y])
        end
    end
    mesh = Grid2d([Nx, Ny], [x_min, y_min], [x_max, y_max])
    data = collect_pts(mesh, pt_result_lst)
    println("finish mc")

    data_normalized = zeros(Nx, Ny)
    dx = (x_max - x_min)/Nx
    for i in 1:Nx
        data_normalized[i, :] = data[i, :]/sum(data[i, :])/dx
    end

    # then we can formulate integral-equation solving as 
    # least squares optimization with equality constraints
    A = data_normalized'*dx
    b = [pdf_target(y) for y in make_grid_pts(Ny, y_min, y_max)]
    C = ones(Nx)
    w = Variable(Nx)
    λ = 0.001
    objective = sumsquares(A*w-b) + λ*sumsquares(w)
    constraints = [w>0; sum(w*dx) == 1]
    problem = minimize(objective, constraints)
    solve!(problem, SCSSolver())
    w_opt = w.value
    #=
    plot(x_grid_lst, [normpdf(x, 0, 2*sqrt(2)) for x in x_grid_lst])
    plot(x_grid_lst, w_opt)
    xlim(-15, 15)
    =#
    return x_grid_lst, w_opt
end

function test()
    npdf(x) = x + randn()
    func_mc = npdf
    x_min = -150
    x_max = 150
    sigma = 3
    pdf_t(x) = 1/sqrt(2*pi*sigma^2)*exp(-(x-0)^2/(2*sigma^2))
    Nx = 1000
    Ny = 1000
    x_lst, w_lst = solve(func_mc, pdf_t, x_min, x_max, Nx, Ny)
end
