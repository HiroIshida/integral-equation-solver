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
    N_trial = 10000

    println("start mc")
    x_grid_lst = make_grid_pts(Nx, x_min, x_max)
    pt_result_lst = Vector{Float64}[]
    y_min = Inf; y_max = -Inf
    for x_grid in x_grid_lst
        println(x_grid)
        for _ in 1:N_trial
            y = func_mc(x_grid)
            if y != Inf
                y < y_min && (y_min = y - eps)
                y > y_max && (y_max = y + eps)
                push!(pt_result_lst, [x_grid, y])
            end
        end
    end
    y_min -= 20
    y_max += 20
    mesh = Grid2d([Nx, Ny], [x_min, y_min], [x_max, y_max])
    data = collect_pts(mesh, pt_result_lst)
    println("finish mc")

    # M_{i, j] = p(y_j|x_i) 
    data_normalized = zeros(Nx, Ny)
    dx = (x_max - x_min)/Nx
    dy = (y_max - y_min)/Ny
    for i in 1:Nx
        data_normalized[i, :] = data[i, :]/sum(data[i, :])/dy
    end

    # then we can formulate integral-equation solving as 
    # least squares optimization with equality constraints
    A = data_normalized'
    y_grid_lst = make_grid_pts(Ny, y_min, y_max)
    b = [pdf_target(y) for y in y_grid_lst]
    C = ones(Nx)
    w = Variable(Nx)
    λ = 0.005
    objective = sumsquares(A*w*dx-b) + λ*sumsquares(w)
    constraints = [w>0; sum(w*dx) == 1]
    problem = minimize(objective, constraints)
    println("start solving")
    solve!(problem, SCSSolver())
    w_opt = w.value
    return x_grid_lst, y_grid_lst, w_opt, A
end

function test()

    npdf(x) = x + randn()
    func_mc = npdf
    x_min = -150
    x_max = 150
    sigma = 10
    pdf_t(x) = 1/sqrt(2*pi*sigma^2)*exp(-(x-0)^2/(2*sigma^2))
    Nx = 300
    Ny = 300
    x_lst, y_lst, w_opt, A = solve_integral_equation(func_mc, pdf_t, x_min, x_max, Nx, Ny)
    PyPlot.plot(y_lst, [pdf_t(y) for y in y_lst])
    PyPlot.plot(y_lst, A*w_opt)
end
#mat = test()
