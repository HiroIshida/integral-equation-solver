
struct Grid2d
    N::Vector{Int64}
    dx::Vector{Float64}
    b_min::Vector{Float64}
    b_max::Vector{Float64}
    function Grid2d(N, b_min, b_max)
        dx = (b_max - b_min)./N
        new(N, dx, b_min, b_max)
    end
end

function collect_pts(gr::Grid2d, pts::Vector{Vector{Float64}})
    data = zeros(Int, gr.N[1], gr.N[2])
    for pt in pts
        idxs = _whereami(gr, pt)
        data[idxs[1], idxs[2]] += 1
    end
    return data
end

function _whereami(gr::Grid2d, pt::Vector{Float64})
    n = floor.(Int, (pt - gr.b_min)./gr.dx) .+ 1
    return n
end

