@kwdef mutable struct ROMISkeletonParams
    # volume threshold
    t::Float64 = 0.5 # p(∈ plant) > p(∉ plant)

    # center line conductance
    β::Float64 = 2.0 # shape of the conductance channel
    νmin::Float64 = 0.01 # surface speed relative to center line

    # prominence
    h::Float64 = 15.0

    # Nadaraya-Watson curve smoothing scale
    k_trunc::Int = 6 # kernel truncation
    σstem::Float64 = 5.0
    σbranch::Float64 = 1.5

    # curve reparametrization dense sampling factor
    n_min::Int = 100
    dt_factor::Real = 0.25
end
function Base.copyto!(dest::ROMISkeletonParams, src::ROMISkeletonParams)
    dest.t = src.t
    dest.β = src.β
    dest.h = src.h
    dest.k_trunc = src.k_trunc
    dest.σstem = src.σstem
    dest.σbranch = src.σbranch
    dest.n_min = src.n_min
    dest.dt_factor = src.dt_factor
    return dest
end
function Base.copy(p::ROMISkeletonParams)
    pnew = ROMISkeletonParams()
    copyto!(pnew, p)
end

# add the missing coordinate of a CartesianIndex from a slice
function add_missing_coordinate(ix::CartesianIndex{2}, axis::Int, at::Int)
    return (axis == 1 ? CartesianIndex(at, ix.I...) : (axis == 2 ? CartesianIndex(ix.I[1], at, ix.I[2]) : CartesianIndex(ix.I..., at)))
end

"""
    detect_truncation_faces(vb::AbstractArray{Bool}, axis::Int, at::Int; min_voxels::Int = 3)

Flags a bounding-box face as a truncation face if more than `min_voxels` occupied voxels touch it.
A single isolated voxel touching the box edge is just an ordinary boundary point,
a cluster indicates a real structure (stem or fruit) was sliced by the box.
"""
function detect_truncation_faces(vb::AbstractArray{Bool}, axis::Int, at::Int; min_voxels::Int = 3)
    slice = selectdim(vb, axis, at)
    labels = label_components(slice)
    counts = component_lengths(labels)
    coords = CartesianIndex{3}[]
    for (i, indices) in Iterators.enumerate(Iterators.drop(component_indices(CartesianIndex, labels), 1))
        (counts[i] < min_voxels) && continue
        for ix in indices
            # add the missing coordinate
            c = add_missing_coordinate(ix, axis, at)
            push!(coords, c)
        end
    end
    return coords
end

struct ROMIBinaryVolume <: AbstractArray{Bool, 3}
    # volume and mapping
    vb::BitArray{3}
    idmap::Array{Int, 3} # 0 = not a node, else 1-based id
    coords::Vector{CartesianIndex{3}} # node id → voxel index

    # voxel coordinates grid
    voxel_size::Float64
    vox_grid::ROMIVoxelGrid{Float64}

    # euclidean distance transform
    dt::Array{Float64, 3}

    # local radius
    rlocal::Vector{Float64}

    # node conductance
    γ::Vector{Float64}

    # root node
    root::CartesianIndex{3}
    root_id::Int

    # boundary voxels
    bv::Vector{CartesianIndex{3}}

    # truncation faces
    tf::Vector{CartesianIndex{3}}

    # number of nodes and edges
    nv::Int
    ne::Int

    # graph adjacency representation for Dijkstra
    adj_ptr::Vector{Int}
    adj_nbr::Vector{Int}
    adj_w::Vector{Float64}
    adj_d::Vector{Float64}

    function ROMIBinaryVolume(vb::AbstractArray{Bool, 3}, β::Real, νmin::Real;
            bbox_origin::Point3 = zero(Point3d), voxel_size::Real = 1.0, root::Union{Nothing, Int} = nothing)
        @assert any(vb) "Thresholded volume is empty!"
        vox_grid = ROMIVoxelGrid(bbox_origin, size(vb), voxel_size)
        keep_largest_component!(vb) # for safety
        idmap = zeros(Int, size(vb))
        coords = CartesianIndex{3}[]
        sizehint!(coords, sum(vb))
        for ix in CartesianIndices(vb)
            vb[ix] || continue
            push!(coords, ix)
            idmap[ix] = length(coords)
        end
        nv = length(coords) # number of nodes
        dt = distance_transform(feature_transform(.!vb))

        # identify truncation faces (use to filter local maxima)
        tf = detect_truncation_faces(vb, 3, 1) # bottom face (z: bottom → top)
        append!(tf, detect_truncation_faces(vb, 3, size(vb, 3))) # top face
        append!(tf, detect_truncation_faces(vb, 2, 1)) # front face (y: front → back)
        append!(tf, detect_truncation_faces(vb, 2, size(vb, 2))) # back face
        append!(tf, detect_truncation_faces(vb, 1, 1)) # left face (x: left → right)
        append!(tf, detect_truncation_faces(vb, 1, size(vb, 1))) # rigth face

        # compute volume boundary (padding ⟹ truncated faces interior included)
        #vb_pad = padarray(vb, Fill(false, (1, 1, 1)))
        #@views vb_boundary = vb .& .!erode(vb_pad, strel_diamond((3, 3, 3)))[CartesianIndices(vb)] # 6-connectivity
        vb_boundary = vb .& .!erode(vb, strel_diamond((3, 3, 3))) # 6-connectivity
        boundary_voxel = findall(vb_boundary)

        # get root node
        if root === nothing
            btm_ix = findfirst(x -> any(x .> 0), eachslice(dt; dims = 3))
            root_vox = argmax(view(dt, :, :, btm_ix))
            root_vox = CartesianIndex(root_vox.I..., btm_ix)
            root_id = idmap[root_vox]
        else
            root_vox = coords[root]
            root_id = root
        end
        
        # identify Centers of Maximal Balls
        is_medial_axis = falses(nv)
        for v in 1:nv
            I = coords[v]
            dt_v = dt[I]
            is_cmb = true
            
            for (off, dist) in ((CartesianIndex(i, j, k), sqrt(i ^ 2 + j ^ 2 + k ^ 2)) for i in -1:1, j in -1:1, k in -1:1 if (i, j, k) != (0, 0, 0))
                J = I + off
                checkbounds(Bool, vb, J) || continue
                if vb[J] && ((dt[J] - dt_v) ≥ dist)
                    is_cmb = false
                    break
                end
            end
            is_medial_axis[v] = is_cmb
        end
        
        # compute local radius for every voxel using Maximal Ball Deposition
        rlocal = fill(-1.0, nv)

        # deposit maximal spheres from medial axis nodes into rlocal
        medial_nodes = findall(is_medial_axis)
        for c in medial_nodes
            I_c = coords[c]
            r_c = dt[I_c]
            
            # add 0.5 voxel tolerance to ensure spheres reach boundary voxels
            r_padded = r_c + 0.5
            r_int = ceil(Int, r_padded)
            r2_max = r_padded ^ 2
            
            # rasterize sphere of radius r_c around I_c
            for dx in -r_int:r_int, dy in -r_int:r_int, dz in -r_int:r_int
                if ((dx ^ 2) + (dy ^ 2) + (dz ^ 2)) ≤ r2_max
                    J = I_c + CartesianIndex(dx, dy, dz)
                    checkbounds(Bool, vb, J) || continue
                    if vb[J]
                        w = idmap[J]
                        if w > 0
                            rlocal[w] = max(rlocal[w], r_c)
                        end
                    end
                end
            end
        end

        # compute node conductance (higher close to the center line)
        γ = Vector{Float64}(undef, nv)
        for v in 1:nv
            I = coords[v]

            # relative position from the center line (0: boundary → 1: center)
            r = (rlocal[v] == 1.0 ? 1.0 : (dt[I] - 1) / (rlocal[v] - 1))

            # node conductance νmin at the boundary and 1 in the center
            γ[v] = νmin + (1 - νmin) * (r ^ β)
        end

        # build adjacency topology
        ev, em, edist = Int[], Int[], Float64[]
        for v in 1:nv
            I = coords[v]
            for (off, dist) in ((CartesianIndex(i, j, k), sqrt(i ^ 2 + j ^ 2 + k ^ 2))
                                for i in -1:1, j in -1:1, k in -1:1 if (i, j, k) != (0, 0, 0))
                J = I + off
                checkbounds(Bool, vb, J) || continue
                vb[J] || continue
                m = idmap[J]
                m < v && continue
                push!(ev, v)
                push!(em, m)
                push!(edist, dist)
            end
        end
        ne = length(ev) # number of edge
        deg = zeros(Int, nv)
        for k in 1:ne
            deg[ev[k]] += 1
            deg[em[k]] += 1
        end
        adj_ptr = Vector{Int}(undef, nv + 1)
        adj_ptr[1] = 1
        for v in 1:nv
            adj_ptr[v + 1] = adj_ptr[v] + deg[v]
        end
        total_edges = adj_ptr[nv + 1] - 1
        adj_nbr = Vector{Int}(undef, total_edges)
        adj_w = Vector{Float64}(undef, total_edges)
        adj_d = Vector{Float64}(undef, total_edges)
        
        # cost to approximate flux travel time with high conductance near the center
        cursor = copy(adj_ptr)
        for k in 1:ne
            v, m, d = ev[k], em[k], edist[k]
            
            # travel time
            w = d / (0.5 * (γ[v] + γ[m]))

            # write into adjacency vectors
            adj_nbr[cursor[v]] = m
            adj_w[cursor[v]] = w
            adj_d[cursor[v]] = d
            cursor[v] += 1
            adj_nbr[cursor[m]] = v
            adj_w[cursor[m]] = w
            adj_d[cursor[m]] = d
            cursor[m] += 1
        end

        return new(vb, idmap, coords, voxel_size, vox_grid, dt, rlocal, γ, root_vox, root_id, boundary_voxel, tf, nv, ne, adj_ptr, adj_nbr, adj_w, adj_d)
    end
end
Base.size(vb::ROMIBinaryVolume) = size(vb.vb)
function Base.getindex(vb::ROMIBinaryVolume, I...; dt::Bool = false)
    dt && return vb.dt[I...] # return distance transform
    return vb.vb[I...]
end
function Base.Array(vb::ROMIBinaryVolume; dt::Bool = false)
    dt && return vb.dt # return distance transform
    return vb.vb
end

# constructor
function ROMIBinaryVolume(vol::ROMIVolume, t::Real, β::Real, νmin::Real; root::Union{Nothing, Int} = nothing)
    return ROMIBinaryVolume(vol .> t, β, νmin;
            bbox_origin = Point3d(vol.params.bbox.origin), voxel_size = vol.params.voxel_size, root = root)
end

"""
    DijkstraShortestPath

Represents the shortest path to the closest source for all node of a graph.
"""
struct DijkstraShortestPath
    # distance field
    dist::Vector{Float64}

    # parent nodes on the shortest path
    prev::Vector{Int}

    # nearest source (in case of multi-source)
    nearest_source::Vector{Int}
end

"""
    dijkstra_shortest_path(g::ROMIBinaryVolume, src::Union{Int, AbstractVector{Int}}; weighted::Bool = true)

Distance from each node to its nearest source. `src` can be a single node ID or a vector of node ID.
Could either use edge weights or euclidean distance by specifying `weighted`.
"""
function dijkstra_shortest_path(vb::ROMIBinaryVolume, src::Union{Int, AbstractVector{Int}}; weighted::Bool = true)
    dist = fill(Inf, vb.nv)
    prev = zeros(Int, vb.nv)
    nearest_source = zeros(Int, vb.nv)
    visited = falses(vb.nv)

    pq = PriorityQueue{Int, Float64}()
    for s in src
        dist[s] = 0.0
        nearest_source[s] = s
        pq[s] = 0.0
    end
    while !isempty(pq)
        v = dequeue!(pq)
        visited[v] && continue
        visited[v] = true
        for idx in vb.adj_ptr[v]:(vb.adj_ptr[v + 1] - 1)
            m = vb.adj_nbr[idx]
            visited[m] && continue
            alt = dist[v] + (weighted ? vb.adj_w[idx] : vb.adj_d[idx])
            if alt < dist[m]
                dist[m] = alt
                prev[m] = v
                nearest_source[m] = nearest_source[v]
                pq[m] = alt
            end
        end
    end
    return DijkstraShortestPath(dist, prev, nearest_source)
end

"""
    extract_shortest_path(s::DijkstraShortestPath, start_node::Int)

Shortest path toward the nearest source node. Returns a vector of node id.
"""
function extract_shortest_path(d::DijkstraShortestPath, start_node::Int)
    path_id = Int[] # node id from tip → source
    s = d.nearest_source[start_node] # nearest source
    current = start_node
    
    # walk backward until we hit the source or an unreachable node (0)
    while current != 0
        push!(path_id, current)
        if current == s
            break
        end
        current = d.prev[current]
    end
    return path_id
end

"""
    arclength(points::AbstractVector{Point3})

Cumulative arc length along a curve.
"""
function arclength(points::AbstractVector{Point3{F}}) where {F <: AbstractFloat}
    n = length(points)
    t = zeros(F, n)
    for i in 2:n
        t[i] = t[i - 1] + norm(points[i] - points[i - 1])
    end
    return t
end

"""
    NWSmoothedCurve{F}

Callable Nadaraya–Watson smoothed curve, evaluated in the same chord-length
parameter `t` as `arclength(points)`. `curve(t)` gives position, `derivative`/
`tangent`/`evaluate_with_tangent` give the local derivative / unit tangent.

Valid roughly over `t ∈ [t_start - k_trunc*σ, t_end + k_trunc*σ]` — the
point-reflection padding only extends that far past each endpoint.
"""
struct NWSmoothedCurve{F <: AbstractFloat}
    padded_points::Vector{Point3{F}}
    padded_t::Vector{F}
    inv_2σ²::F
    inv_σ²::F
    trunc_radius::F
    t_start::F
    t_end::F

    # optional pinning
    p_start::Point3{F}
    p_end::Point3{F}
end

# constructor
function NWSmoothedCurve(points::AbstractVector{Point3{F}}, σ::Real;
                          k_trunc::Int = 6) where {F <: AbstractFloat}
    n = length(points)
    @assert n ≥ 2 "Need at least 2 points!"

    t_raw = arclength(points)
    t_start, t_end = t_raw[1], t_raw[end]
    σ_F = convert(F, σ)
    trunc_radius = k_trunc * σ_F

    # Bound padding search range in O(log N) rather than scanning all N points
    i_prefix_max = searchsortedlast(t_raw, t_start + trunc_radius)
    i_suffix_min = searchsortedfirst(t_raw, t_end - trunc_radius)

    n_prefix = max(0, i_prefix_max - 1)
    n_suffix = max(0, n - i_suffix_min)
    total_len = n_prefix + n + n_suffix

    padded_points = Point3{F}[]
    padded_t = F[]
    sizehint!(padded_points, total_len)
    sizehint!(padded_t, total_len)

    # Point-reflection prefix padding
    for i in i_prefix_max:-1:2
        dt = t_raw[i] - t_start
        push!(padded_points, 2 * points[1] - points[i])
        push!(padded_t, t_start - dt)
    end
    append!(padded_points, points)
    append!(padded_t, t_raw)

    # Point-reflection suffix padding
    for i in (n - 1):-1:i_suffix_min
        dt = t_end - t_raw[i]
        push!(padded_points, 2 * points[end] - points[i])
        push!(padded_t, t_end + dt)
    end

    return NWSmoothedCurve(padded_points, padded_t, inv(2 * (σ_F ^ 2)), inv(σ_F ^ 2),
                            trunc_radius, t_start, t_end, points[1], points[end])
end

# random access — safe for arbitrary, non-monotonic t (used by the public API)
function _window(padded_t::Vector{F}, t, trunc_radius) where {F}
    n = length(padded_t)
    lo = searchsortedfirst(padded_t, t - trunc_radius)
    hi = searchsortedlast(padded_t, t + trunc_radius)
    if lo > hi
        j = clamp(searchsortedlast(padded_t, t), 1, n - 1)
        lo, hi = j, j + 1
    end
    return lo, hi
end

# forward-only for a monotonic non-decreasing sequence of t
function _advance_window(lo::Int, hi::Int, padded_t::Vector{F},
                                  t, trunc_radius) where {F}
    n = length(padded_t)
    while (lo < n) && (padded_t[lo] < t - trunc_radius)
        lo += 1
    end
    while (hi < n) && (padded_t[hi + 1] ≤ t + trunc_radius)
        hi += 1
    end
    i_lo, i_hi = lo, hi

    # must have t ≥ padded_t[1]!
    if i_lo > i_hi
        j = clamp(hi, 1, n - 1)
        i_lo, i_hi = j, j + 1
    end
    return lo, hi, i_lo, i_hi
end

# core weighted sum
function _accumulate(padded_points::Vector{Point3{F}}, padded_t::Vector{F},
                              inv_2σ²::F, inv_σ²::F, t, lo::Int, hi::Int,
                              ::Val{want_deriv}) where {F, want_deriv}
    S0 = zero(F)
    S1 = zero(Point3{F})
    if want_deriv
        S0p = zero(F)
        S1p = zero(Point3{F})
    end
    @inbounds for i in lo:hi
        dt = padded_t[i] - t
        w = exp(-abs2(dt) * inv_2σ²)
        S0 += w
        S1 += w * padded_points[i]
        if want_deriv
            wp = w * dt * inv_σ²
            S0p += wp
            S1p += wp * padded_points[i]
        end
    end
    return want_deriv ? (S0, S1, S0p, S1p) : (S0, S1)
end

# callable API

# Position C(s)
function (curve::NWSmoothedCurve{F})(t) where {F}
    lo, hi = _window(curve.padded_t, t, curve.trunc_radius)
    S0, S1 = _accumulate(curve.padded_points, curve.padded_t, curve.inv_2σ²,
                          curve.inv_σ², t, lo, hi, Val(false))
    return S1 / S0
end

# Derivative dC/dt
function _pos_and_raw_derivative(curve::NWSmoothedCurve{F}, t) where {F}
    lo, hi = _window(curve.padded_t, t, curve.trunc_radius)
    S0, S1, S0p, S1p = _accumulate(curve.padded_points, curve.padded_t, curve.inv_2σ²,
                                    curve.inv_σ², t, lo, hi, Val(true))
    pos = S1 / S0
    deriv = (S1p - pos * S0p) / S0
    return pos, deriv
end
derivative(curve::NWSmoothedCurve, t) = _pos_and_raw_derivative(curve, t)[2]

# Unit Tangent T(s) = dC/ds
function tangent(curve::NWSmoothedCurve, t)
    d = derivative(curve, t)
    return d / sqrt(sum(abs2, d))
end
function evaluate_with_tangent(curve::NWSmoothedCurve, t)
    pos, d = _pos_and_raw_derivative(curve, t)
    return pos, d / sqrt(sum(abs2, d))
end

"""
    ArcLengthNWSmoothedCurve{F <: AbstractFloat}

Unit-speed (arc-length) reparameterized NWSmoothedCurve.
"""
struct ArcLengthNWSmoothedCurve{F <: AbstractFloat}
    curve::NWSmoothedCurve{F}
    s_profile::Vector{F} # Monotonic smoothed arc lengths [0, L_smooth]
    t_dense::Vector{F} # Corresponding raw parameter t values [0, L_raw]
    dense_pts::Vector{Point3{F}} # densely sample points
    L_smooth::F # Total exact length of the smoothed curve
end

# constructor
function ArcLengthNWSmoothedCurve(points::AbstractVector{Point3{F}}, σ::Real;
                       dt_factor::Real = 0.25, n_min::Int = 100, k_trunc::Int = 6) where {F <: AbstractFloat}
    curve = NWSmoothedCurve(points, σ; k_trunc = k_trunc)
    
    # Dense sampling in raw parameter t to build an accurate s(t) lookup table
    dt_dense = convert(F, σ) * convert(F, dt_factor)
    dense_n = max(n_min, ceil(Int, (curve.t_end - curve.t_start) / dt_dense) + 1)
    t_dense = collect(range(curve.t_start, curve.t_end; length = dense_n))
    
    dense_pts = Vector{Point3{F}}(undef, dense_n)
    lo, hi = 1, 1
    for i in 1:dense_n
        lo, hi, w_lo, w_hi = _advance_window(lo, hi, curve.padded_t, t_dense[i], curve.trunc_radius)
        S0, S1 = _accumulate(curve.padded_points, curve.padded_t, curve.inv_2σ²,
                             curve.inv_σ², t_dense[i], w_lo, w_hi, Val(false))
        dense_pts[i] = S1 / S0
    end
    
    s_profile = arclength(dense_pts)
    L_smooth = s_profile[end]
    
    return ArcLengthNWSmoothedCurve(curve, s_profile, t_dense, dense_pts, L_smooth)
end

# map smoothed arc length s ∈ [0, L_smooth] to raw parameter t ∈ [0, L_raw] (and reciprocally from t to s)
function s_to_t(ac::ArcLengthNWSmoothedCurve{F}, s) where {F}
    s_F = convert(F, s)
    if s_F ≤ 0
        return ac.curve.t_start
    elseif s_F ≥ ac.L_smooth
        return ac.curve.t_end
    end

    idx = clamp(searchsortedfirst(ac.s_profile, s_F) - 1, 1, length(ac.s_profile) - 1)
    ds_step = ac.s_profile[idx + 1] - ac.s_profile[idx]
    frac = (iszero(ds_step) ? zero(F) : (s_F - ac.s_profile[idx]) / ds_step)
    return ac.t_dense[idx] + frac * (ac.t_dense[idx + 1] - ac.t_dense[idx])
end
function t_to_s(ac::ArcLengthNWSmoothedCurve{F}, t) where {F}
    if t ≤ ac.curve.t_start
        return zero(F)
    elseif t ≥ ac.curve.t_end
        return ac.L_smooth
    end

    idx = clamp(searchsortedfirst(ac.t_dense, t) - 1, 1, length(ac.t_dense) - 1)
    dt_step = ac.t_dense[idx + 1] - ac.t_dense[idx]
    frac = (iszero(dt_step) ? zero(F) : (t - ac.t_dense[idx]) / dt_step)
    return ac.s_profile[idx] + frac * (ac.s_profile[idx + 1] - ac.s_profile[idx])
end

# callable API

# Position C(s)
function (ac::ArcLengthNWSmoothedCurve)(s)
    t = s_to_t(ac, s)
    return ac.curve(t)
end

# Unit Tangent T(s) = dC/ds
function tangent(ac::ArcLengthNWSmoothedCurve, s)
    t = s_to_t(ac, s)
    return tangent(ac.curve, t)
end
function evaluate_with_tangent(ac::ArcLengthNWSmoothedCurve, s)
    t = s_to_t(ac, s)
    return evaluate_with_tangent(ac.curve, t)
end

"""
    sample_uniform(ac::ArcLengthNWSmoothedCurve, ds::Real)

Uniformly sample positions along a smoothed 1D curve at chord length step size `ds`.
We use the end points to pin the curve exactly. While the reflection padding 
in NWSmoothedCurve garanty that the end points are exactly the raw end points 
the pinning remove the litle numerical jitter to recover bitwise identity!
"""
function sample_uniform(ac::ArcLengthNWSmoothedCurve{F}, ds::Real) where {F <: AbstractFloat}
    total_len = ac.L_smooth
    num_samples = max(2, ceil(Int, total_len / convert(F, ds)) + 1)
    s_uniform = range(zero(F), total_len; length = num_samples)

    pts = Vector{Point3{F}}(undef, num_samples)
    
    # end points pinning
    pts[1] = ac.curve.p_start
    pts[end] = ac.curve.p_end

    dense_n = length(ac.s_profile)
    lo, hi = 1, 1
    idx = 1
    for i in 2:(num_samples - 1)
        s_val = s_uniform[i]

        # slide s_profile pointer forward
        while (idx < dense_n - 1) && (ac.s_profile[idx + 1] < s_val)
            idx += 1
        end

        # map s -> t
        ds_step = ac.s_profile[idx + 1] - ac.s_profile[idx]
        frac = iszero(ds_step) ? zero(F) : (s_val - ac.s_profile[idx]) / ds_step
        t_target = ac.t_dense[idx] + frac * (ac.t_dense[idx + 1] - ac.t_dense[idx])

        # slide padded_t window forward
        lo, hi, w_lo, w_hi = _advance_window(lo, hi, ac.curve.padded_t, t_target, ac.curve.trunc_radius)

        # accumulate weights
        S0, S1 = _accumulate(ac.curve.padded_points, ac.curve.padded_t,
                            ac.curve.inv_2σ², ac.curve.inv_σ²,
                            t_target, w_lo, w_hi, Val(false))
        pts[i] = S1 / S0
    end

    return pts
end

"""
    sample_uniform_with_tangents(ac::ArcLengthNWSmoothedCurve, ds::Real)

Similar to `sample_uniform`, but also sample the unit tangent vectors.
"""
function sample_uniform_with_tangents(ac::ArcLengthNWSmoothedCurve{F}, ds::Real) where {F <: AbstractFloat}
    total_len = ac.L_smooth
    num_samples = max(2, ceil(Int, total_len / convert(F, ds)) + 1)
    s_uniform = range(zero(F), total_len; length = num_samples)

    pts = Vector{Point3{F}}(undef, num_samples)
    tgt = Vector{Vec3{F}}(undef, num_samples)
    
    # end points pinning
    pts[1] = ac.curve.p_start
    pts[end] = ac.curve.p_end

    dense_n = length(ac.s_profile)
    lo, hi = 1, 1
    idx = 1
    for i in 1:num_samples
        s_val = s_uniform[i]

        if i == 1
            t_target = ac.curve.t_start
        elseif i == num_samples
            t_target = ac.curve.t_end
        else
            # slide s_profile pointer forward
            while (idx < dense_n - 1) && (ac.s_profile[idx + 1] < s_val)
                idx += 1
            end

            # map s -> t
            ds_step = ac.s_profile[idx + 1] - ac.s_profile[idx]
            frac = iszero(ds_step) ? zero(F) : (s_val - ac.s_profile[idx]) / ds_step
            t_target = ac.t_dense[idx] + frac * (ac.t_dense[idx + 1] - ac.t_dense[idx])
        end

        # slide padded_t window forward
        lo, hi, w_lo, w_hi = _advance_window(lo, hi, ac.curve.padded_t, t_target, ac.curve.trunc_radius)

        # accumulate weights for both position & derivative weights
        S0, S1, S0p, S1p = _accumulate(ac.curve.padded_points, ac.curve.padded_t,
                                       ac.curve.inv_2σ², ac.curve.inv_σ²,
                                       t_target, w_lo, w_hi, Val(true))

        # end points pinning
        if i == 1
            pts[i] = ac.curve.p_start
        elseif i == num_samples
            pts[i] = ac.curve.p_end
        else
            pts[i] = S1 / S0
        end

        # unit tangent T(s) = dP/dt / ||dP/dt||
        deriv = (S1p - (S1 / S0) * S0p) / S0
        tgt[i] = deriv / sqrt(sum(abs2, deriv))
    end

    return pts, tgt
end

# Code adapted from Press, W. H. (2007), Numerical recipes 3rd edition: The art of scientific computing, p460-461
function newtsafe_rootfinding(f, df, a, b; x₀ = (a + b) / 2, tol = 10 * eps(), maxiter = 100, offset = 0.0)
    fa = f(a) + offset
    fb = f(b) + offset
    (abs(fa) < tol) && return (a, true)
    (abs(fb) < tol) && return (b, true)
    @assert (fa * fb) ≤ 0 "Root not bracketed"
    # orient the search so that f(xₗ) < 0
    xₗ, xₕ = ifelse(fa > 0, (b, a), (a, b))
    x = x₀
    dx_old = abs(xₗ - xₕ)
    dx = dx_old
    fx = f(x) + offset # use the offset to solve for f(x) = offset
    (abs(fx) < tol) && return (x, true)
    dfx = df(x)
    for _ in 1:maxiter
        # fallback to bisection if Newton would go out of bounds or not converge fast enough
        reject_newton = ((((x - xₕ) * dfx - fx) * ((x - xₗ) * dfx - fx)) > 0) || ((abs(2.0 * fx) > abs(dx_old * dfx)))
        dx_old = dx
        dx = (reject_newton ? (0.5 * (xₕ - xₗ)) : (fx / dfx))
        x = ifelse(reject_newton, (xₗ + dx), (x - dx))
        (abs(dx) < tol) && return (x, true)
        fx = f(x) + offset
        dfx = df(x)
        # update brackets
        lower_bracket_update = fx < 0
        xₗ = ifelse(lower_bracket_update, x, xₗ)
        xₕ = ifelse(lower_bracket_update, xₕ, x)
    end
    @warn "Reached maximum number of iterations" func = "newtsafe_rootfinding"
    return (x, false)
end

"""
    closest_point_t(ac::ArcLengthNWSmoothedCurve, target::Point3)

Project `target` onto the smoothed curve, returning the chord-length parameter
`t` of the closest point.
"""
function closest_point_t(ac::ArcLengthNWSmoothedCurve{F}, target::Point3{F}; tol = 1e-6) where {F <: AbstractFloat}
    dense_pts = ac.dense_pts
    n_dense = length(dense_pts)

    # coarse search to locate discrete minimum index
    best_i = argmin(i -> sum(abs2, dense_pts[i] - target), 1:n_dense)

    # handle exact endpoints
    if best_i == 1
        return ac.curve.t_start
    elseif best_i == n_dense
        return ac.curve.t_end
    end
    x0 = ac.t_dense[best_i]

    # caching to avoid recomputation
    cache_t = Ref(F(NaN))
    cache_pos = Ref{Point3{F}}(zero(Point3{F}))
    cache_deriv = Ref{Point3{F}}(zero(Point3{F}))
    function update_cache!(t)
        if cache_t[] !== t
            cache_pos[], cache_deriv[] = _pos_and_raw_derivative(ac.curve, t)
            cache_t[] = t
        end
    end

    # esidual function f(t) = (p(t) - target) ⋅ p'(t)
    f = t -> begin
        update_cache!(t)
        return sum((cache_pos[] - target) .* cache_deriv[])
    end

    # Gauss-Newton approximation of the derivative function df(t) ≈ ||p'(t)||^2
    df = t -> begin
        update_cache!(t)
        return sum(abs2, cache_deriv[])
    end

    # solve via NewtSafe root finding
    t_opt, _ = newtsafe_rootfinding(f, df, ac.curve.t_start, ac.curve.t_end; x₀ = x0, tol = tol)

    return t_opt
end

# to keep track of connected components
struct UnionFind
    parent::Vector{Int}
end
UnionFind(n::Int) = UnionFind(collect(1:n))
function find(uf::UnionFind, x)
    # path-halving
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]]
        x = uf.parent[x]
    end
    return x
end

"""
    JoinTree{F}

Represents the superlevel-set topology of a scalar field on a graph.
"""
struct JoinTree{F}
    # peak label for each voxel
    seg::Vector{Int}

    # topological persistence per peak ID
    prominence::Vector{F}

    # voxel index of the merge saddle for each peak
    saddle_node::Vector{Int}

    # scalar value u at the saddle for each peak
    saddle_val::Vector{F}

    # ancestor peak into which each peak merges
    parent_peak::Vector{Int}

    # scalar field value
    u::Vector{F}

    # voxel coordinates
    coords::Vector{CartesianIndex{3}}
end

"""
    build_join_tree(vb::ROMIBinaryVolume, u::AbstractVector{F})

Performs a superlevel-set filtration sweep to construct the Join Tree of the scalar field `u`.
"""
function build_join_tree(vb::ROMIBinaryVolume, u::AbstractVector{F}) where {F}
    N = vb.nv
    order = sortperm(u; rev = true) # sweep from highest potential to lowest
    uf = UnionFind(N)
    processed = falses(N)
    peak_of = zeros(Int, N) # component root → node ID of its peak

    # initialize segmentation and saddle tracking vectors
    seg = zeros(Int, N) # basin/peak label per voxel
    prominence = fill(-one(F), N) # stores finalized prominence values (-1 means not a peak)
    saddle_node = zeros(Int, N)
    saddle_val = fill(-one(F), N)
    parent_peak = zeros(Int, N)

    # Buffer for unique neighbor roots (max degree in 26-connectivity is 26)
    roots_buf = zeros(Int, 26)

    # superlevel-set filtration sweep
    for v in order
        num_roots = 0

        # scan neighborhood
        for idx in vb.adj_ptr[v]:(vb.adj_ptr[v + 1] - 1)
            w = vb.adj_nbr[idx]
            if processed[w]
                r = find(uf, w)
                already_added = false
                for i in 1:num_roots
                    if roots_buf[i] == r
                        already_added = true
                        break
                    end
                end
                
                if !already_added
                    num_roots += 1
                    roots_buf[num_roots] = r
                end
            end
        end

        if num_roots == 0
            # v is a brand new local maximum (isolated peak)
            uf.parent[v] = v
            peak_of[v] = v
            prominence[v] = typemax(F)
        else
            # v bridges existing components; sort neighbor roots by peak height descending
            for i in 2:num_roots
                key = roots_buf[i]
                key_val = u[peak_of[key]]
                j = i - 1
                while (j ≥ 1) && (u[peak_of[roots_buf[j]]] < key_val)
                    roots_buf[j + 1] = roots_buf[j]
                    j -= 1
                end
                roots_buf[j + 1] = key
            end

            # The tallest peak survives, shorter peaks die
            main_root = roots_buf[1]
            uf.parent[v] = main_root
            
            for i in 2:num_roots
                r = roots_buf[i]
                lower_peak = peak_of[r]

                # Record topological saddle data
                saddle_node[lower_peak] = v
                saddle_val[lower_peak] = u[v]
                parent_peak[lower_peak] = peak_of[main_root]
                
                # Finalize prominence of dying peak at current saddle height u[v]
                prominence[lower_peak] = u[lower_peak] - u[v]
                uf.parent[r] = main_root
            end
        end
        seg[v] = peak_of[find(uf, v)]
        processed[v] = true
    end

    return JoinTree{F}(seg, prominence, saddle_node, saddle_val, parent_peak, collect(u), copy(vb.coords))
end

"""
    FieldMaxima{F}

Represents a local maxima of a scalar field on a graph/grid.
"""
struct FieldMaxima{F <: AbstractFloat}
    node_id::Int
    node_idx::CartesianIndex{3}
    prominence::F
    potential::F
end

"""
    findlocalmaxima(jt::JoinTree{F}, h::Real)

Extract local maxima with topographic prominence ≥ `h` from a Join tree.
Returns a Vector of `FieldMaxima`, sorted by prominence descending.
The global maximum is guaranteed to have a prominence of `Inf`.
"""
function findlocalmaxima(jt::JoinTree{F}, h::Real) where {F}
    # collect and package the valid peaks
    maxima = FieldMaxima{F}[]
    for v in eachindex(jt.prominence)
        if jt.prominence[v] ≥ h
            m = FieldMaxima(v, jt.coords[v], jt.prominence[v], jt.u[v])
            push!(maxima, m)
        end
    end
    
    # sort final list by prominence descending
    sort!(maxima; by = x -> x.prominence, rev = true)
    
    return maxima
end

"""
    findtruncatedtips(vb::ROMIBinaryVolume; min_voxels::Int = 3)

Finds tip points for truncated faces other than the bottom ones.
The tips are the points of maximal euclidean distance transform value.
"""
function findtruncatedtips(vb::ROMIBinaryVolume; min_voxels::Int = 3)
    tips = FieldMaxima{Float64}[]
    for (axis, at) in zip((3, 2, 2, 1, 1), (size(vb, 3), 1, size(vb, 2), 1, size(vb, 1)))
        slice = selectdim(vb, axis, at)
        labels = label_components(slice)
        counts = component_lengths(labels)
        for (i, indices) in Iterators.enumerate(Iterators.map(ixs -> add_missing_coordinate.(ixs, axis, at), Iterators.drop(component_indices(CartesianIndex, labels), 1)))
            (counts[i] < min_voxels) && continue
            u, ix = findmax(view(vb.dt, indices))
            t = indices[ix]
            t_id = vb.idmap[t]
            push!(tips, FieldMaxima(t_id, t, -1.0, u))
        end
    end
    return tips
end

mutable struct ROMISkeleton # mutable to add or remove tips
    # raw stem and branches (define at the voxel scale)
    stem::Vector{Int}
    branch::Vector{Vector{Int}}

    # smoothed stem and branches
    stem_curve::ArcLengthNWSmoothedCurve{Float64}
    branch_curve::Vector{ArcLengthNWSmoothedCurve{Float64}}

    # stem root and top
    stem_root::Point3d
    stem_top::Point3d
    stem_top_id::Int

    # fruits tips and branch points
    tip_ids::Vector{Int}
    tip_points::Vector{Point3d}
    branchpoints::Vector{Point3d}
    branchpoints_arclength::Vector{Float64}

    # binary volume and its graph representation
    vb::ROMIBinaryVolume

    # distance fields
    u_root::DijkstraShortestPath
    jt_root::JoinTree{Float64}
    u_stem::DijkstraShortestPath
    u_branch::DijkstraShortestPath

    # parameters used
    params::ROMISkeletonParams
end
function Base.copy(s::ROMISkeleton)
    return ROMISkeleton(s.stem,
                s.branch,
                s.stem_curve,
                s.branch_curve,
                s.stem_root,
                s.stem_top,
                s.stem_top_id,
                s.tip_ids,
                s.tip_points,
                s.branchpoints,
                s.branchpoints_arclength,
                s.vb,
                s.u_root, 
                s.jt_root,
                s.u_stem,
                s.u_branch,
                s.params)
end

# convenient constructors
function ROMISkeleton(vb::ROMIBinaryVolume, params::ROMISkeletonParams)
    @info "Building skeleton..."
    # identify local maxima from the distance field to the root (this is the tip of the fruits)
    root_point = vb.vox_grid[vb.root]
    u_root = dijkstra_shortest_path(vb, vb.root_id; weighted = false) # euclidean distance
    jt_root = build_join_tree(vb, u_root.dist)
    t_max = findlocalmaxima(jt_root, params.h) # filter by prominence

    # filter tips on truncated faces
    filter!(m -> m.node_idx ∉ Set(vb.tf), t_max)

    # find truncated face tip (idealy only one corresponding to the top of the stem)
    t_trunc = findtruncatedtips(vb)

    # identify the stem top as having the largest radius
    if isempty(t_trunc) # use the farthest local maxima from the root
        stem_top = argmax(t -> u_root.dist[t.node_id], t_max)
        deleteat!(t_max, findfirst(==(stem_top), t_max))
    else
        if findfirst(t -> t.node_idx.I[3] == size(vb, 3), t_trunc) === nothing
            if isempty(t_max)
                stem_top = argmax(t -> u_root.dist[t.node_id], t_trunc)
                deleteat!(t_trunc, findfirst(==(stem_top), t_trunc))
            else
                stem_top = argmax(t -> u_root.dist[t.node_id], t_max)
                deleteat!(t_max, findfirst(==(stem_top), t_max))
            end
        else
            stem_top = argmax(t -> ((t.node_idx.I[3] == size(vb, 3)) ? vb.dt[t.node_idx] : -Inf), t_trunc)
            deleteat!(t_trunc, findfirst(==(stem_top), t_trunc))
        end
        append!(t_max, t_trunc)
    end
    stem_id = stem_top.node_id
    stem_point = vb.vox_grid[stem_top.node_idx]
    tip_points = vb.vox_grid[map(m -> m.node_idx, t_max)]
    tip_ids = map(m -> m.node_id, t_max)

    # extract main stem center line via conductance weighted distance field
    u_stem = dijkstra_shortest_path(vb, vb.root_id; weighted = true) # flux travel time
    raw_stem = reverse!(extract_shortest_path(u_stem, stem_id)) # from root → top

    # extract the branches from tips shortest path to the stem
    u_branch = dijkstra_shortest_path(vb, raw_stem; weighted = true)
    raw_branch = map(m -> reverse!(extract_shortest_path(u_branch, m.node_id)), t_max) # from stem → fruit tip
    raw_branchpoints = map(b -> vb.vox_grid[vb.coords[b[1]]], raw_branch) # coordinates of the junction point

    # Nadaraya-Watson smoothing to get sub-voxel accuracy
    @views smooth_stem_curve = ArcLengthNWSmoothedCurve(vb.vox_grid[vb.coords[raw_stem]], params.σstem;
                                                            dt_factor = params.dt_factor,
                                                            n_min = params.n_min,
                                                            k_trunc = params.k_trunc)

    # idem with each branch
    smooth_branch_curve = Vector{ArcLengthNWSmoothedCurve{Float64}}(undef, length(raw_branch))
    branchpoints = Vector{Point3d}(undef, length(raw_branch))
    branchpoints_arclength = Vector{Float64}(undef, length(raw_branch))
    Threads.@threads for i in eachindex(smooth_branch_curve)
        # find where the junction point maps onto the smoothed stem
        t = closest_point_t(smooth_stem_curve, raw_branchpoints[i])
        branchpoints_arclength[i] = t_to_s(smooth_stem_curve, t)
        branchpoints[i] = smooth_stem_curve.curve(t)

        # smooth the branch
        @views branch = vcat([branchpoints[i]], vb.vox_grid[vb.coords[raw_branch[i]]])
        smooth_branch_curve[i] = ArcLengthNWSmoothedCurve(branch, params.σbranch;
                                                            dt_factor = params.dt_factor,
                                                            n_min = params.n_min,
                                                            k_trunc = params.k_trunc)
    end

    # sort fruits by branchpoint arc length
    if length(branchpoints) > 1
        perm = sortperm(branchpoints_arclength)
        permute!(raw_branch, perm)
        permute!(smooth_branch_curve, perm)
        permute!(tip_ids, perm)
        permute!(tip_points, perm)
        permute!(branchpoints, perm)
        permute!(branchpoints_arclength, perm)
    end

    return ROMISkeleton(raw_stem, raw_branch, smooth_stem_curve, smooth_branch_curve,
        root_point, stem_point, stem_id, tip_ids, tip_points, branchpoints, branchpoints_arclength,
        vb, u_root, jt_root, u_stem, u_branch, params)
end
function ROMISkeleton(vb::AbstractArray{Bool, 3}, params::ROMISkeletonParams;
                bbox_origin::Point3 = zero(Point3d), voxel_size::Real = 1.0, root::Union{Nothing, Int} = nothing)
    # create a binary volume
    @info "Building binary volume..."
    vb = ROMIBinaryVolume(vb, params.β, params.νmin; bbox_origin = bbox_origin, voxel_size = voxel_size, root = root)
    return ROMISkeleton(vb, params)
end
function ROMISkeleton(v::ROMIVolume, params::ROMISkeletonParams; root::Union{Nothing, Int} = nothing)
    # create a binary volume
    @info "Building binary volume..."
    vb = ROMIBinaryVolume(v, params.t, params.β, params.νmin; root = root)
    return ROMISkeleton(vb, params)
end

# display
Base.show(io::IO, s::ROMISkeleton) = print(io, "ROMISkeleton with ", length(s.branchpoints), " branch(es)")

# shared logic for (re)deriving one branch against the current stem
function _recompute_branch!(s::ROMISkeleton, i::Int)
    vb = s.vb
    raw_branch = reverse!(extract_shortest_path(s.u_branch, s.tip_ids[i]))
    raw_branchpoint = vb.vox_grid[vb.coords[raw_branch[1]]]

    # find where the junction point maps onto the smoothed stem
    t = closest_point_t(s.stem_curve, raw_branchpoint)
    branchpoint_arclength = t_to_s(s.stem_curve, t)
    branchpoint = s.stem_curve.curve(t)

    # smooth the branch
    @views branch = vcat([branchpoint], vb.vox_grid[vb.coords[raw_branch]])
    smooth_branch_curve = ArcLengthNWSmoothedCurve(branch, s.params.σbranch;
                                                        dt_factor = s.params.dt_factor,
                                                        n_min = s.params.n_min,
                                                        k_trunc = s.params.k_trunc)

    # unsorted need to be resort after
    s.branch[i] = raw_branch
    s.branch_curve[i] = smooth_branch_curve
    s.branchpoints[i] = branchpoint
    s.branchpoints_arclength[i] = branchpoint_arclength

    return nothing
end

"""
    update_stem_root!(s::ROMISkeleton, node_id::Int)

Re-extract the stem centerline using `node_id` as the new root.
Every existing fruit branch is re-derived against the new stem.
"""
function update_stem_root!(s::ROMISkeleton, node_id::Int)
    (s.vb.root_id == node_id) && return nothing

    # we need to make a new ROMIBinaryVolume instance
    vb = ROMIBinaryVolume(s.vb.vb, s.params.β, s.params.νmin;
            bbox_origin = Point3d(Rect3(s.vb.vox_grid).origin), voxel_size = s.vb.voxel_size, root = node_id)
    s.vb = vb

    # identify local maxima from the distance field to the root (this is the tip of the fruits)
    s.stem_root = vb.vox_grid[vb.root]
    s.u_root = dijkstra_shortest_path(vb, vb.root_id; weighted = false) # euclidean distance
    s.jt_root = build_join_tree(vb, s.u_root.dist)
    t_max = findlocalmaxima(s.jt_root, s.params.h) # filter by prominence

    # filter tips on truncated faces
    filter!(m -> m.node_idx ∉ Set(vb.tf), t_max)

    # find truncated face tip (idealy only one corresponding to the top of the stem)
    t_trunc = findtruncatedtips(vb)

    # keep the existing stem top
    if !isempty(t_trunc)
        ix_tmax = findfirst(==(s.stem_top_id), map(m -> m.node_id, t_max))
        ix_ttrunc = findfirst(==(s.stem_top_id), map(m -> m.node_id, t_trunc))
        if !isnothing(ix_tmax)
            deleteat!(t_max, ix_tmax)
        elseif !isnothing(ix_ttrunc)
            deleteat!(t_trunc, ix_ttrunc)
        end
        append!(t_max, t_trunc)
    end
    s.tip_points = vb.vox_grid[map(m -> m.node_idx, t_max)]
    s.tip_ids = map(m -> m.node_id, t_max)

    # extract main stem center line via conductance weighted distance field
    s.u_stem = dijkstra_shortest_path(vb, vb.root_id; weighted = true) # flux travel time
    s.stem = reverse!(extract_shortest_path(s.u_stem, s.stem_top_id)) # from root → top

    # stem changed → branch-attachment distance field must be rebuilt
    s.u_branch = dijkstra_shortest_path(vb, s.stem; weighted = true)

    # recompute stem curve
    @views s.stem_curve = ArcLengthNWSmoothedCurve(vb.vox_grid[vb.coords[s.stem]], s.params.σstem;
                                                            dt_factor = s.params.dt_factor,
                                                            n_min = s.params.n_min,
                                                            k_trunc = s.params.k_trunc)

    # resize branch containers
    n_tips = length(s.tip_ids)
    resize!(s.branch, n_tips)
    resize!(s.branch_curve, n_tips)
    resize!(s.branchpoints, n_tips)
    resize!(s.branchpoints_arclength, n_tips)

    Threads.@threads for i in eachindex(s.tip_ids)
        _recompute_branch!(s, i)
    end

    # sort fruits by branchpoint arc length
    if length(s.branchpoints) > 1
        perm = sortperm(s.branchpoints_arclength)
        permute!(s.branch, perm)
        permute!(s.branch_curve, perm)
        permute!(s.tip_ids, perm)
        permute!(s.tip_points, perm)
        permute!(s.branchpoints, perm)
        permute!(s.branchpoints_arclength, perm)
    end

    return nothing
end

"""
    update_stem_top!(s::ROMISkeleton, node_id::Int)

Re-extract the stem centerline using `node_id` as the new top.
Every existing fruit branch is re-derived against the new stem.
"""
function update_stem_top!(s::ROMISkeleton, node_id::Int)
    (s.stem_top_id == node_id) && return nothing
    vb = s.vb
    s.stem_top_id = node_id
    s.stem_top = vb.vox_grid[vb.coords[node_id]]
    s.stem = reverse!(extract_shortest_path(s.u_stem, node_id))

    # stem changed → branch-attachment distance field must be rebuilt
    s.u_branch = dijkstra_shortest_path(vb, s.stem; weighted = true)

    # recompute stem and branch curves
    @views s.stem_curve = ArcLengthNWSmoothedCurve(vb.vox_grid[vb.coords[s.stem]], s.params.σstem;
                                                            dt_factor = s.params.dt_factor,
                                                            n_min = s.params.n_min,
                                                            k_trunc = s.params.k_trunc)
    Threads.@threads for i in eachindex(s.tip_ids)
        _recompute_branch!(s, i)
    end

    # sort fruits by branchpoint arc length
    if length(s.branchpoints) > 1
        perm = sortperm(s.branchpoints_arclength)
        permute!(s.branch, perm)
        permute!(s.branch_curve, perm)
        permute!(s.tip_ids, perm)
        permute!(s.tip_points, perm)
        permute!(s.branchpoints, perm)
        permute!(s.branchpoints_arclength, perm)
    end

    return nothing
end

"""
    add_fruit_tip!(s::ROMISkeleton, node_id::Int)

Add a new fruit tip, extracting its branch against the current stem via the already-built distance field.
"""
function add_fruit_tip!(s::ROMISkeleton, node_id::Int)
    (node_id in s.tip_ids) && return nothing # already present
    push!(s.tip_ids, node_id)
    push!(s.tip_points, s.vb.vox_grid[s.vb.coords[node_id]])
    
    # resize branch containers
    n_tips = length(s.tip_ids)
    resize!(s.branch, n_tips)
    resize!(s.branch_curve, n_tips)
    resize!(s.branchpoints, n_tips)
    resize!(s.branchpoints_arclength, n_tips)
    _recompute_branch!(s, length(s.tip_ids))

    # sort fruits by branchpoint arc length
    if length(s.branchpoints) > 1
        perm = sortperm(s.branchpoints_arclength)
        permute!(s.branch, perm)
        permute!(s.branch_curve, perm)
        permute!(s.tip_ids, perm)
        permute!(s.tip_points, perm)
        permute!(s.branchpoints, perm)
        permute!(s.branchpoints_arclength, perm)
    end

    return nothing
end

"""
    remove_fruit_tip!(s::ROMISkeleton, node_id::Int)

Remove a fruit tip and its branch by node id.
"""
function remove_fruit_tip!(s::ROMISkeleton, node_id::Int)
    i = findfirst(==(node_id), s.tip_ids)
    (i === nothing) && return nothing
    deleteat!(s.tip_ids, i)
    deleteat!(s.tip_points, i)
    deleteat!(s.branch, i)
    deleteat!(s.branch_curve, i)
    deleteat!(s.branchpoints, i)
    deleteat!(s.branchpoints_arclength, i)
    return nothing
end

"""
    update_weights!(s::ROMISkeleton, β::Real, νmin::Real)

Recomputes edge weights for the updated `β` and `νmin` values.
"""
function update_weights!(s::ROMISkeleton, β::Real, νmin::Real)
    ((s.params.β == β) && (s.params.νmin == νmin)) && return nothing
    s.params.β = β
    s.params.νmin = νmin
    update_weights!(s.vb, s.params.β, s.params.νmin)

    # update scalar fields
    s.u_stem = dijkstra_shortest_path(s.vb, s.vb.root_id; weighted = true)
    s.stem = reverse!(extract_shortest_path(s.u_stem, s.stem_top_id))
    s.u_branch = dijkstra_shortest_path(s.vb, s.stem; weighted = true)

    # recompute stem and branch curves
    @views s.stem_curve = ArcLengthNWSmoothedCurve(s.vb.vox_grid[s.vb.coords[s.stem]], s.params.σstem;
                                                            dt_factor = s.params.dt_factor,
                                                            n_min = s.params.n_min,
                                                            k_trunc = s.params.k_trunc)
    Threads.@threads for i in eachindex(s.tip_ids)
        _recompute_branch!(s, i)
    end

    # sort fruits by branchpoint arc length
    if length(s.branchpoints) > 1
        perm = sortperm(s.branchpoints_arclength)
        permute!(s.branch, perm)
        permute!(s.branch_curve, perm)
        permute!(s.tip_ids, perm)
        permute!(s.tip_points, perm)
        permute!(s.branchpoints, perm)
        permute!(s.branchpoints_arclength, perm)
    end

    return nothing
end
function update_weights!(vb::ROMIBinaryVolume, β::Real, νmin::Real)
    # recompute node conductances
    for v in 1:vb.nv
        I = vb.coords[v]

        # relative position from the center line (0: boundary → 1: center)
        r = (vb.rlocal[v] == 1.0 ? 1.0 : (vb.dt[I] - 1) / (vb.rlocal[v] - 1))

        # node conductance νmin at the boundary and 1 in the center
        vb.γ[v] = νmin + (1 - νmin) * (r ^ β)
    end

    # recompute graph edge weights
    for v in 1:vb.nv
        γ_v = vb.γ[v]
        for idx in vb.adj_ptr[v]:(vb.adj_ptr[v + 1] - 1)
            m = vb.adj_nbr[idx]
            γ_m = vb.γ[m]
            d = vb.adj_d[idx]
            vb.adj_w[idx] = d / (0.5 * (γ_v + γ_m))
        end
    end

    return nothing
end

"""
    update_threshold!(s::ROMISkeleton, v, t::Real)

Recomputes the binary volume for the updated threshold `t`.
"""
function update_threshold!(s::ROMISkeleton, v, t::Real)
    # check whether we need to recompute
    bbox_origin = Point3d(Rect3(v.vox_grid).origin)
    voxel_size = v.params.voxel_size
    vox_grid = ROMIVoxelGrid(bbox_origin, size(v), voxel_size)
    ((s.params.t == t) && (s.vb.vox_grid == vox_grid)) && return nothing

    s.params.t = t
    vb = ROMIBinaryVolume(v .> t, s.params.β, s.params.νmin; bbox_origin = bbox_origin, voxel_size = voxel_size)
    s.vb = vb

    # identify local maxima from the distance field to the root (this is the tip of the fruits)
    s.stem_root = vb.vox_grid[vb.root]
    s.u_root = dijkstra_shortest_path(vb, vb.root_id; weighted = false) # euclidean distance
    s.jt_root = build_join_tree(vb, s.u_root.dist)
    t_max = findlocalmaxima(s.jt_root, s.params.h) # filter by prominence

    # filter tips on truncated faces
    filter!(m -> m.node_idx ∉ Set(vb.tf), t_max)

    # find truncated face tip (idealy only one corresponding to the top of the stem)
    t_trunc = findtruncatedtips(vb)

    # identify the stem top as having the largest radius
    if isempty(t_trunc) # use the farthest local maxima from the root
        stem_top = argmax(t -> s.u_root.dist[t.node_id], t_max)
        deleteat!(t_max, findfirst(==(stem_top), t_max))
    else
        if findfirst(t -> t.node_idx.I[3] == size(vb, 3), t_trunc) === nothing
            stem_top = argmax(t -> s.u_root.dist[t.node_id], t_max)
            deleteat!(t_max, findfirst(==(stem_top), t_max))
        else
            stem_top = argmax(t -> ((t.node_idx.I[3] == size(vb, 3)) ? vb.dt[t.node_idx] : -Inf), t_trunc)
            deleteat!(t_trunc, findfirst(==(stem_top), t_trunc))
        end
        append!(t_max, t_trunc)
    end
    s.stem_top_id = stem_top.node_id
    s.stem_top = vb.vox_grid[stem_top.node_idx]
    s.tip_points = vb.vox_grid[map(m -> m.node_idx, t_max)]
    s.tip_ids = map(m -> m.node_id, t_max)

    # extract main stem center line via conductance weighted distance field
    s.u_stem = dijkstra_shortest_path(vb, vb.root_id; weighted = true) # flux travel time
    s.stem = reverse!(extract_shortest_path(s.u_stem, s.stem_top_id)) # from root → top

    # stem changed → branch-attachment distance field must be rebuilt
    s.u_branch = dijkstra_shortest_path(vb, s.stem; weighted = true)

    # recompute stem curve
    @views s.stem_curve = ArcLengthNWSmoothedCurve(vb.vox_grid[vb.coords[s.stem]], s.params.σstem;
                                                            dt_factor = s.params.dt_factor,
                                                            n_min = s.params.n_min,
                                                            k_trunc = s.params.k_trunc)

    # resize branch containers
    n_tips = length(s.tip_ids)
    resize!(s.branch, n_tips)
    resize!(s.branch_curve, n_tips)
    resize!(s.branchpoints, n_tips)
    resize!(s.branchpoints_arclength, n_tips)

    Threads.@threads for i in eachindex(s.tip_ids)
        _recompute_branch!(s, i)
    end

    # sort fruits by branchpoint arc length
    if length(s.branchpoints) > 1
        perm = sortperm(s.branchpoints_arclength)
        permute!(s.branch, perm)
        permute!(s.branch_curve, perm)
        permute!(s.tip_ids, perm)
        permute!(s.tip_points, perm)
        permute!(s.branchpoints, perm)
        permute!(s.branchpoints_arclength, perm)
    end

    return nothing
end

"""
    update_prominence!(s::ROMISkeleton, h::Real)

Update tips for the updated prominence value `h`.
"""
function update_prominence!(s::ROMISkeleton, h::Real)
    (s.params.h == h) && return nothing
    s.params.h = h

    # update local maxima
    vb = s.vb
    t_max = findlocalmaxima(s.jt_root, s.params.h)

    # filter tips on truncated faces
    filter!(m -> m.node_idx ∉ Set(vb.tf), t_max)

    # find truncated face tip (idealy only one corresponding to the top of the stem)
    t_trunc = findtruncatedtips(vb)

    # keep the existing stem top
    if !isempty(t_trunc)
        ix_tmax = findfirst(==(s.stem_top_id), map(m -> m.node_id, t_max))
        ix_ttrunc = findfirst(==(s.stem_top_id), map(m -> m.node_id, t_trunc))
        if !isnothing(ix_tmax)
            deleteat!(t_max, ix_tmax)
        elseif !isnothing(ix_ttrunc)
            deleteat!(t_trunc, ix_ttrunc)
        end
        append!(t_max, t_trunc)
    end
    s.tip_points = vb.vox_grid[map(m -> m.node_idx, t_max)]
    s.tip_ids = map(m -> m.node_id, t_max)

    # resize branch containers
    n_tips = length(s.tip_ids)
    resize!(s.branch, n_tips)
    resize!(s.branch_curve, n_tips)
    resize!(s.branchpoints, n_tips)
    resize!(s.branchpoints_arclength, n_tips)

    Threads.@threads for i in eachindex(s.tip_ids)
        _recompute_branch!(s, i)
    end

    # sort fruits by branchpoint arc length
    if length(s.branchpoints) > 1
        perm = sortperm(s.branchpoints_arclength)
        permute!(s.branch, perm)
        permute!(s.branch_curve, perm)
        permute!(s.tip_ids, perm)
        permute!(s.tip_points, perm)
        permute!(s.branchpoints, perm)
        permute!(s.branchpoints_arclength, perm)
    end

    return nothing
end

"""
    update_smoothing!(s::ROMISkeleton, σstem::Real, σbranch::Real)

Recompute smooth stem and branch using new smoothing scale.
"""
function update_smoothing!(s::ROMISkeleton, σstem::Real, σbranch::Real)
    if s.params.σstem != σstem
        # update parameters
        s.params.σstem = σstem
        s.params.σbranch = σbranch

        # recompute stem and branch curves
        @views s.stem_curve = ArcLengthNWSmoothedCurve(s.vb.vox_grid[s.vb.coords[s.stem]], s.params.σstem;
                                                                dt_factor = s.params.dt_factor,
                                                                n_min = s.params.n_min,
                                                                k_trunc = s.params.k_trunc)
        Threads.@threads for i in eachindex(s.tip_ids)
            _recompute_branch!(s, i)
        end

        # sort fruits by branchpoint arc length
        if length(s.branchpoints) > 1
            perm = sortperm(s.branchpoints_arclength)
            permute!(s.branch, perm)
            permute!(s.branch_curve, perm)
            permute!(s.tip_ids, perm)
            permute!(s.tip_points, perm)
            permute!(s.branchpoints, perm)
            permute!(s.branchpoints_arclength, perm)
        end
    elseif s.params.σbranch != σbranch
        # update parameters
        s.params.σbranch = σbranch

        # recompute branch curves
        Threads.@threads for i in eachindex(s.tip_ids)
            _recompute_branch!(s, i)
        end

        # sort fruits by branchpoint arc length
        if length(s.branchpoints) > 1
            perm = sortperm(s.branchpoints_arclength)
            permute!(s.branch, perm)
            permute!(s.branch_curve, perm)
            permute!(s.tip_ids, perm)
            permute!(s.tip_points, perm)
            permute!(s.branchpoints, perm)
            permute!(s.branchpoints_arclength, perm)
        end
    end

    return nothing
end