@kwdef mutable struct ROMIVolumeParams
    # bounding box
    const bbox::Rect3d

    # voxel size in mm
    const voxel_size::Float64 = 0.5

    # prior probability of occupancy
    prior_prob::Float64 = 0.05

    # masking sensitivity (True Positive Rate)
    tpr::Float64 = 0.95

    # masking specificity (False Positive Rate)
    fpr::Float64 = 0.1

    # GMRF edge scale (soft threshold)
    τ::Float64 = 10.0

    # GMRF smoothing penalty (0 ⟹ no smoothing)
    λ::Float64 = 0.0
end
function Base.copyto!(dest::ROMIVolumeParams, src::ROMIVolumeParams)
    dest.prior_prob = src.prior_prob
    dest.tpr = src.tpr
    dest.fpr = src.fpr
    dest.τ = src.τ
    dest.λ = src.λ
    return dest
end
function Base.copy(p::ROMIVolumeParams)
    pnew = ROMIVolumeParams(bbox = p.bbox, voxel_size = p.voxel_size)
    copyto!(pnew, p)
end

"""
    point2frame(p::Point3, frame::ROMIFrame)

Projects a 3D world point into a 2D pixel space.
"""
point2frame(p::Point3, frame::ROMIFrame) = point2frame(Int, p, frame)
function point2frame(::Type{I}, p::Point3, frame::ROMIFrame) where {I <: Integer}
    # Transform to camera coordinate frame: X_c = R * X_w + t
    X_c = frame.R * p + frame.t
    
    # Check if point is behind camera focal plane
    (X_c[3] ≤ 0) && return nothing

    # correct for distortion
    cam = frame.camera
    x = X_c[1] / X_c[3]
    y = X_c[2] / X_c[3]
    r2 = x ^ 2 + y ^ 2
    radial = 1 + cam.k1 * r2 + cam.k2 * (r2 ^ 2)
    x_d = x * radial + 2 * cam.p1 * x * y + cam.p2 * (r2 + 2 * (x ^ 2))
    y_d = y * radial + cam.p1 * (r2 + 2 * (y ^ 2)) + 2 * cam.p2 * x * y

    # Projection to pixel space
    u = floor(I, cam.fx * x_d + cam.cx) + one(I)
    v = floor(I, cam.fy * y_d + cam.cy) + one(I)

    # check if point ∈ frame
    ((u < 1) || (u > cam.width) || (v < 1) || (v > cam.height)) && return nothing
    
    # y indexes the vertical axis (row) and x the horizontal axis (column)
    return CartesianIndex{2}(v, u)
end
function _point2frame(p::Point3{T}, frame::ROMIFrame) where {T <: AbstractFloat}
    # Transform to camera coordinate frame: X_c = R * X_w + t
    X_c = frame.R * p + frame.t
    
    # Check if point is behind camera focal plane
    (X_c[3] ≤ 0) && return nothing

    # correct for distortion
    cam = frame.camera
    x = X_c[1] / X_c[3]
    y = X_c[2] / X_c[3]
    r2 = x ^ 2 + y ^ 2
    radial = 1 + cam.k1 * r2 + cam.k2 * (r2 ^ 2)
    x_d = x * radial + 2 * cam.p1 * x * y + cam.p2 * (r2 + 2 * (x ^ 2))
    y_d = y * radial + cam.p1 * (r2 + 2 * (y ^ 2)) + 2 * cam.p2 * x * y

    # Pinhole projection
    return Point2{T}(cam.fx * x_d + cam.cx, cam.fy * y_d + cam.cy)
end

# to work on the GPU (isbitstype == true)
function togpu(cam::ROMICamera{I, F}) where {I, F}
    if (I == Int32) && (F == Float32)
        return cam
    else
        return ROMICamera{Int32, Float32}(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.k1, cam.k2, cam.p1, cam.p2)
    end
end
function togpu(f::ROMIFrame{I, F}) where {I, F}
    if (I == Int32) && (F == Float32)
        return f
    else
        return ROMIFrame{Int32, Float32}(Float32.(f.R), Float32.(f.t), togpu(f.camera))
    end
end

struct ROMIVoxelGrid{T} <: AbstractArray{Point3{T}, 3}
    # xyz voxel centers coordinates
    x::StepRangeLen{T, Float64, Float64, Int}
    y::StepRangeLen{T, Float64, Float64, Int}
    z::StepRangeLen{T, Float64, Float64, Int}
end
Base.size(v::ROMIVoxelGrid) = (length(v.x), length(v.y), length(v.z))
function Base.getindex(v::ROMIVoxelGrid{T}, i::Int, j::Int, k::Int) where {T}
    return Point3{T}(v.x[i], v.y[j], v.z[k])
end

# constructor from a bounding box
function ROMIVoxelGrid(bbox::Rect3{T}, voxel_size::Real) where {T}
    o, w = bbox.origin, bbox.widths
    x = range(o[1], o[1] + w[1]; step = T(voxel_size)) .+ T(0.5 * voxel_size)
    y = range(o[2], o[2] + w[2]; step = T(voxel_size)) .+ T(0.5 * voxel_size)
    z = range(o[3], o[3] + w[3]; step = T(voxel_size)) .+ T(0.5 * voxel_size)

    return ROMIVoxelGrid{T}(x, y, z)
end

# constructor from 3-dimensional array (or just its dimension)
function ROMIVoxelGrid(origin::Point3{F}, dims::NTuple{3, Int}, voxel_size::Real) where {F}
    T = promote_type(F, typeof(float(voxel_size)))
    s = T(voxel_size)
    half_s = T(0.5 * voxel_size)

    # First center coordinate for each axis
    x_start = T(origin[1]) + half_s
    y_start = T(origin[2]) + half_s
    z_start = T(origin[3]) + half_s

    # Construct exact StepRangeLens matching dimensions
    x = range(x_start; step = s, length = dims[1])
    y = range(y_start; step = s, length = dims[2])
    z = range(z_start; step = s, length = dims[3])

    return ROMIVoxelGrid{T}(x, y, z)
end
function ROMIVoxelGrid(origin::Point3, A::AbstractArray{3}, voxel_size::Real)
    return ROMIVoxelGrid(origin, size(A), voxel_size)
end

# extract a bounding box from a voxel grid
function GeometryBasics.Rect3(v::ROMIVoxelGrid{T}) where {T}
    s = step(v.x)
    o = Point3{T}(v.x[1] - 0.5 * s, v.y[1] - 0.5 * s, v.z[1] - 0.5 * s)
    w = Vec3{T}(length(v.x) * s, length(v.y) * s, length(v.z) * s)
    return Rect3{T}(o, w)
end

# Gaussian Markov Random Field (GMRF) grid for 7-point stencil Laplacian.
struct ROMIGMRFGrid{A <: AbstractArray}
    wx::A
    wy::A
    wz::A
    diagacc::A
    Lxbuf::A

    function ROMIGMRFGrid(D::A) where {A <: AbstractArray}
        dims = size(D)
        wx = similar(D, dims[1] - 1, dims[2], dims[3])
        wy = similar(D, dims[1], dims[2] - 1, dims[3])
        wz = similar(D, dims[1], dims[2], dims[3] - 1)
        diagacc = similar(D)
        Lxbuf = similar(D)
        return new{A}(wx, wy, wz, diagacc, Lxbuf)
    end
end
arraytype(::ROMIGMRFGrid{A}) where A = A

struct ROMIVolume{I <: Integer,
                F <: AbstractFloat,
                S <: AbstractVector,
                M <: AbstractArray{Bool, 3},
                G <: AbstractArray{F, 3},
                V <: AbstractVector} <: AbstractArray{F, 3}
    # CPU data to avoid repeated data transfert from the GPU
    cpu_data::Array{F, 3}
    use_gpu::Bool

    # voxel coordinates grid
    vox_grid::ROMIVoxelGrid{F}
    dims::NTuple{3, Int}

    # voxels voting log-odds
    lod::S

    # frames and corresponding masks
    frames::V
    masks::M

    # grid for GMRF volume smoothing
    gmrf_grid::ROMIGMRFGrid{G}

    # Krylov CgWorkspace used for volume smoothing
    ws::CgWorkspace{F, F, S}

    # parameters used
    params::ROMIVolumeParams

    function ROMIVolume(mf::ROMIMaskedFrames, params::ROMIVolumeParams; use_gpu::Bool = CUDA.functional())
        bbox = params.bbox
        voxel_size = params.voxel_size
        
        # use GPU if available
        if use_gpu && CUDA.functional()
            # volume carving
            vox_grid = ROMIVoxelGrid(Rect3f(bbox), voxel_size)
            @info "Volume averaging of $(length(vox_grid)) voxels using GPU"
            dims = size(vox_grid)
            n = prod(dims)
            lod = CuVector{Float32}(undef, n)
            gpu_frames = adapt(CuArray, togpu.(mf.frames))
            gpu_masks  = adapt(CuArray, mf.masks)
            f = VoxelVotingGPUCallable(gpu_frames,
                                        gpu_masks,
                                        Float32(params.prior_prob),
                                        Float32(params.tpr),
                                        Float32(params.fpr))
            map!(f, lod, vox_grid)

            # volume smoothing
            if params.λ > 0
                @info "Volume smoothing of $(length(vox_grid)) voxels using GPU"
            end
            lod3d = reshape(lod, dims)
            gmrf_grid = ROMIGMRFGrid(lod3d)
            update_weights!(Float32, gmrf_grid, lod3d, params.τ)
            A = build_system_matrix(gmrf_grid, Float32(params.λ))
            ws = Krylov.CgWorkspace(A, lod)
            Krylov.cg!(ws, A, lod, lod) # warm start at the raw data

            # transfert data from the GPU
            cpu_data = Array{Float32, 3}(undef, dims)
            copyto!(cpu_data, ws.x)

            S = typeof(lod)
            M = typeof(gpu_masks)
            G = arraytype(gmrf_grid)
            V = typeof(gpu_frames)
            return new{Int32, Float32, S, M, G, V}(
                cpu_data, use_gpu, vox_grid, dims, lod, gpu_frames, gpu_masks, gmrf_grid, ws, params)
        else
            # volume carving
            vox_grid = ROMIVoxelGrid(bbox, voxel_size)
            if params.λ > 0
                @info "Volume averaging of $(length(vox_grid)) voxels using CPU with $(Threads.nthreads()) threads"
            end
            dims = size(vox_grid)
            n = prod(dims)
            lod = Vector{Float64}(undef, n)
            prob, tpr, fpr = params.prior_prob, params.tpr, params.fpr
            Threads.@threads for i in LinearIndices(vox_grid)
                lod[i] = voxel_voting(vox_grid[i], mf.frames, mf.masks, prob, tpr, fpr)
            end

            # volume smoothing
            @info "Volume smoothing of $(length(vox_grid)) voxels using CPU"
            lod3d = reshape(lod, dims)
            gmrf_grid = ROMIGMRFGrid(reshape(lod3d, dims))
            update_weights!(Float64, gmrf_grid, lod3d, params.τ)
            A = build_system_matrix(gmrf_grid, params.λ)
            ws = Krylov.CgWorkspace(A, lod)
            Krylov.cg!(ws, A, lod, lod) # warm start at the raw data

            # copy data
            cpu_data = Array{Float64, 3}(undef, dims)
            copyto!(cpu_data, ws.x)

            G = arraytype(gmrf_grid)
            V = typeof(mf.frames)
            return new{Int, Float64, Vector{Float64}, Array{Bool, 3}, G, V}(
                cpu_data, use_gpu, vox_grid, dims, lod, mf.frames, mf.masks, gmrf_grid, ws, params)
        end
    end
end
Base.size(v::ROMIVolume) = v.dims
Base.getindex(v::ROMIVolume, I...) = v.cpu_data[I...]
Base.Array(v::ROMIVolume) = v.cpu_data

# free GPU memory
function CUDA.unsafe_free!(v::ROMIVolume)
    CUDA.functional() || return nothing
    
    # Only deallocate if data resides on GPU
    if v.use_gpu
        # handle CuArray
        CUDA.unsafe_free!(v.lod)
        CUDA.unsafe_free!(v.frames)
        CUDA.unsafe_free!(v.masks)

        # handle ROMIGMRFGrid
        CUDA.unsafe_free!(v.gmrf_grid.wx)
        CUDA.unsafe_free!(v.gmrf_grid.wy)
        CUDA.unsafe_free!(v.gmrf_grid.wz)
        CUDA.unsafe_free!(v.gmrf_grid.diagacc)
        CUDA.unsafe_free!(v.gmrf_grid.Lxbuf)

        # handle CgWorkspace
        CUDA.unsafe_free!(v.ws.Δx)
        CUDA.unsafe_free!(v.ws.x)
        CUDA.unsafe_free!(v.ws.r)
        CUDA.unsafe_free!(v.ws.npc_dir)
        CUDA.unsafe_free!(v.ws.p)
        CUDA.unsafe_free!(v.ws.Ap)
        CUDA.unsafe_free!(v.ws.z)
    end

    return nothing
end

# Updates the Welsch edge weights, exp(-(ΔD / τ) ^ 2), in-place.
function update_weights!(v::ROMIVolume{I, F, S, M, G}) where {I, F, S, M, G}
    return update_weights!(F, v.gmrf_grid, reshape(v.lod, v.dims), v.params.τ)
end
function update_weights!(::Type{T}, grid, D, τ) where {T}
    invτ2 = convert(T, inv(τ ^ 2))
    (; wx, wy, wz, diagacc) = grid
    @views begin
        @. wx = exp(-((D[1:(end - 1), :, :] - D[2:end, :, :]) ^ 2) * invτ2)
        @. wy = exp(-((D[:, 1:(end - 1), :] - D[:, 2:end, :]) ^ 2) * invτ2)
        @. wz = exp(-((D[:, :, 1:(end - 1)] - D[:, :, 2:end]) ^ 2) * invτ2)
    end
    fill!(diagacc, zero(T))
    @views begin
        diagacc[1:(end - 1), :, :] .+= wx
        diagacc[2:end, :, :] .+= wx
        diagacc[:, 1:(end - 1), :] .+= wy
        diagacc[:, 2:end, :] .+= wy
        diagacc[:, :, 1:(end - 1)] .+= wz
        diagacc[:, :, 2:end] .+= wz
    end
    return nothing
end

# Computes the matrix-vector products of L and x with L the Laplacian matrix of the Gaussian Markov Random Field (GMRF)
function apply_L!(grid::ROMIGMRFGrid, x3d)
    Lxbuf = grid.Lxbuf
    @. Lxbuf = grid.diagacc * x3d
    @views begin
        @. Lxbuf[1:(end - 1), :, :] -= grid.wx * x3d[2:end, :, :]
        @. Lxbuf[2:end, :, :] -= grid.wx * x3d[1:(end - 1), :, :]
        @. Lxbuf[:, 1:(end - 1), :] -= grid.wy * x3d[:, 2:end, :]
        @. Lxbuf[:, 2:end, :] -= grid.wy * x3d[:, 1:(end - 1), :]
        @. Lxbuf[:, :, 1:(end - 1)] -= grid.wz * x3d[:, :, 2:end]
        @. Lxbuf[:, :, 2:end] -= grid.wz * x3d[:, :, 1:(end - 1)]
    end
    return Lxbuf
end

# Creates the LinearOperator corresponding to the linear system (I + λL)x = b.
function build_system_matrix(grid::ROMIGMRFGrid, λ)
    nx, ny, nz = size(grid.Lxbuf)
    N = nx * ny * nz
    prod! = (y, x, α, β) -> begin
        x3d = reshape(x, nx, ny, nz)
        y3d = reshape(y, nx, ny, nz)
        Lx = apply_L!(grid, x3d)
        if iszero(β)
            @. y3d = α * (x3d + λ * Lx)
        else
            @. y3d = α * (x3d + λ * Lx) + β * y3d
        end
    end
    T = eltype(grid.Lxbuf)
    S = (isa(grid.Lxbuf, CuArray) ? CuVector{T} : Vector{T})
    return LinearOperator{T, S}(N, N, true, true, prod!, nothing, nothing)
end

"""
    voxel_voting(p::Point3{T}, frames::ROMIFrame, masks::AbstractArray{Bool, 3}, prob, tpr, fpr)

Compute the log-odds of occupancy using Bayesian space carving.
`prob`, `tpr`, and `fpr` are respectively the prior probability of occupancy and the object detection sensitivity and specificity.
"""
function voxel_voting(p::Point3, frames::AbstractVector, masks::AbstractArray{Bool, 3}, prob, tpr, fpr)
    lod = log(prob / (1 - prob)) # log-odds
    for i in eachindex(frames)
        # project p to the current frame
        ix = point2frame(p, frames[i])
        (ix === nothing) && continue

        # check if p fall in mask
        lod += (masks[ix, i] ? log(tpr / fpr) : log((1 - tpr) / (1 - fpr)))
    end
    return lod
end

"""
    VoxelVotingGPUCallable(frames, masks, prob, tpr, fpr)
 
Callable wrapping the voxel-voting reduction for use with `map!` over a `ROMIVoxelGrid` on the GPU.
"""
struct VoxelVotingGPUCallable{V <: AbstractVector, M <: AbstractArray{Bool, 3}, F <: AbstractFloat}
    frames::V
    masks::M
    prob::F
    tpr::F
    fpr::F
end
(f::VoxelVotingGPUCallable)(p::Point3) = voxel_voting(p, f.frames, f.masks, f.prob, f.tpr, f.fpr)
Adapt.@adapt_structure VoxelVotingGPUCallable

"""
   compute_lod!(v::ROMIVolume)

Given a bounding box, voxel size, and masks compute the log-odds of occupancy from Bayesian space carving using CPU or GPU threads.
"""
function compute_lod!(v::ROMIVolume)
    if v.use_gpu
        @info "Volume averaging of $(length(v.vox_grid)) voxels using GPU"
        f = VoxelVotingGPUCallable(v.frames,
                                    v.masks,
                                    Float32(v.params.prior_prob),
                                    Float32(v.params.tpr), 
                                    Float32(v.params.fpr))
        map!(f, v.lod, v.vox_grid)
    else
        @info "Volume averaging of $(length(v.vox_grid)) voxels using CPU with $(Threads.nthreads()) threads"
        prob, tpr, fpr = v.params.prior_prob, v.params.tpr, v.params.fpr
        Threads.@threads for i in LinearIndices(v.vox_grid)
            v.lod[i] = voxel_voting(v.vox_grid[i], v.frames, v.masks, prob, tpr, fpr)
        end
    end
    update_weights!(v)
    copyto!(v.cpu_data, v.lod) # raw data as output
end
function compute_lod!(v::ROMIVolume, prior_prob::Real, tpr::Real, fpr::Real)
    all((v.params.prior_prob == prior_prob, v.params.tpr == tpr, v.params.fpr == fpr)) && return v.cpu_data
    v.params.prior_prob = prior_prob
    v.params.tpr = tpr
    v.params.fpr = fpr
    return compute_lod!(v::ROMIVolume)
end

"""
   smooth_lod!(v::ROMIVolume)

Smooth the 3D occupancy volume `v` in-place by computing the Maximum A Posteriori (MAP) 
estimate of a Gaussian Markov Random Field (GMRF).
"""
function smooth_lod!(v::ROMIVolume)
    if v.use_gpu
        @info "Volume smoothing of $(length(v.vox_grid)) voxels using GPU"
        A = build_system_matrix(v.gmrf_grid, Float32(v.params.λ))
        b = v.lod
        Krylov.cg!(v.ws, A, b, b) # warm start at the raw data
    else
        @info "Volume smoothing of $(length(v.vox_grid)) voxels using CPU with $(Threads.nthreads()) threads"
        A = build_system_matrix(v.gmrf_grid, v.params.λ)
        b = v.lod
        Krylov.cg!(v.ws, A, b, b) # warm start at the raw data
    end
    copyto!(v.cpu_data, v.ws.x)
end
function smooth_lod!(v::ROMIVolume, τ::Real, λ::Real)
    ((v.params.τ == τ) && (v.params.λ == λ)) && return v.cpu_data
    v.params.τ = τ
    v.params.λ = λ
    update_weights!(v)
    return smooth_lod!(v)
end

"""
    makemesh(v::ROMIVolume, t::Real)

Create a mesh from a volume by marching cubes with iso value `t`.
"""
function makemesh(v::ROMIVolume, t::Real)
    mc = MarchingCubes.MC(Array(v); x = v.vox_grid.x, y = v.vox_grid.y, z = v.vox_grid.z)
    MarchingCubes.march(mc, t)
    return MarchingCubes.makemesh(GeometryBasics, mc)
end