# shared fixtures for the test suite

# silence the @info logs of the pipeline
quiet(f) = with_logger(f, NullLogger())

# distance from p to the segment [a, b]
function segdist(p, a, b)
    ab = b - a
    t = clamp(dot(p - a, ab) / dot(ab, ab), 0, 1)
    return norm(p - (a + t * ab))
end

"""
    voxel_plant(; n, H, rs, rb, L, tilt, chirality, top_truncated, voxel_size)

Voxelize an analytic plant: a vertical stem of height `H` and radius `rs` from z = 0, with
`n` straight branches of length `L` and radius `rb` evenly spaced between 13% and 60% of `H`,
spiralling at the golden angle (counter-clockwise for `chirality = 1`, clockwise for -1) and
tilted upward by `tilt` (z component of the unnormalized direction). When `top_truncated`
is true the volume is cut 10 voxels below the stem top, like a bounding box that is too
short, otherwise the whole stem fits inside. Scaling every length and `voxel_size` by the
same factor gives exactly the same voxels.

The defaults match real ROMI reconstructions (voxel size 0.5 mm, stem radius ≈ 2.2 voxels,
fruit radius 1 voxel) so that the default `ROMISkeletonParams` (smoothing scales in mm) apply.
A fatter stem relative to `σbranch` inflates the azimuth errors well beyond real data.

Return `(V, origin, truth)` with the ground truth geometry in `truth`.
"""
function voxel_plant(; n::Int = 5, H::Real = 75.0, rs::Real = 1.1, rb::Real = 0.5, L::Real = 12.0,
                     tilt::Real = 0.3, chirality::Int = 1, top_truncated::Bool = true, voxel_size::Real = 0.5)
    golden = π * (3 - sqrt(5))
    caps = [(Vec3d(0, 0, 0), Vec3d(0, 0, H), rs)]
    heights = Float64[]
    azimuths = Float64[]
    for i in 1:n
        h = H * (0.13 + 0.47 * (i - 1) / (n - 1))
        φ = chirality * i * golden
        d = normalize(Vec3d(cos(φ), sin(φ), tilt))
        push!(caps, (Vec3d(0, 0, h), Vec3d(0, 0, h) + L * d, rb))
        push!(heights, h)
        push!(azimuths, mod2pi(φ))
    end
    # margins in voxels so that scaling every length with voxel_size gives the same voxels
    lo = Vec3d(-L - 5voxel_size, -L - 5voxel_size, -2voxel_size)
    hi = Vec3d(L + 5voxel_size, L + 5voxel_size, H + (top_truncated ? -10 : 5) * voxel_size)
    dims = Tuple(ceil.(Int, (hi - lo) ./ voxel_size))
    V = falses(dims)
    for I in CartesianIndices(V)
        p = lo + voxel_size .* (Vec3d(I.I...) .- 0.5)
        V[I] = any(c -> segdist(p, c[1], c[2]) ≤ c[3], caps)
    end
    truth = (heights = heights,
             azimuths = azimuths,
             internode = 0.47 * H / (n - 1),
             polar = acos(tilt / sqrt(1 + tilt ^ 2)),
             branch_length = L + rb, # from the stem axis to the end of the cap
             div_angle = rad2deg(golden),
             voxel_size = voxel_size)
    return V, Point3d(lo), truth
end

# skeleton of a voxel plant (copy: ROMIBinaryVolume keeps the largest component in place)
voxel_plant_skeleton(V, origin, truth; params = ROMISkeletonParams(), root = nothing) =
    quiet(() -> ROMIPlantAnalyser.ROMISkeleton(copy(V), params;
        bbox_origin = origin, voxel_size = truth.voxel_size, root = root))

"""
    synthetic_truth(; w, h, n_branches)

Ground truth geometry of `generate_synthetic_plant_scan`, recomputed from the same pixel
targets as `src/ROMISyntheticPlant.jl` (keep both in sync).
"""
function synthetic_truth(; w::Int, h::Int, n_branches::Int)
    fx = w / (2 * tan(deg2rad(53.0) / 2))
    H = 0.8 * h * 100.0 / fx
    golden = π * (3 - sqrt(5))
    heights = [H * (0.15 + 0.75 * (i - 1) / max(n_branches - 1, 1)) for i in 1:n_branches]
    return (stem_height = H,
            heights = heights,
            internode = 0.75 * H / (n_branches - 1),
            azimuths = [mod2pi(i * golden) for i in 1:n_branches],
            div_angle = rad2deg(golden))
end

# angular distance in degrees on the circle
angdist(a, b) = abs(rem(a - b, 360, RoundNearest))
