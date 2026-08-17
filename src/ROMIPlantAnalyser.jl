"""
    ROMIPlantAnalyser
 
3D plant phenotyping pipeline: COLMAP-based pose recovery, masking, Bayesian voxel-carving
volume reconstruction with GMRF smoothing, skeleton extraction, and divergence-angle /
internode analysis, with an interactive GLMakie viewer.
"""
module ROMIPlantAnalyser

using GeometryBasics, JSON, Serialization, Logging
using CUDA, Adapt, StaticArrays
using LinearAlgebra, Krylov, LinearOperators
using DataStructures, MarchingCubes
using GLMakie, Images, ImageMorphology, StatsBase
using PrecompileTools: @setup_workload, @compile_workload

include("ROMITypes.jl")
include("ROMIMasks.jl")
include("ROMIVoxels.jl")
include("ROMISkeleton.jl")
include("ROMIAnglesAndInternodes.jl")
include("ROMIViewer.jl")
include("ROMISyntheticPlant.jl")

export ROMIScan, ROMIFrame, ROMICamera
export ROMIVolume, ROMISkeleton, ROMIAnglesAndInternodes
export ROMIMaskParams, ROMIVolumeParams, ROMISkeletonParams, ROMIBboxParams
export ROMIViewer, ROMIResults, run_and_load_colmap, romi_launch

# Precompile workload for CPU pipeline & GLMakie plot recipes
@setup_workload begin
    scan, vol_params, bbox_params = generate_synthetic_plant_scan(mktempdir())
    @compile_workload begin
        ROMIViewer(scan; bbox_params = bbox_params, vol_params = vol_params, verbose = false)
    end
end

# GPU warm-up: JIT-compiles CUDA kernels for runtime GPU execution
# GLMakie Shader & Screen Warm-up
function __init__()
    (get(ENV, "ROMI_SKIP_WARMUP", "false") == "true") && return nothing
    CUDA.functional() || return nothing
    try
        scan, vol_params, bbox_params = generate_synthetic_plant_scan(mktempdir())
        rv = ROMIViewer(scan; bbox_params = bbox_params, vol_params = vol_params, verbose = false)
        fig = Figure()
        gl_mask = GridLayout(fig[1, 1])
        gl_vol = GridLayout(fig[1, 2])
        gl_skl = GridLayout(fig[1, 3])
        romi_mask_and_bbox!(rv, gl_mask)
        romi_volume!(rv, gl_vol)
        romi_skeleton!(rv, gl_skl)
        Makie.colorbuffer(fig; px_per_unit = 1)
        Makie.second_resolve(fig, :gl_renderobject)
        CUDA.reclaim()
    catch e
        @warn "GPU kernel & GLMakie warm-up failed!" exception = (e, catch_backtrace())
    end
    return nothing
end
end