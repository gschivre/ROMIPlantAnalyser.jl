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
using GLMakie, Images, ImageMorphology
using StatsBase
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
export ROMIViewer, run_and_load_colmap, romi_launch

# Precompile workload for CPU pipeline & GLMakie plot recipes
@setup_workload begin
    scan, vol_params, bbox_params = generate_synthetic_plant_scan(mktempdir())
    fig = Figure(; size = (1500, 950))
    gl = GridLayout(fig[1, 1])

    @compile_workload begin
        rv = ROMIViewer(scan; bbox_params = bbox_params, vol_params = vol_params)
        romi_mask_and_bbox!(rv, gl)
        romi_volume!(rv, gl)
        romi_skeleton!(rv, gl)
    end
end

# GPU warm-up: JIT-compiles CUDA kernels for runtime GPU execution
function __init__()
    CUDA.functional() || return nothing
    try
        scan, vol_params, bbox_params = generate_synthetic_plant_scan(mktempdir())
        ROMIViewer(scan; bbox_params = bbox_params, vol_params = vol_params)
    catch e
        @warn "GPU kernel warm-up failed!" exception = (e, catch_backtrace())
    end
    return nothing
end
end