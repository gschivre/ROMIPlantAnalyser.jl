# the GPU/GLMakie warm-up is not needed (nor possible without a display) to test the API
ENV["ROMI_SKIP_WARMUP"] = "true"

using ROMIPlantAnalyser
using Test
using Logging, LinearAlgebra, Serialization
using GeometryBasics, ImageCore, JSON, PythonCall, CUDA

const R = ROMIPlantAnalyser
using ROMIPlantAnalyser: ROMITipID

include("helpers.jl")

# results conversion script (scripts/results_io.jl), in its own module
module ResultsIO
    include(joinpath(@__DIR__, "..", "scripts", "results_io.jl"))
end

@testset "ROMIPlantAnalyser.jl" begin
    include("test_geometry.jl")
    include("test_masks.jl")
    include("test_curves_angles.jl")
    include("test_skeleton.jl")
    include("test_pipeline.jl")
end
