"""
    validate_scan_folder(path::String)

Inspects a path and returns:
  - `(:single, [path])` if it's a direct valid scan folder.
  - `(:experiment, scan_paths)` if it's a parent folder containing valid scan subfolders.
  - `(:invalid, String[])` if neither matches.
"""
function validate_scan_folder(path::String)
    !isdir(path) && return (:invalid, String[])

    # Direct single scan
    if is_valid_scan_folder(path)
        return (:single, [path])
    end

    # Experiment folder (find all valid scan subdirectories)
    subdirs = filter(isdir, readdir(path, join = true))
    valid_scans = filter(is_valid_scan_folder, subdirs)

    if !isempty(valid_scans)
        return (:experiment, valid_scans)
    end

    return (:invalid, String[])
end

"""
    run_and_load_colmap(project_dir::String;
                force_colmap::Bool = false,
                rm_colmapdb::Bool = false,
                use_GPU::Bool = true,
                xyz_error::Float64 = 3.0)

Run colmap on a separated thread and retrieve the `ROMIScan` object with pose information.
Use `force_colmap = true`, to force colmap run even if the `ROMIScan` already exist.
Use `rm_colmapdb = true`, to delete existing colmap database and force feature extraction.
Use `xyz_error` to specify the pose prior error in mm.
"""
function run_and_load_colmap(project_dir::String;
                    force_colmap::Bool = false,
                    rm_colmapdb::Bool = false,
                    use_GPU::Bool = true,
                    use_pairs::Bool = true,
                    xyz_error::Float64 = 3.0)
    abs_project_dir = abspath(project_dir)
    plant_id = basename(rstrip(abs_project_dir, ('/', '\\')))
    jls_file = joinpath(abs_project_dir, "$(plant_id)_ROMIScan.jls")
    if (!isfile(jls_file)) || force_colmap
        if isfile(jls_file)
            @info "Overwriting existing colmap outputs and $(plant_id)_ROMIScan.jls"
        end
        sparse_dir = joinpath(abs_project_dir, "sparse")
        rerun_colmap = (!isdir(sparse_dir)) || force_colmap
        (rerun_colmap && rm_colmapdb) && rm(joinpath(abs_project_dir, "colmap", "$(plant_id)_colmap_database.db"); force = true)
        pkg_env = pkgdir(@__MODULE__)
        ROMIcolmap_script = joinpath(pkg_env, "src", "ROMIcolmap.jl")
        cmd = addenv(
            `$(Base.julia_cmd()) --project=$pkg_env $ROMIcolmap_script $project_dir $rerun_colmap $use_GPU $use_pairs $xyz_error`,
            "ROMI_SKIP_WARMUP" => "true"
        )
        run(cmd)
    end
    dataset = deserialize(jls_file)
    return dataset
end

@kwdef mutable struct ROMIBboxParams
    # xyz_offset
    x_offset::Int = 0
    y_offset::Int = 0
    z_offset::Int = -70

    # xyz_span
    x_span::Int = 50
    y_span::Int = 50
    z_span::Int = 250
end
function Base.copyto!(dest::ROMIBboxParams, src::ROMIBboxParams)
    dest.x_offset = src.x_offset
    dest.y_offset = src.y_offset
    dest.z_offset = src.z_offset
    dest.x_span = src.x_span
    dest.y_span = src.y_span
    dest.z_span = src.z_span
    return dest
end
function Base.copy(p::ROMIBboxParams)
    pnew = ROMIBboxParams()
    copyto!(pnew, p)
end

"""
    pose_centroid(dataset::ROMIScan)

Compute the centroid of all camera poses.
"""
function pose_centroid(dataset::ROMIScan)
    # Calculate the 3D optical center C = -R' * t for every camera
    camera_centers = [-f.R' * f.t for f in dataset.frames]
    return Point3d(sum(camera_centers) / length(camera_centers))
end

"""
    initialize_bbox(dataset::ROMIScan; xyz_span::Vec3 = Vec3(50, 50, 250), xyz_offset::Vec3 = Vec3d(0, 0, -70))

Create a bounding box centered arround the camera centroid with xyz span given by xyz_span and offseted by xyz_offset.
"""
function initialize_bbox(dataset::ROMIScan;
                            xyz_span::Vec3 = Vec3(50, 50, 250),
                            xyz_offset::Vec3 = Vec3(0, 0, -70))
    center = pose_centroid(dataset) # camera centroid lies on top of the plant
    origin = Point3d(center[1] - xyz_span[1] / 2,
                        center[2] - xyz_span[2] / 2,
                        center[3] - xyz_span[3]) + xyz_offset
    return Rect3d(origin, xyz_span)
end
function initialize_bbox(dataset::ROMIScan, params::ROMIBboxParams)
    xyz_offset = Vec3(params.x_offset, params.y_offset, params.z_offset)
    xyz_span = Vec3(params.x_span, params.y_span, params.z_span)
    initialize_bbox(dataset; xyz_offset = xyz_offset, xyz_span = xyz_span)
end

# bounding box from 3-dimensional array (or just its dimension)
function GeometryBasics.Rect3(origin::Point3{F}, dims::NTuple{3, Int}, voxel_size::Real) where {F}
    T = promote_type(F, typeof(float(voxel_size)))
    widths = Vec3{T}(dims .* voxel_size)
    return Rect3{T}(Point3{T}(origin), widths)
end
function GeometryBasics.Rect3(origin::Point3, A::AbstractArray{3}, voxel_size::Real)
    return GeometryBasics.Rect3(origin, size(A), voxel_size)
end

"""
    clip_segment(p1, p2, xmin, xmax, ymin, ymax)

Liang-Barsky clip of segment (p1, p2) against an axis-aligned rectangle.
Returns `nothing` if the segment is entirely outside.
"""
function clip_segment(p1::Point2{T}, p2::Point2{T}, xmin, xmax, ymin, ymax) where {T}
    dx = p2[1] - p1[1]
    dy = p2[2] - p1[2]
    t0, t1 = 0.0, 1.0
    p = (-dx, dx, -dy, dy)
    q = (p1[1] - xmin, xmax - p1[1], p1[2] - ymin, ymax - p1[2])
    for i in 1:4
        if p[i] == 0
            (q[i] < 0) && return nothing
        else
            r = q[i] / p[i]
            if p[i] < 0
                (r > t1) && return nothing
                (r > t0) && (t0 = r)
            else
                (r < t0) && return nothing
                (r < t1) && (t1 = r)
            end
        end
    end
    return (Point2{T}(p1[1] + t0 * dx, p1[2] + t0 * dy), Point2{T}(p1[1] + t1 * dx, p1[2] + t1 * dy))
end

"""
    project_bbox(bbox::Rect3, frame::ROMIFrame)

Project a 3D bounding box to a 2D frame, returning a list of Tuple of Point2 to be pass to Makie.linesegments.
"""
function project_bbox(bbox::Rect3{T}, frame::ROMIFrame) where {T}
    corners_3d = coordinates(bbox)
    pix = [_point2frame(c, frame) for c in corners_3d]
    edge_indices = [
        (1, 5), (2, 6), (3, 7), (4, 8), # edges along X
        (1, 3), (2, 4), (5, 7), (6, 8), # edges along Y
        (1, 2), (3, 4), (5, 6), (7, 8)  # edges along Z
    ]
    W, H = Float64(frame.camera.width), Float64(frame.camera.height)
    segments = NTuple{2, Point2{T}}[]
    for (i, j) in edge_indices
        a, b = pix[i], pix[j]
        (a === nothing || b === nothing) && continue # edge entirely behind camera — accepted simplification
        clipped = clip_segment(a, b, 0.0, W, 0.0, H)
        (clipped === nothing) && continue
        push!(segments, clipped)
    end
    return segments
end

# Single NaN-delimited Point3f vector so all branches render in one draw call.
function _build_nan_branches(branches::Vector{Vector{Point3d}})
    pts = Point3f[]
    for b in branches
        append!(pts, Point3f.(b))
        push!(pts, Point3f(NaN32, NaN32, NaN32))
    end
    return pts
end

# object picker from a given plot
function pick_from(scene, mp, target_plot; range = 15)
    for (plt, idx) in Makie.pick_sorted(scene, mp, range)
        (plt == target_plot) && return idx
    end
    return nothing
end

# recursive layout cleaner
function clear_panel!(gl::GridLayout)
    function clear!(x)
        try
            empty!(x)
        catch
        finally
            delete!(x)
        end
    end
    function clear!(x::GridLayout)
        for el in contents(x)
            clear!(el)
        end
    end
    clear!(gl)
    Makie.trim!(gl)
    GC.gc()
    return nothing
end

mutable struct ROMIViewer
    data::ROMIScan

    # bbox
    bbox_params::ROMIBboxParams
    voxel_size::Float64

    # masks
    mf::ROMIMaskedFrames

    # volume
    vol::ROMIVolume
    msh::GeometryBasics.Mesh{3, Float32, TriangleFace{Int}}

    # skeleton
    skl::ROMISkeleton
end

# constructor
function ROMIViewer(data::ROMIScan;
                    bbox_params::ROMIBboxParams = ROMIBboxParams(),
                    mask_params::ROMIMaskParams = ROMIMaskParams(),
                    vol_params::ROMIVolumeParams = ROMIVolumeParams(bbox = initialize_bbox(data, bbox_params), λ = 30.0),
                    skel_params::ROMISkeletonParams = ROMISkeletonParams(),
                    stem_root::Union{Nothing, Int} = nothing,
                    stem_top::Union{Nothing, Int} = nothing,
                    branch_tips::Union{Nothing, Vector{ROMITipID}} = nothing,
                    use_gpu::Bool = CUDA.functional(),
                    verbose::Bool = true)
    verbose && @info "Initializing viewer data..."
    with_logger(NullLogger()) do
        # we need at least the masked frame but precomputing the volume with a 
        # smoothing will trigger compilation of the CUDA kernel!
        mf = ROMIMaskedFrames(data, mask_params)
        vol = ROMIVolume(mf, vol_params; use_gpu = use_gpu)
        msh = makemesh(vol, skel_params.t)
        skl = ROMISkeleton(vol, skel_params)
        
        # reconstruct skeleton from stem root/top and branch tips
        isnothing(stem_root) || update_stem_root!(skl, stem_root)
        isnothing(stem_top) || update_stem_top!(skl, stem_top)
        isnothing(branch_tips) || update_fruit_tips!(skl, branch_tips)

        ROMIViewer(data, bbox_params, vol_params.voxel_size, mf, vol, msh, skl)
    end
end

struct ROMIResults
    result::ROMIAnglesAndInternodes
    bbox_params::ROMIBboxParams
    mask_params::ROMIMaskParams
    vol_params::ROMIVolumeParams
    skel_params::ROMISkeletonParams

    # to reconstruct the full skeleton we need to save stem root/top and branch tips
    # this reduce the results file size compared to saving the whole ROMISkeleton instance!
    stem_root::Int
    stem_top::Int
    branch_tips::Vector{ROMITipID}
end

"""
    load_results(path::String)

Open saved results as a Dict{String, ROMIResults}. `path` should be either a direct path to a `*_ROMIAnglesAndInternodes.jls` 
file or to a ROMI experiment folder.
"""
function load_results(path::String)
    f = (isdir(path) ? joinpath(path, basename(rstrip(path, ('/', '\\')) * "_ROMIAnglesAndInternodes.jls")) : path)
    return (isfile(f) ? deserialize(f) : Dict{String, ROMIResults}())
end

"""
    export_results(path::String)

Export ROMIResults to a .csv and .json files. `path` should be either a direct path to a `*_ROMIAnglesAndInternodes.jls` 
file or to a ROMI experiment folder.
"""
function export_results(path::String)
    # load results
    res = load_results(path)
    f = (isdir(path) ? joinpath(path, basename(rstrip(path, ('/', '\\')) * "_ROMIAnglesAndInternodes.jls")) : path)
    csv_file = replace(f, ".jls" => ".csv")
    json_file = replace(f, ".jls" => ".json")

    # write csv file
    open(csv_file, "w") do io
        println(io, "plant_id,fruit_id,fruit_length (mm),polar (rad),se_polar (°),azimuth (rad),se_azimuth (°),cov_polar_azimuth (°²),flagged,internode (mm),div_angle (°),chirality")
        for plant_id in sort(eachindex(res) |> collect)
            data = res[plant_id].result
            println(io, "$plant_id,1,$(data.lengths[1]),$(data.polar[1]),$(data.se_polar[1]),$(data.azimuth[1]),$(data.se_azimuth[1]),$(data.cov_θϕ[1]),$(data.flagged[1]),,,$(data.orientation)")
            for i in 2:length(data.polar)
                println(io, ",$i,$(data.lengths[i]),$(data.polar[i]),$(data.se_polar[i]),$(data.azimuth[i]),$(data.se_azimuth[i]),$(data.cov_θϕ[i]),$(data.flagged[i]),$(data.internodes[i - 1]),$(data.div_angles[i - 1]),")
            end
        end
    end

    # write json file
    dict = Dict(plant_id => res[plant_id].result for plant_id in eachindex(res))
    open(json_file, "w") do io
        JSON.json(io, dict; pretty = true, inline_limit = 1)
    end

    println("Successfully saved:\n - $csv_file\n - $json_file")
end

mutable struct ROMIViewerState
    root_path::String
    paths::Vector{String}
    idx::Int
    rv::Union{Nothing, ROMIViewer}
    results::Dict{String, ROMIResults}
end

plant_id(path::String) = basename(rstrip(path, ('/', '\\')))

results_file(state::ROMIViewerState) = joinpath(state.root_path, basename(rstrip(state.root_path, ('/', '\\')) * "_ROMIAnglesAndInternodes.jls"))

function load_results!(state::ROMIViewerState)
    f = results_file(state)
    state.results = (isfile(f) ? deserialize(f) : Dict{String, ROMIResults}())
    return nothing
end

function commit_result!(state::ROMIViewerState, rv::ROMIViewer)
    id = plant_id(state.paths[state.idx])
    result = ROMIResults(ROMIAnglesAndInternodes(rv.skl),
        copy(rv.bbox_params),
        copy(rv.mf.params),
        copy(rv.vol.params),
        copy(rv.skl.params),
        rv.skl.vb.root_id,
        rv.skl.stem_top_id,
        copy(rv.skl.tip_ids)
    )
    state.results[id] = result
    res_file = results_file(state)
    @info "Saving to $res_file"
    serialize(res_file, state.results)
    return nothing
end

# ==========================================
# TAB 1: Bounding Box & Mask
# ==========================================
function romi_mask_and_bbox!(rv::ROMIViewer, gl::GridLayout)
    n = length(rv.data)
    (n == 0) && error("No frames in dataset")
    mask_params = copy(rv.mf.params)

    # Controls sidebar card
    right_gl = GridLayout(gl[1, 2]; tellheight = false, valign = :center)
    colsize!(gl, 2, Makie.Fixed(400))
    colgap!(gl, 15)
    rowsize!(right_gl, 1, Makie.Fixed(580))
    Box(right_gl[1, 1]; color = (:bisque, 1.0), strokecolor = :gray80, strokewidth = 1, cornerradius = 8)
    sidebar = GridLayout(right_gl[1, 1]; alignmode = Outside(15, 15, 50, 50))

    # Image preview axis
    plot_gl = GridLayout(gl[1, 1]; tellheight = false, tellwidth = false)
    ax = Makie.Axis(plot_gl[1, 1]; aspect = DataAspect(), yreversed = true, halign = :center)
    hidedecorations!(ax)
    hidespines!(ax)

    # Bounding Box Controls
    Label(sidebar[1, 1], "Bounding Box Parameters"; font = :bold, halign = :center, tellwidth = false)
    sg_bbox = SliderGrid(sidebar[2, 1],
        (label = "x origin", range = -50:50, format = "{:.0f}mm", startvalue = rv.bbox_params.x_offset),
        (label = "y origin", range = -50:50, format = "{:.0f}mm", startvalue = rv.bbox_params.y_offset),
        (label = "z origin", range = -150:0, format = "{:.0f}mm", startvalue = rv.bbox_params.z_offset),
        (label = "x width", range = 10:150, format = "{:.0f}mm", startvalue = rv.bbox_params.x_span),
        (label = "y width", range = 10:150, format = "{:.0f}mm", startvalue = rv.bbox_params.y_span),
        (label = "z width", range = 100:500, format = "{:.0f}mm", startvalue = rv.bbox_params.z_span);
        tellheight = true, tellwidth = false)
    x_off, y_off, z_off = (s.value for s in sg_bbox.sliders[1:3])
    x_sp,  y_sp,  z_sp  = (s.value for s in sg_bbox.sliders[4:6])

    # Voxel Size input
    tb_grid = GridLayout(sidebar[3, 1]; tellwidth = false)
    Label(tb_grid[1, 1], "Voxel size (mm):"; halign = :left)
    tb_vox_size = Textbox(tb_grid[1, 2]; placeholder = string(rv.voxel_size), validator = Float64, tellwidth = false)

    # Features Parameters
    Label(sidebar[4, 1], "Features Parameters", font = :bold, halign = :center, tellwidth = false)
    sg_feat = SliderGrid(sidebar[5, 1],
        (label = "gaussian blur σ", range = 0:0.01:3, format = "{:.2f}", startvalue = mask_params.σ),
        (label = (mask_params.use_line ? "half-length lh" : "brightness l"),
            range = (mask_params.use_line ? (1:5) : (0:(255 * 3))),
            format = "{:.0f}",
            startvalue = (mask_params.use_line ? mask_params.lh : round(Int, mask_params.l * 255)));
        tellheight = true, tellwidth = false)
    σ, lorlh = (s.value for s in sg_feat.sliders[1:2])

    # Checkbox & Toggle
    ft_grid = GridLayout(sidebar[6, 1]; tellwidth = false, halign = :left)
    feat_menu = Menu(ft_grid[1, 1];
        options = ["Line Enhancement", "Excess Green"],
        default = (mask_params.use_line ? "Line Enhancement" : "Excess Green"),
        height = 20)
    feat_prev = Toggle(ft_grid[1, 3]; active = false, halign = :left)
    Label(ft_grid[1, 4], "Show features"; halign = :left)
    colsize!(ft_grid, 1, Makie.Fixed(150))

    # Mask Parameters
    Label(sidebar[7, 1], "Mask Parameters", font = :bold, halign = :center, tellwidth = false)
    sg_mask = SliderGrid(sidebar[8, 1],
        (label = "threshold t", range = 0:255, format = "{:.0f}", startvalue = round(Int, mask_params.t * 255)),
        (label = "min size m", range = 0:50:10000, format = "{:.0f}", startvalue = mask_params.m),
        (label = "dilation d", range = [0; 1:2:15], format = "{:.0f}", startvalue = mask_params.d);
        tellheight = true, tellwidth = false)
    t, m, d = (s.value for s in sg_mask.sliders[1:3])

    # Toggle
    cb_grid = GridLayout(sidebar[9, 1]; tellwidth = false, halign = :left)
    mask_prev = Toggle(cb_grid[1, 1]; active = false)
    Label(cb_grid[1, 2], "Masks overlay"; halign = :left)

    # Action Buttons
    btn_grid = GridLayout(sidebar[10, 1]; tellwidth = false)
    bt_feat = Button(btn_grid[1, 1]; label = "Update Features")
    bt_mask = Button(btn_grid[1, 2]; label = "Update Masks")
    bt_vol = Button(btn_grid[1, 3]; label = "Update Volume")

    # status
    status_lbl = Label(sidebar[11, 1], ""; halign = :center, tellwidth = false, height = 10)

    # adjust row gap between label headers and sliders
    rowgap!(sidebar, 10)
    rowgap!(sidebar, 1, 5)
    rowgap!(sidebar, 4, 5)
    rowgap!(sidebar, 7, 5)

    # Observables & Logic
    idx = Observable{Int}(1)
    frame_obs = @lift rv.data.frames[$idx]
    img_obs = @lift load(joinpath(rv.data.images_dir, rv.data.images_list[$idx]))'
    
    feat_obs = @lift begin
        mask_params.σ = $σ
        if mask_params.use_line
            mask_params.lh = $lorlh
        else
            mask_params.l = $lorlh / 255
        end
        ($(feat_prev.active) ? get_feat($img_obs, mask_params) : similar(Matrix{Gray{N0f8}}, axes($img_obs)))
    end

    mask_obs = @lift begin
        mask_params.t = $t / 255
        mask_params.m = $m
        mask_params.d = $d
        if $(mask_prev.active)
            if !$(feat_prev.active)
                $feat_obs = get_feat($img_obs, mask_params)
            end
            return get_mask($feat_obs, mask_params)
        else
            return similar(BitMatrix, axes($feat_obs))
        end
    end

    bbox_obs = @lift begin
        rv.bbox_params.x_offset = $x_off
        rv.bbox_params.y_offset = $y_off
        rv.bbox_params.z_offset = $z_off
        rv.bbox_params.x_span = $x_sp
        rv.bbox_params.y_span = $y_sp
        rv.bbox_params.z_span = $z_sp

        xyz_offset = Vec3($x_off, $y_off, $z_off)
        xyz_span = Vec3($x_sp, $y_sp, $z_sp)
        initialize_bbox(rv.data; xyz_offset = xyz_offset, xyz_span = xyz_span)
    end

    segments_obs = @lift project_bbox($bbox_obs, $frame_obs)

    # Rendering
    image!(ax, img_obs)
    image!(ax, feat_obs; visible = feat_prev.active)
    image!(ax, mask_obs; alpha = 0.4, colormap = [:transparent, :red], colorrange = [0, 1], visible = mask_prev.active)
    linesegments!(ax, segments_obs; color = :yellow, linewidth = 1.5)

    # Keyboard navigation
    on(events(ax).keyboardbutton) do event
        if (event.action == Keyboard.press) || (event.action == Keyboard.repeat)
            if event.key == Keyboard.right
                idx[] = mod1(idx[] + 1, n)
            elseif event.key == Keyboard.left
                idx[] = mod1(idx[] - 1, n)
            end
        end
    end

    # Callbacks
    on(feat_menu.selection) do s
        is_line = (s == "Line Enhancement")
        mask_params.use_line = is_line

        sl = sg_feat.sliders[2]
        lbl = sg_feat.labels[2]

        if is_line
            set_close_to!(sl, mask_params.lh)
            sl.range[] = 1:5
            lbl.text[] = "half-length lh"
        else
            sl.range[] = 0:(255 * 3)
            set_close_to!(sl, round(Int, mask_params.l * 255))
            lbl.text[] = "brightness l"
        end
    end

    on(bt_feat.clicks) do _
        tf = @elapsed begin
            with_logger(NullLogger()) do
                update_maskedframes!(rv.mf, mask_params, rv.data)
            end
        end
        status_lbl.text[] = "Features updated in $(round(tf; digits = 2))s."
    end

    on(bt_mask.clicks) do _
        tm = @elapsed begin
            with_logger(NullLogger()) do
                update_maskedframes!(rv.mf, mask_params, rv.data)
            end
        end
        status_lbl.text[] = "Masks updated in $(round(tm; digits = 2))s."
    end

    on(tb_vox_size.stored_string) do s
        rv.voxel_size = parse(Float64, s)
    end

    on(bt_vol.clicks) do _
        tv = @elapsed begin
            # need to create a new ROMIVolume
            with_logger(NullLogger()) do
                vol_params = ROMIVolumeParams(bbox = initialize_bbox(rv.data, rv.bbox_params), voxel_size = rv.voxel_size)
                copyto!(vol_params, rv.vol.params) # copy existing free parameters
                rv.vol = ROMIVolume(rv.mf, vol_params)
            end
        end
        status_lbl.text[] = "$(length(rv.vol)) voxels processed in $(round(tv; digits = 2))s."
    end

    return nothing
end

# ==========================================
# TAB 2: Volume Reconstruction
# ==========================================
function romi_volume!(rv::ROMIViewer, gl::GridLayout)
    vol_params = copy(rv.vol.params)
    skel_params = copy(rv.skl.params)

    # Controls sidebar card
    right_gl = GridLayout(gl[1, 2]; tellheight = false, valign = :center)
    colsize!(gl, 2, Makie.Fixed(400))
    colgap!(gl, 15)
    rowsize!(right_gl, 1, Makie.Fixed(200))
    Box(right_gl[1, 1]; color = (:bisque, 1.0), strokecolor = :gray80, strokewidth = 1, cornerradius = 8)
    sidebar = GridLayout(right_gl[1, 1]; alignmode = Outside(15, 15, 50, 50))

    # Volume viewer axis
    ax = Axis3(gl[1, 1]; aspect = :data)

    # Smoothing parameters
    Label(sidebar[1, 1], "Volume Smoothing Parameters"; font = :bold, halign = :center, tellwidth = false)
    sg_vol = SliderGrid(sidebar[2, 1],
        (label = "smoothing penality λ", range = 0:1:100, format = "{:.0f}", startvalue = vol_params.λ);
        tellheight = true, tellwidth = false)
    bt_vol = Button(sidebar[3, 1]; label = "Apply Smoothing", tellwidth = false)

    # Volume Observable
    vol_obs = Observable(rv.vol)
    vol_max = @lift maximum($vol_obs)
    x_ci = @lift Makie.ClosedInterval(first($vol_obs.vox_grid.x), last($vol_obs.vox_grid.x))
    y_ci = @lift Makie.ClosedInterval(first($vol_obs.vox_grid.y), last($vol_obs.vox_grid.y))
    z_ci = @lift Makie.ClosedInterval(first($vol_obs.vox_grid.z), last($vol_obs.vox_grid.z))
    vol_data = @lift Array($vol_obs)

    # Isovalue
    n = length(rv.data)
    pp = vol_params.prior_prob
    tpr = vol_params.tpr
    fpr = vol_params.fpr
    min_lod = n * log((1 - tpr) / (1 - fpr)) + log(pp / (1 - pp))
    max_lod = n * log(tpr / fpr) + log(pp / (1 - pp))
    iso_range = round(min_lod, RoundUp; digits = 2):0.01:round(max_lod, RoundDown; digits = 2)
    iso_range_trunc = @lift iso_range[1:searchsortedlast(iso_range, $vol_max)]
    iso_strt = @lift clamp(skel_params.t, first($iso_range_trunc), last($iso_range_trunc))
    Label(sidebar[4, 1], "Isovalue Threshold"; font = :bold, halign = :center, tellwidth = false)
    sg_t = SliderGrid(sidebar[5, 1],
        (label = "Isovalue t", range = iso_range_trunc, format = "{:.2f}", startvalue = iso_strt);
        tellheight = true, tellwidth = false)
    bt_iso = Button(sidebar[6, 1]; label = "Update Skeleton", tellwidth = false)

    # status
    status_lbl = Label(sidebar[7, 1], ""; halign = :center, tellwidth = false, height = 10)

    # adjust row gap between label headers and sliders
    rowgap!(sidebar, 10)
    rowgap!(sidebar, 1, 5)
    rowgap!(sidebar, 4, 5)

    # Observables & Logic
    on(iso_range_trunc) do rg
        t = to_value(sg_t.sliders[1].value)
        sg_t.sliders[1].value[] = clamp(t, first(rg), last(rg))
    end

    isoval_func = lift(sg_t.sliders[1].value) do t
        skel_params.t = t
        x -> x ≤ t
    end

    on(bt_vol.clicks) do _
        λ = to_value(sg_vol.sliders[1].value)
        tv = @elapsed begin
            with_logger(NullLogger()) do
                smooth_lod!(rv.vol, rv.vol.params.τ, λ)
            end
            vol_obs[] = rv.vol
        end
        status_lbl.text[] = "$(length(rv.vol)) voxels processed in $(round(tv; digits = 2))s."
    end

    on(bt_iso.clicks) do _
        ts = @elapsed begin
            update_threshold!(rv.skl, rv.vol, skel_params.t)
        end
        status_lbl.text[] = "Skeleton updated in $(round(ts; digits = 2))s."
    end

    # Rendering
    voxels!(ax, x_ci, y_ci, z_ci, vol_data; is_air = isoval_func)

    return nothing
end

# ==========================================
# TAB 3: Skeleton Extraction
# ==========================================
function romi_skeleton!(rv::ROMIViewer, gl::GridLayout; curve_res::Real = 0.1)
    rv.msh = makemesh(rv.vol, rv.skl.params.t)
    s = rv.skl
    vb = s.vb
    skl_params = copy(rv.skl.params)

    # Info sidebar card
    right_gl = GridLayout(gl[1, 2]; tellheight = false, valign = :center)
    colsize!(gl, 2, Makie.Fixed(400))
    colgap!(gl, 15)
    rowsize!(right_gl, 1, Makie.Fixed(320))
    Box(right_gl[1, 1]; color = (:bisque, 1.0), strokecolor = :gray80, strokewidth = 1, cornerradius = 8)
    sidebar = GridLayout(right_gl[1, 1]; alignmode = Outside(15, 15, 50, 50))

    # Mesh viewer axis
    ax = Axis3(gl[1, 1]; aspect = :data, clip = false)
    hidedecorations!(ax)
    hidespines!(ax)

    # Skelton prominence and smoothing
    Label(sidebar[1, 1], "Skeleton Controls", font = :bold, halign = :center, tellwidth = false)
    sg_skl = SliderGrid(sidebar[2, 1],
        (label = "stem smoothing σstem", range = 0.5:0.1:10, format = "{:.2f}", startvalue = skl_params.σstem),
        (label = "branch smoothing σbranch", range = 0.5:0.1:5, format = "{:.2f}", startvalue = skl_params.σbranch),
        (label = "tip prominence h", range = 0:01:100, format = "{:.2f}", startvalue = skl_params.h);
        tellheight = true, tellwidth = false)
    σstem, σbranch, h = (s.value for s in sg_skl.sliders[1:3])

    # Info
    Box(sidebar[3, 1], color = :white, strokecolor = :lightgray, cornerradius = 5)
    instr_gl = GridLayout(sidebar[3, 1]; alignmode = Outside(10), tellwidth = false)
    Label(instr_gl[1, 1],
        "• Press '⎵': Add Mode\n• Press '\u232B': Delete Mode\n• Press 't': Update stem top\n• Press 'r': Update stem root\n• Press 'Esc': Exit\n• Alt + Left Click: Confirm Action", 
        halign = :left, justification = :left)
    branch_count_lbl = Label(sidebar[4, 1], "Branches: $(length(s.branchpoints))", halign = :left, font = :bold, tellwidth = false)
    mode_lbl = Label(sidebar[5, 1], "Mode: VIEWING"; halign = :right, font = :bold, color = :darkgray, tellwidth = false)

    # adjust row gap between label headers and sliders
    rowgap!(sidebar, 10)
    rowgap!(sidebar, 1, 5)

    # Observables & Logic
    mode = Observable(0) # 0: view, 1: add, 2: delete, 3: top, or 4: root
    mouse_circle = Observable(false) # to draw helper circle arround mouse
    mouse_rad = Observable(0)
    int_id = Observable(0) # intermediary point ID
    on(mode) do m
        if m == 1
            mode_lbl.text[] = "Mode: ADDING (Alt + Click to validate\n\tCtrl + Alt + Click to add stem target point)"
            mode_lbl.color[] = parse(Colorant, :dodgerblue)
            mouse_circle[] = true
            mouse_rad[] = 20
            int_id[] = 0
        elseif m == 2
            mode_lbl.text[] = "Mode: DELETING (Alt + Click to validate)"
            mode_lbl.color[] = parse(Colorant, :crimson)
            mouse_circle[] = true
            mouse_rad[] = 40
            int_id[] = 0
        elseif m == 3
            mode_lbl.text[] = "Mode: TOP (Alt + Click to validate)"
            mode_lbl.color[] = parse(Colorant, :olive)
            mouse_circle[] = true
            mouse_rad[] = 20
            int_id[] = 0
        elseif m == 4
            mode_lbl.text[] = "Mode: ROOT (Alt + Click to validate)"
            mode_lbl.color[] = parse(Colorant, :sienna)
            mouse_circle[] = true
            mouse_rad[] = 20
            int_id[] = 0
        else
            mode_lbl.text[] = "Mode: VIEWING"
            mode_lbl.color[] = parse(Colorant, :darkgray)
            mouse_circle[] = false
            mouse_rad[] = 0
            int_id[] = 0
        end
    end

    # Keyboard Listener for Toggling Modes
    on(events(ax).keyboardbutton) do ev
        if (ev.action == Keyboard.press) || (ev.action == Keyboard.repeat)
            if ev.key == Keyboard.space
                mode[] = (mode[] == 1 ? 0 : 1)
            elseif ev.key == Keyboard.backspace
                mode[] = (mode[] == 2 ? 0 : 2)
            elseif ev.key == Keyboard.t
                mode[] = (mode[] == 3 ? 0 : 3)
            elseif ev.key == Keyboard.r
                mode[] = (mode[] == 4 ? 0 : 4)
            elseif ev.key == Keyboard.escape
                mode[] = 0
            end
        end
    end

    # Render mesh geometry
    mesh!(ax, rv.msh; alpha = 0.4, color = :seagreen, transparency = true, ssao = true)

    # invisible boundary points
    boundary_coords = vcat(vb.bv, vb.tf)
    boundary_pts = Point3f.(vb.vox_grid[boundary_coords])
    boundary_plot = scatter!(ax, boundary_pts; alpha = eps(), markersize = 1, transparency = true)

    # skeleton
    stem_obs = @lift begin
        update_prominence!(s, $h)
        update_smoothing!(s, $σstem, $σbranch)
        branch_count_lbl.text[] = "Branches: $(length(s.branchpoints))"
        Point3f.(sample_uniform(s.stem_curve, curve_res))
    end
    branch_obs = @lift begin
        update_prominence!(s, $h)
        update_smoothing!(s, $σstem, $σbranch)
        branch_count_lbl.text[] = "Branches: $(length(s.branchpoints))"
        branchs = map(c -> sample_uniform(c, curve_res), s.branch_curve)
        _build_nan_branches(branchs)
    end
    tips_obs = @lift begin
        update_prominence!(s, $h)
        update_smoothing!(s, $σstem, $σbranch)
        branch_count_lbl.text[] = "Branches: $(length(s.branchpoints))"
        Point3f.(s.tip_points)
    end
    lines!(ax, stem_obs; color = :darkgreen, linewidth = 3)
    lines!(ax, branch_obs; color = :orange, linewidth = 2)
    tips_plot = scatter!(ax, tips_obs; markersize = 0.03, markerspace = :data, color = :coral)

    # Hover Callback (Updates preview line/point while mouse moves)
    node_id = Observable(0)
    preview_pt_obs = Observable(Point3f[])
    preview_line_obs = Observable(Point3f[])
    preview_color = Observable(parse(Colorant, :transparent))
    on(events(ax).mouseposition) do mp
        m = to_value(mode)
        if m == 1
            idx = pick_from(Makie.get_scene(ax), mp, boundary_plot; range = 10)
            if idx !== nothing
                node_id[] = vb.idmap[boundary_coords[idx]]
                if int_id[] == 0
                    raw = reverse!(extract_shortest_path(s.u_branch, node_id[]))
                else
                    u_int = dijkstra_shortest_path(vb, int_id[]; weighted = true)
                    raw = reverse!(extract_shortest_path(u_int, node_id[]))
                end
                preview_pt_obs[] = [boundary_pts[idx]]
                preview_line_obs[] = Point3f.(vb.vox_grid[vb.coords[raw]])
                preview_color[] = parse(Colorant, :dodgerblue)
            else
                node_id[] = 0
                preview_pt_obs[] = Point3f[]
                preview_line_obs[] = Point3f[]
                preview_color[] = parse(Colorant, :transparent)
            end
        elseif m == 2
            idx = pick_from(Makie.get_scene(ax), mp, tips_plot; range = 20)
            if idx !== nothing
                node_id[] = s.tip_ids[idx].node_id
                preview_pt_obs[] = [Point3f(s.tip_points[idx])]
                preview_line_obs[] = Point3f.(sample_uniform(s.branch_curve[idx], curve_res))
                preview_color[] = parse(Colorant, :crimson)
            else
                node_id[] = 0
                preview_pt_obs[] = Point3f[]
                preview_line_obs[] = Point3f[]
                preview_color[] = parse(Colorant, :transparent)
            end
        elseif (m == 3) || (m == 4)
            idx = pick_from(Makie.get_scene(ax), mp, boundary_plot; range = 10)
            if idx !== nothing
                node_id[] = vb.idmap[boundary_coords[idx]]
                preview_pt_obs[] = [boundary_pts[idx]]
                preview_color[] = (m == 3 ? parse(Colorant, :olive) : parse(Colorant, :sienna))
            else
                node_id[] = 0
                preview_pt_obs[] = Point3f[]
                preview_line_obs[] = Point3f[]
                preview_color[] = parse(Colorant, :transparent)
            end
        else
            node_id[] = 0
            preview_pt_obs[] = Point3f[]
            preview_line_obs[] = Point3f[]
            preview_color[] = parse(Colorant, :transparent)
        end
        return Consume(false)
    end

    # Mouse + Keyboard Listener for add/delete
    on(events(ax).mousebutton, priority = 2) do ev
        (ev.button == Mouse.left) || return Consume(false)
        if Keyboard.left_alt in events(ax).keyboardstate
            id = to_value(node_id)
            (id == 0) && return Consume(false)
            m = to_value(mode)
            if m == 1
                if Keyboard.left_control in events(ax).keyboardstate
                    int_id[] = reverse!(extract_shortest_path(s.u_branch, id))[1]
                    return Consume(true)
                else
                    add_fruit_tip!(s, ROMITipID(id, int_id[]))
                    tips_obs[] = Point3f.(s.tip_points)
                    branchs = map(c -> sample_uniform(c, curve_res), s.branch_curve)
                    branch_obs[] = _build_nan_branches(branchs)
                    branch_count_lbl.text[] = "Branches: $(length(s.branchpoints))"
                    return Consume(true)
                end
            elseif m == 2
                remove_fruit_tip!(s, id)
                tips_obs[] = Point3f.(s.tip_points)
                branchs = map(c -> sample_uniform(c, curve_res), s.branch_curve)
                branch_obs[] = _build_nan_branches(branchs)
                branch_count_lbl.text[] = "Branches: $(length(s.branchpoints))"
                return Consume(true)
            elseif m == 3
                update_stem_top!(s, id)
                stem_obs[] = Point3f.(sample_uniform(s.stem_curve, curve_res))
                tips_obs[] = Point3f.(s.tip_points)
                branchs = map(c -> sample_uniform(c, curve_res), s.branch_curve)
                branch_obs[] = _build_nan_branches(branchs)
                return Consume(true)
            elseif m == 4
                update_stem_root!(s, id)
                stem_obs[] = Point3f.(sample_uniform(s.stem_curve, curve_res))
                tips_obs[] = Point3f.(s.tip_points)
                branchs = map(c -> sample_uniform(c, curve_res), s.branch_curve)
                branch_obs[] = _build_nan_branches(branchs)
                branch_count_lbl.text[] = "Branches: $(length(s.branchpoints))"
                return Consume(true)
            end
        end
        return Consume(false)
    end

    # Hover previews
    scatter!(ax, preview_pt_obs; markersize = 0.03, markerspace = :data, color = preview_color)
    lines!(ax, preview_line_obs; color = preview_color, linewidth = 2.5, linestyle = :dash)

    # intermediary point
    int_point = @lift begin
        if $int_id == 0
            return Point3f[]
        else
            raw_point = vb.vox_grid[vb.coords[$int_id]]
            return [Point3f(raw_point)]
        end
    end
    scatter!(ax, int_point; markersize = 0.03, markerspace = :data, color = :firebrick)

    # mouse hover circle
    mouse_pos_local = @lift begin
        mp = $(events(ax).mouseposition)
        vp = $(ax.scene.viewport)
        Point2f(mp[1] - vp.origin[1], mp[2] - vp.origin[2])
    end
    scatter!(ax, mouse_pos_local;
                markersize = mouse_rad,
                marker = :circle,
                space = :pixel,
                color = (:yellow, 0.7),
                transparency = true,
                visible = mouse_circle)


    return nothing
end

# ==========================================
# Data Loading Menu
# ==========================================
function romi_data_loader!(on_start::Function, fig::Figure, gl::GridLayout)
    # State Observables
    folder_path = Observable("")
    valid_paths_ref = Observable(String[])
    status_text = Observable("Drag and drop your ROMI scan folder here.")
    is_valid_folder = Observable(false)

    # Drop Target
    Box(
        gl[1, 1],
        color = (:gray, 0.08),
        strokecolor = :gray,
        strokewidth = 2,
        cornerradius = 12,
        width = 400
    )
    drop_gl = GridLayout(gl[1, 1]; alignmode = Outside(30, 30, 30, 30), tellwidth = true)
    Label(drop_gl[1, 1], status_text; fontsize = 18, word_wrap = true, tellwidth = true)

    # Start Button
    btn = Button(
        gl[2, 1],
        label = "Start Computation",
        buttoncolor = :lightgray,
        buttoncolor_active = :lightgray,
        buttoncolor_hover = :lightgray,
        labelcolor = :darkgray,
        labelcolor_active = :darkgray,
        labelcolor_hover = :darkgray,
        tellwidth = false
    )

    # Listen for Drag & Drop Events
    on(events(fig).dropped_files) do files
        if !isempty(files)
            path = first(files)
            type, valid_paths = validate_scan_folder(path)

            if type != :invalid
                folder_path[] = path
                valid_paths_ref[] = valid_paths
                is_valid_folder[] = true
                if type == :single
                    status_text[] = "Valid single plant scan:\n$(basename(path))"
                else
                    status_text[] = "Valid experiment directory:\n$(length(valid_paths)) plant scans detected."
                end
                
                # set the button in active mode
                btn.buttoncolor = :dodgerblue
                btn.buttoncolor_active = Makie.COLOR_ACCENT[]
                btn.buttoncolor_hover = Makie.COLOR_ACCENT_DIMMED[]
                btn.labelcolor = :white
                btn.labelcolor_active = :white
                btn.labelcolor_hover = :black
            else
                folder_path[] = ""
                valid_paths_ref[] = String[]
                is_valid_folder[] = false
                status_text[] = "Invalid folder structure!"
                
                # set the button in inactive state
                btn.buttoncolor = :lightgray
                btn.buttoncolor_active = :lightgray
                btn.buttoncolor_hover = :lightgray
                btn.labelcolor = :darkgray
                btn.labelcolor_active = :darkgray
                btn.labelcolor_hover = :darkgray
            end
        end
    end

    # Handle Button Click Action
    on(btn.clicks) do _
        is_valid_folder[] && on_start(folder_path[], valid_paths_ref[])
    end

    return nothing
end

# ==========================================
# Interactive Data Extraction
# ==========================================
function romi_viewer!(start_scan!::Function, state::ROMIViewerState, fig::Figure, gl::GridLayout)
    rv = state.rv
    if rv === nothing
        Label(gl[1, 1], "Failed to load scan!", fontsize = 24)
        return nothing
    end
    n = length(state.paths)

    active_tab = Observable{Int}(1)
    save_status = Observable("")
    confirm_overwrite = Observable(false)
    next_or_save = Observable("")

    # Renders the full navigation, tab header, and active tab content
    function render_main_ui()
        clear_panel!(gl)

        # Top row: Navigation & Tab Headers
        nav_bar = GridLayout(gl[1, 1]; tellheight = true, tellwidth = false, halign = :center)
        tab_bar = GridLayout(gl[2, 1]; tellheight = true, halign = :center)

        if n > 1
            id_menu = Menu(nav_bar[1, 1];
                options = plant_id.(state.paths),
                default = plant_id(state.paths[state.idx]),
                height = 20, width = 150)
            prev_btn = Button(nav_bar[1, 2], label = "← Prev")
            Label(nav_bar[1, 3], "Plant $(state.idx) / $n: $(basename(state.paths[state.idx]))")
            next_btn = Button(nav_bar[1, 4], label = "Next →")
            
            on(id_menu.i_selected) do idx
                start_scan!(idx)
            end
            on(prev_btn.clicks) do _
                start_scan!(mod1(state.idx - 1, n))
            end
            on(next_btn.clicks) do _
                if haskey(state.results, plant_id(state.paths[state.idx]))
                    next_or_save[] = "next"
                    confirm_overwrite[] = true
                else
                    commit_result!(state, rv)
                    start_scan!(mod1(state.idx + 1, n))
                end
            end
            
            btn_save = Button(nav_bar[1, 5];
                                label = "Save",
                                buttoncolor = :limegreen,
                                buttoncolor_active = :olivedrab,
                                buttoncolor_hover = :olivedrab,
                                labelcolor = :white,
                                labelcolor_active = :white,
                                labelcolor_hover = :white)
            btn_exit = Button(nav_bar[1, 6];
                                label = "Exit",
                                buttoncolor = :crimson,
                                buttoncolor_active = :firebrick,
                                buttoncolor_hover = :firebrick,
                                labelcolor = :white,
                                labelcolor_active = :white,
                                labelcolor_hover = :white)
            Label(nav_bar[1, 7], save_status)
        else
            btn_save = Button(nav_bar[1, 1];
                                label = "Save",
                                buttoncolor = :limegreen,
                                buttoncolor_active = :olivedrab,
                                buttoncolor_hover = :olivedrab,
                                labelcolor = :white,
                                labelcolor_active = :white,
                                labelcolor_hover = :white)
            btn_exit = Button(nav_bar[1, 2];
                                label = "Exit",
                                buttoncolor = :crimson,
                                buttoncolor_active = :firebrick,
                                buttoncolor_hover = :firebrick,
                                labelcolor = :white,
                                labelcolor_active = :white,
                                labelcolor_hover = :white)
            Label(nav_bar[1, 3], save_status)
        end

        on(btn_save.clicks) do _
            if haskey(state.results, plant_id(state.paths[state.idx]))
                next_or_save[] = "save"
                confirm_overwrite[] = true
            else
                try
                    commit_result!(state, rv)
                    save_status[] = "Saved successfully!"
                catch err
                    save_status[] = "Save failed!"
                    @error "Failed to compute or commit results" exception = (err, catch_backtrace())
                end
            end
        end
        on(btn_exit.clicks) do _
            # Close the screen
            screen = Makie.getscreen(fig.scene)
            isnothing(screen) || GLMakie.GLFW.SetWindowShouldClose(screen.glscreen, true)
        end

        # Tab Navigation Buttons with active highlight
        btn_labels = ["1. Bounding Box & Mask", "2. Volume", "3. Skeleton"]
        buttons = Button[]

        for (i, label) in enumerate(btn_labels)
            b = Button(tab_bar[1, i]; 
                       label = label, 
                       buttoncolor = (i == active_tab[] ? :lightskyblue : :whitesmoke), 
                       tellwidth = false)
            push!(buttons, b)
            on(b.clicks) do _
                active_tab[] = i
            end
        end

        # Main content viewport
        panel_gl = GridLayout(gl[3, 1])

        # Load active tab view
        i = active_tab[]
        if i == 1
            romi_mask_and_bbox!(rv, panel_gl)
        elseif i == 2
            romi_volume!(rv, panel_gl)
        elseif i == 3
            romi_skeleton!(rv, panel_gl)
        end
    end

    # Tab change listener
    on(active_tab) do _
        !(confirm_overwrite[]) && render_main_ui()
    end

    # Overwrite confirmation modal view
    on(confirm_overwrite) do show_confirm
        clear_panel!(gl) # Clears all top nav bars and content views
        if show_confirm
            card_gl = GridLayout(gl[:, 1]; halign = :center, valign = :center, tellwidth = false, tellheight = false)
            Box(
                card_gl[1, 1],
                color = (:gray, 0.08),
                strokecolor = :gray,
                strokewidth = 2,
                cornerradius = 12,
                width = 550,
                tellheight = false,
                tellwidth = false
            )
            dial_gl = GridLayout(card_gl[1, 1]; alignmode = Outside(30, 30, 30, 30))
            Label(dial_gl[1, 1],
                "Plant $(state.idx) / $n ($(basename(state.paths[state.idx]))) already has saved results.\nOverwrite them with the current analysis?",
                fontsize = 18, halign = :center, justification = :center)

            btn_row = GridLayout(dial_gl[2, 1]; halign = :center, tellwidth = false)
            btn_yes = Button(btn_row[1, 1];
                            label = "Yes",
                            buttoncolor = :limegreen,
                            buttoncolor_active = :olivedrab,
                            buttoncolor_hover = :olivedrab,
                            labelcolor = :white,
                            labelcolor_active = :white,
                            labelcolor_hover = :white)
            btn_no = Button(btn_row[1, 2];
                            label = "No",
                            buttoncolor = :crimson,
                            buttoncolor_active = :firebrick,
                            buttoncolor_hover = :firebrick,
                            labelcolor = :white,
                            labelcolor_active = :white,
                            labelcolor_hover = :white)
            btn_cancel = Button(btn_row[1, 3]; label = "Cancel")

            on(btn_yes.clicks) do _
                try
                    commit_result!(state, rv)
                    if next_or_save[] == "save"
                        save_status[] = "Saved successfully!"
                    end
                catch err
                    if next_or_save[] == "save"
                        save_status[] = "Save failed!"
                    end
                    @error "Failed to compute or commit results" exception = (err, catch_backtrace())
                end
                confirm_overwrite[] = false
                (next_or_save[] == "next") && start_scan!(mod1(state.idx + 1, n))
                next_or_save[] = ""
            end
            on(btn_no.clicks) do _
                confirm_overwrite[] = false
                (next_or_save[] == "next") && start_scan!(mod1(state.idx + 1, n))
                next_or_save[] = ""
                save_status[] = ""
            end
            on(btn_cancel.clicks) do _
                confirm_overwrite[] = false
                next_or_save[] = ""
                save_status[] = ""
            end
        else
            render_main_ui()
        end
    end

    # Initial view render
    render_main_ui()

    return nothing
end

# ==========================================
# Main Application Launcher
# ==========================================
function romi_launch(; use_pairs::Bool = true, xyz_error::Float64 = 3.0)
    GLMakie.activate!(; title = "Plant Phyllotaxis Analyser")
    fig = Figure(; size = (1500, 950))
    root_gl = GridLayout(fig[1, 1])

    # app status
    status = Observable(:loading) # :loading, :colmap, :viewer, :ready
    state = ROMIViewerState("", String[], 1, nothing, Dict{String, ROMIResults}())

    function start_scan!(idx::Int)
        # Clear memory
        !(isnothing(state.rv)) && CUDA.unsafe_free!(state.rv.vol)
        state.rv = nothing
        GC.gc()
        CUDA.functional() && CUDA.reclaim()

        state.idx = idx
        path = state.paths[idx]
        status[] = :colmap
        @async begin
            # Run colmap
            colmap_task = Threads.@spawn begin
                try
                    run_and_load_colmap($path; use_pairs = use_pairs, xyz_error = xyz_error)
                catch # try to not use the GPU
                    run_and_load_colmap($path; force_colmap = true, rm_colmapdb = true, use_GPU = false)
                end
            end
            dataset = try
                fetch(colmap_task)
            catch err
                @error "Failed to load scan $path" exception = (err, catch_backtrace())
                return nothing
            end
            
            # Initialize viewer data
            status[] = :viewer
            res = state.results
            viewer_task = Threads.@spawn begin
                saved = get($res, plant_id($path), nothing)
                if saved !== nothing
                    # reopening an already-processed plant
                    ROMIViewer($dataset;
                        bbox_params = saved.bbox_params,
                        mask_params = saved.mask_params,
                        vol_params = saved.vol_params,
                        skel_params = saved.skel_params,
                        stem_root = saved.stem_root,
                        stem_top = saved.stem_top,
                        branch_tips = saved.branch_tips,
                        verbose = false)
                else
                    ROMIViewer($dataset; verbose = false)
                end
            end
            rv = try
                fetch(viewer_task)
            catch err
                @error "Failed to initialize viewer" exception = (err, catch_backtrace())
                return nothing
            end

            # Update UI to Ready state
            state.rv = rv
            status[] = :ready
        end
    end

    # Free memory on exit
    on(events(fig.scene).window_open) do is_open
        if !is_open
            if !isnothing(state.rv)
                CUDA.unsafe_free!(state.rv.vol)
                state.rv = nothing
            end
            GC.gc()
            CUDA.functional() && CUDA.reclaim()
        end
    end

    # Main logic to switch between data loading and interactive window
    on(status) do s
        clear_panel!(root_gl)
        if s == :loading
            romi_data_loader!(fig, root_gl) do root_path, valid_paths
                state.root_path = root_path
                state.paths = valid_paths
                load_results!(state)
                start_idx = findfirst(p -> !haskey(state.results, plant_id(p)), valid_paths)
                if start_idx === nothing
                    @warn "All $(length(valid_paths)) plants in this experiment already have saved results — opening the first for review."
                    start_idx = 1
                end
                start_scan!(Int(start_idx))
            end
        elseif s == :colmap
            card_gl = GridLayout(root_gl[:, 1]; halign = :center, valign = :center, tellwidth = false, tellheight = false)
            Box(
                card_gl[1, 1],
                color = (:gray, 0.08),
                strokecolor = :gray,
                strokewidth = 2,
                cornerradius = 12,
                width = 550,
                tellheight = false,
                tellwidth = false
            )
            msg_gl = GridLayout(card_gl[1, 1]; alignmode = Outside(30, 30, 30, 30))
            Label(msg_gl[1, 1],
                  "Running COLMAP for $(basename(state.paths[state.idx]))\nPlant $(state.idx) / $(length(state.paths))…",
                  fontsize = 18, halign = :center, justification = :center)
        elseif s == :viewer
            card_gl = GridLayout(root_gl[:, 1]; halign = :center, valign = :center, tellwidth = false, tellheight = false)
            Box(
                card_gl[1, 1],
                color = (:gray, 0.08),
                strokecolor = :gray,
                strokewidth = 2,
                cornerradius = 12,
                width = 550,
                tellheight = false,
                tellwidth = false
            )
            msg_gl = GridLayout(card_gl[1, 1]; alignmode = Outside(30, 30, 30, 30))
            Label(msg_gl[1, 1],
                  "Initializing viewer data for $(basename(state.paths[state.idx]))\nPlant $(state.idx) / $(length(state.paths))…",
                  fontsize = 18, halign = :center, justification = :center)
        elseif s == :ready
            romi_viewer!(start_scan!, state, fig, root_gl)
        end
    end

    notify(status)
    display(fig)
    return fig
end