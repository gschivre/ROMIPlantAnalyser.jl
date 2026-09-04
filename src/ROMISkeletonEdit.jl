# provide methods to run the skeleton tuning out of ROMI reconstruction 
# this will allow to outsource heavy computations outside of the app!

# recover bounding box and voxel size from ROMI results
function get_romi_grid_params(path::String)
    # locate the Voxels_*.json file in the metadata
    metadata_dir = joinpath(path, "metadata")
    vox_json = JSON.parsefile(joinpath(metadata_dir, first(filter(f -> occursin(r"^Voxels_.*\.json$", f), readdir(metadata_dir)))))

    # extract bbox coordinates from the .json
    x_start, x_end = sort(Float64.(vox_json["task_params"]["bounding_box"]["x"]))
    y_start, y_end = sort(Float64.(vox_json["task_params"]["bounding_box"]["y"]))
    z_start, z_end = sort(Float64.(vox_json["task_params"]["bounding_box"]["z"]))
    bbox_start = Point3d(x_start, y_start, z_start)
    bbox_end = Point3d(x_end, y_end, z_end)
    bbox = Rect3d(bbox_start, bbox_end - bbox_start)

    # get the voxel size
    vox_size = Float64(vox_json["task_params"]["voxel_size"])

    return (bbox, vox_size)
end

# recover thresholded volume from ROMI results
function get_binary_voxel(path::String)
    # locate the Voxels.tiff file
    voxels_dir = joinpath(path, first(filter!(startswith("Voxels_"), readdir(path))))
    vol = permutedims(Float64.(load(joinpath(voxels_dir, "Voxels.tiff"); verbose = false)), (1, 3, 2))

    # loading the tifffile.py produced file in julia results in wrong numeric value for -1 and +1 values!
    Threads.@threads for i in eachindex(vol)
        vol[i] = sign(vol[i])
    end

    return vol
end

mutable struct ROMISkeletonEdit
    # mesh
    bbox::Rect3d
    vox_size::Float64
    msh::GeometryBasics.Mesh{3, Float32, TriangleFace{Int}}

    # skeleton
    skl::ROMISkeleton
end

# constructor
function ROMISkeletonEdit(path::String;
                    skel_params::ROMISkeletonParams = ROMISkeletonParams(; t = 0.0),
                    stem_root::Union{Nothing, Int} = nothing,
                    stem_top::Union{Nothing, Int} = nothing,
                    branch_tips::Union{Nothing, Vector{ROMITipID}} = nothing,
                    verbose::Bool = true)
    if verbose 
        @info "Initializing viewer data..."
        (skel_params.t != 0.0) && @warn "ROMI plant-3d-vision output binary voxels with negative value for background\n\tusing 0 as a default threshold"
        skel_params.t = 0.0
    end
    with_logger(NullLogger()) do
        bbox, vox_size = get_romi_grid_params(path)
        vox_grid = ROMIVoxelGrid(bbox, vox_size)
        vol = get_binary_voxel(path)
        mc = MarchingCubes.MC(vol; x = vox_grid.x, y = vox_grid.y, z = vox_grid.z)
        MarchingCubes.march(mc, skel_params.t)
        msh = MarchingCubes.makemesh(GeometryBasics, mc)
        skl = ROMISkeleton(vol .> skel_params.t, skel_params;
                bbox_origin = Point3d(bbox.origin), voxel_size = vox_size, root = stem_root)
        
        # reconstruct skeleton from stem root/top and branch tips
        isnothing(stem_top) || update_stem_top!(skl, stem_top)
        isnothing(branch_tips) || update_fruit_tips!(skl, branch_tips)

        ROMISkeletonEdit(bbox, vox_size, msh, skl)
    end
end

# create a new state instance to use multiple dispatch
mutable struct ROMISkeletonEditState
    root_path::String
    paths::Vector{String}
    idx::Int
    rv::Union{Nothing, ROMISkeletonEdit}
    results::Dict{String, ROMIResults}
end

results_file(state::ROMISkeletonEditState) = joinpath(state.root_path, basename(rstrip(state.root_path, ('/', '\\')) * "_ROMIAnglesAndInternodes.jls"))

function load_results!(state::ROMISkeletonEditState)
    f = results_file(state)
    state.results = (isfile(f) ? deserialize(f) : Dict{String, ROMIResults}())
    return nothing
end

function commit_result!(state::ROMISkeletonEditState, rv::ROMISkeletonEdit)
    id = plant_id(state.paths[state.idx])

    # use dummy parameters for bbox, mask and volume
    result = ROMIResults(ROMIAnglesAndInternodes(rv.skl),
        ROMIBboxParams(0, 0, 0, 0, 0, 0),
        ROMIMaskParams(false, 0.0, 0, 0.0, 0.0, 0, 0),
        ROMIVolumeParams(rv.bbox, rv.vox_size, 0.0, 0.0, 0.0, 0.0, 0.0),
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
# TAB 3: Skeleton Extraction
# ==========================================
function romi_skeleton!(rv::ROMISkeletonEdit, gl::GridLayout; curve_res::Real = 0.1)
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
# Interactive Data Extraction
# ==========================================
function romi_viewer!(start_scan!::Function, state::ROMISkeletonEditState, fig::Figure, gl::GridLayout)
    rv = state.rv
    if rv === nothing
        Label(gl[1, 1], "Failed to load scan!", fontsize = 24)
        return nothing
    end
    n = length(state.paths)

    save_status = Observable("")
    confirm_overwrite = Observable(false)
    next_or_save = Observable("")

    # Renders the full navigation, tab header, and active tab content
    function render_main_ui()
        clear_panel!(gl)

        # Top row: Navigation
        nav_bar = GridLayout(gl[1, 1]; tellheight = true, tellwidth = false, halign = :center)

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

        # Main content viewport
        panel_gl = GridLayout(gl[2, 1])
        romi_skeleton!(rv, panel_gl)
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
function romi_launch_skeledit()
    GLMakie.activate!(; title = "Plant Phyllotaxis Analyser")
    fig = Figure(; size = (1500, 950))
    root_gl = GridLayout(fig[1, 1])

    # app status
    status = Observable(:loading) # :loading, :viewer, :ready
    state = ROMISkeletonEditState("", String[], 1, nothing, Dict{String, ROMIResults}())

    function start_scan!(idx::Int)
        # Clear memory
        state.rv = nothing
        GC.gc()

        state.idx = idx
        path = state.paths[idx]
        status[] = :viewer
        @async begin
            # Initialize viewer data
            res = state.results
            viewer_task = Threads.@spawn begin
                saved = get($res, plant_id($path), nothing)
                if saved !== nothing
                    # reopening an already-processed plant
                    ROMISkeletonEdit($path;
                        skel_params = saved.skel_params,
                        stem_root = saved.stem_root,
                        stem_top = saved.stem_top,
                        branch_tips = saved.branch_tips,
                        verbose = false)
                else
                    ROMISkeletonEdit($path; verbose = false)
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
                state.rv = nothing
            end
            GC.gc()
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