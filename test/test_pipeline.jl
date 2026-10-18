@testset "Pipeline (synthetic scan)" begin
    # an experiment folder with one rendered plant (64×128 px, 24 views)
    w, h, n_frames, n_branches = 64, 128, 24, 4
    exp_dir = mktempdir()
    scan_dir = joinpath(exp_dir, "plant_A")
    scan, vol_params, bbox_params = quiet(() -> R.generate_synthetic_plant_scan(scan_dir; w, h, n_frames, n_branches))
    truth = synthetic_truth(; w, h, n_branches)
    centers = [Point3d(-f.R' * f.t) for f in scan.frames]

    @testset "Scan folder" begin
        @test R.is_valid_scan_folder(scan_dir)
        @test length(scan) == n_frames
        @test scan.plant_id == "plant_A"
        @test scan.images_list == [lpad(i, 5, '0') * "_rgb.jpg" for i in 0:(n_frames - 1)]
        # CNC poses read from the metadata are the camera centers
        @test all(scan.cnc_xyz .≈ centers)
        @test isfile(scan.poses_path) && length(readlines(scan.poses_path)) == n_frames

        # folder detection: single scan, experiment, invalid
        @test R.validate_scan_folder(scan_dir) == (:single, [scan_dir])
        @test R.validate_scan_folder(exp_dir) == (:experiment, [scan_dir])
        @test R.validate_scan_folder(mktempdir())[1] == :invalid
        @test R.validate_scan_folder(joinpath(exp_dir, "missing"))[1] == :invalid
        broken_dir = mktempdir()
        cp(scan_dir, joinpath(broken_dir, "plant_B"))
        rm(joinpath(broken_dir, "plant_B", "metadata", "images", "00003_rgb.json"))
        @test !R.is_valid_scan_folder(joinpath(broken_dir, "plant_B")) # an image without metadata
        @test_throws AssertionError ROMIScan(joinpath(broken_dir, "plant_B"))
    end

    @testset "Camera poses from plant-3d-vision" begin
        # write the Colmap task outputs the way plant-3d-vision does, then read them back
        romi_dir = mktempdir()
        romi_scan = joinpath(romi_dir, "plant_R")
        cp(scan_dir, romi_scan)
        colmap_dir = mkpath(joinpath(romi_scan, "Colmap_1_abcdef"))
        c = scan.frames[1].camera
        JSON.json(joinpath(colmap_dir, "cameras.json"),
            Dict("1" => Dict("width" => c.width, "height" => c.height, "model" => "OPENCV",
                             "params" => [c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.p1, c.p2])))
        JSON.json(joinpath(colmap_dir, "images.json"),
            Dict(img => Dict("rotmat" => [collect(f.R[i, :]) for i in 1:3], "tvec" => collect(f.t))
                 for (img, f) in zip(scan.images_list, scan.frames)))

        loaded = quiet(() -> run_and_load_colmap(romi_scan; from_romi = true))
        @test all(f.R ≈ g.R && f.t ≈ g.t && f.camera == g.camera for (f, g) in zip(loaded.frames, scan.frames))
        jls = joinpath(romi_scan, "plant_R_ROMIScan.jls")
        @test isfile(jls)

        # once the poses are cached, COLMAP is not needed anymore
        cached = quiet(() -> run_and_load_colmap(romi_scan))
        @test all(f.R ≈ g.R && f.t ≈ g.t for (f, g) in zip(cached.frames, scan.frames))
    end

    @testset "Masks" begin
        mf = quiet(() -> R.ROMIMaskedFrames(scan, ROMIMaskParams()))
        @test size(mf) == (h, w, n_frames)
        # the stem axis projects inside the mask in every view (renderer and projection agree)
        axis = [Point3d(0, 0, z) for z in range(0.1, 0.9; length = 20) .* truth.stem_height]
        @test all(mf.masks[R.point2frame(p, f), i] for (i, f) in enumerate(scan.frames) for p in axis)
        # the plant only covers a small part of the images
        @test 0.01 < sum(mf.masks) / length(mf.masks) < 0.3

        # changing only the threshold recomputes the masks but not the features
        feats = copy(mf.features)
        p = ROMIMaskParams(t = 40 / 255)
        quiet(() -> R.update_maskedframes!(mf, p, scan))
        @test mf.features == feats
        @test mf.params.t == p.t
        @test sum(mf.masks) < sum(quiet(() -> R.ROMIMaskedFrames(scan, ROMIMaskParams())).masks)
    end

    @testset "Volume carving" begin
        mf = quiet(() -> R.ROMIMaskedFrames(scan, ROMIMaskParams()))
        vp = ROMIVolumeParams(bbox = vol_params.bbox, voxel_size = vol_params.voxel_size, λ = 0.0)
        vol = quiet(() -> R.ROMIVolume(mf, vp; use_gpu = false))
        grid = vol.vox_grid
        @test size(vol) == size(grid)

        # a point on the stem axis is seen in the mask by every camera
        p = Point3d(0, 0, truth.stem_height / 2)
        expected = log(vp.prior_prob / (1 - vp.prior_prob)) + n_frames * log(vp.tpr / vp.fpr)
        @test R.voxel_voting(p, mf.frames, mf.masks, vp.prior_prob, vp.tpr, vp.fpr) ≈ expected
        # the voxel holding it is occupied, the bounding box corners are empty
        i = CartesianIndex(argmin(abs.(grid.x)), argmin(abs.(grid.y)), argmin(abs.(grid.z .- p[3])))
        @test vol[i] > 0
        @test all(vol[c...] < 0 for c in Iterators.product((1, size(vol, 1)), (1, size(vol, 2)), (1, size(vol, 3))))

        # GMRF smoothing keeps the inside/outside decision on the stem and changes the values
        raw = copy(Array(vol))
        quiet(() -> R.smooth_lod!(vol, vp.τ, 30.0))
        @test vol.params.λ == 30.0
        @test vol[i] > 0
        @test Array(vol) != raw
        quiet(() -> R.smooth_lod!(vol, vp.τ, 0.0)) # λ = 0 gives back the raw votes
        @test Array(vol) ≈ raw

        msh = R.makemesh(vol, 0.0)
        @test !isempty(R.GeometryBasics.coordinates(msh))

        q = copy(vp)
        @test q !== vp && q.bbox == vp.bbox && q.λ == vp.λ

        if CUDA.functional()
            # the GPU kernel gives the same occupancy as the CPU one
            vg = quiet(() -> R.ROMIVolume(mf, vp; use_gpu = true))
            @test sum(sign.(Array(vg)) .== sign.(raw)) / length(raw) > 0.99
            CUDA.unsafe_free!(vg)
        else
            @info "CUDA not functional: GPU volume test skipped"
        end
    end

    @testset "Full pipeline" begin
        rv = quiet(() -> ROMIViewer(scan; bbox_params = bbox_params, vol_params = vol_params, use_gpu = false, verbose = false))
        s = rv.skl
        a = ROMIAnglesAndInternodes(s)
        @test length(s.branchpoints) == n_branches
        @test norm(s.stem_top[1:2]) < 3 && abs(s.stem_top[3] - truth.stem_height) < 10
        @test norm(s.stem_root[1:2]) < 3 && abs(s.stem_root[3]) < 10
        @test all(abs.(a.internodes .- truth.internode) .< 3)
        @test a.orientation == :ccw
        @test all(angdist.(rad2deg.(a.azimuth), rad2deg.(truth.azimuths)) .< 30)
        @test all(d -> angdist(d, truth.div_angle) < 35, a.div_angles)

        # save in the experiment folder the way the app does
        state = R.ROMIViewerState(exp_dir, [scan_dir], 1, rv, Dict{String, ROMIResults}())
        quiet(() -> R.commit_result!(state, rv))
        res_file = joinpath(exp_dir, basename(exp_dir) * "_ROMIAnglesAndInternodes.jls")
        @test isfile(res_file)
        res = load_results(exp_dir)
        @test collect(keys(res)) == ["plant_A"]
        saved = res["plant_A"]
        @test saved.result.div_angles ≈ a.div_angles
        @test !R.is_dummy_volparams(saved.vol_params)
        @test load_results(res_file)["plant_A"].stem_top == s.stem_top_id
        @test isempty(load_results(mktempdir()))

        @testset "Reopen" begin
            rv2 = quiet(() -> ROMIViewer(scan;
                bbox_params = saved.bbox_params,
                mask_params = saved.mask_params,
                vol_params = saved.vol_params,
                skel_params = saved.skel_params,
                stem_root = saved.stem_root,
                stem_top = saved.stem_top,
                branch_tips = saved.branch_tips,
                use_gpu = false, verbose = false))
            a2 = ROMIAnglesAndInternodes(rv2.skl)
            @test rv2.skl.tip_ids == s.tip_ids
            @test a2.internodes ≈ a.internodes
            @test a2.div_angles ≈ a.div_angles
        end

        @testset "Results saved by earlier versions" begin
            # test/data/legacy_results.jls was written for this same synthetic scan by the version
            # before the emergence offset (fc8d79e): old files must keep loading and be updatable
            legacy = load_results(joinpath(@__DIR__, "data", "legacy_results.jls"))
            old = legacy["plant_A"]
            @test old isa ROMIResults
            @test length(old.result.azimuth) == n_branches
            rv3 = quiet(() -> ROMIViewer(scan;
                bbox_params = old.bbox_params, mask_params = old.mask_params, vol_params = old.vol_params,
                skel_params = old.skel_params, stem_root = old.stem_root, stem_top = old.stem_top,
                branch_tips = old.branch_tips, use_gpu = false, verbose = false))
            @test rv3.skl.tip_ids == old.branch_tips
            # same skeleton: offset = 0 gives back the old numbers, the default updates the angles only
            a_old = ROMIAnglesAndInternodes(rv3.skl; offset = 0)
            @test a_old.azimuth ≈ old.result.azimuth
            @test a_old.polar ≈ old.result.polar
            a_new = ROMIAnglesAndInternodes(rv3.skl)
            @test a_new.internodes ≈ old.result.internodes
            @test a_new.lengths ≈ old.result.lengths
            # and the updated azimuths are closer to the truth
            @test a_new.azimuth != old.result.azimuth
            err_new = angdist.(rad2deg.(a_new.azimuth), rad2deg.(truth.azimuths))
            err_old = angdist.(rad2deg.(old.result.azimuth), rad2deg.(truth.azimuths))
            @test all(err_new .≤ err_old .+ 1) && sum(err_new) < sum(err_old)
            # saving over an old file updates it in place
            upd_dir = mktempdir()
            upd_file = joinpath(upd_dir, basename(upd_dir) * "_ROMIAnglesAndInternodes.jls")
            cp(joinpath(@__DIR__, "data", "legacy_results.jls"), upd_file)
            st = R.ROMIViewerState(upd_dir, [scan_dir], 1, rv3, Dict{String, ROMIResults}())
            R.load_results!(st)
            quiet(() -> R.commit_result!(st, rv3))
            @test load_results(upd_dir)["plant_A"].result.azimuth ≈ a_new.azimuth
            CUDA.unsafe_free!(rv3.vol)
        end

        @testset "Export" begin
            quiet(() -> redirect_stdout(devnull) do
                export_results(exp_dir)
            end)
            csv = replace(res_file, ".jls" => ".csv")
            lines = readlines(csv)
            @test startswith(lines[1], "plant_id,fruit_id,fruit_length (mm)")
            @test length(lines) == 1 + n_branches
            header = split(lines[1], ',')
            rows = split.(lines[2:end], ',')
            @test all(length.(rows) .== length(header))
            @test rows[1][1] == "plant_A" && all(r -> r[1] == "", rows[2:end])
            @test parse.(Int, getindex.(rows, 2)) == 1:n_branches
            @test parse.(Float64, getindex.(rows[2:end], 11)) ≈ a.div_angles
            @test rows[1][end] == "ccw"

            js = JSON.parsefile(replace(res_file, ".jls" => ".json"))
            @test collect(keys(js)) == ["plant_A"]
            @test Float64.(js["plant_A"]["internodes"]) ≈ a.internodes

            @test_throws AssertionError export_results(mktempdir())
        end
        CUDA.unsafe_free!(rv.vol)
    end

    @testset "Skeleton editing from plant-3d-vision voxels" begin
        # write a Voxels task output (binary volume from space carving) next to the scan
        V, origin, vtruth = voxel_plant()
        ske_dir = mktempdir()
        ske_scan = joinpath(ske_dir, "plant_V")
        cp(scan_dir, ske_scan)
        widths = vtruth.voxel_size .* size(V)
        bb = Dict("x" => [origin[1], origin[1] + widths[1]], "y" => [origin[2], origin[2] + widths[2]],
                  "z" => [origin[3] + widths[3], origin[3]]) # unsorted on purpose
        JSON.json(joinpath(ske_scan, "metadata", "Voxels_1_abcdef.json"),
            Dict("task_params" => Dict("bounding_box" => bb, "voxel_size" => vtruth.voxel_size, "type" => "carving")))
        vox_dir = mkpath(joinpath(ske_scan, "Voxels_1_abcdef"))
        iio = PythonCall.pyimport("imageio.v3")
        np = PythonCall.pyimport("numpy")
        iio.imwrite(joinpath(vox_dir, "Voxels.tiff"), np.asarray(UInt8.(V)))

        @test R.get_voxels_metadata(ske_scan) == ("carving", n_frames - 2)
        bbox, vs = R.get_romi_grid_params(ske_scan)
        @test bbox.origin ≈ origin && bbox.widths ≈ Vec3d(widths...) && vs == vtruth.voxel_size
        @test R.get_binary_voxel(ske_scan, "carving", 0) == V

        se = quiet(() -> R.ROMISkeletonEdit(ske_scan; verbose = false))
        direct = voxel_plant_skeleton(V, origin, vtruth)
        @test length(se.skl.branchpoints) == length(vtruth.heights)
        @test se.skl.branchpoints_arclength ≈ direct.branchpoints_arclength
        @test !isempty(R.GeometryBasics.coordinates(se.msh))

        # results saved from this app carry dummy volume parameters…
        state = R.ROMISkeletonEditState(ske_dir, [ske_scan], 1, se, Dict{String, ROMIResults}())
        quiet(() -> R.commit_result!(state, se))
        saved = load_results(ske_dir)["plant_V"]
        @test R.is_dummy_volparams(saved.vol_params)
        @test saved.stem_top == se.skl.stem_top_id
        # …and can be overwritten from the same app
        quiet(() -> R.commit_result!(state, se))
        # but results from the full pipeline are never overwritten
        full = load_results(exp_dir)["plant_A"]
        state.results["plant_V"] = full
        @test_throws ErrorException quiet(() -> R.commit_result!(state, se))
        @test R.is_dummy_volparams(load_results(ske_dir)["plant_V"].vol_params) # file untouched
    end
end
