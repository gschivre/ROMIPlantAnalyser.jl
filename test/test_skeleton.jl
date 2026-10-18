@testset "Skeleton (analytic voxel plant)" begin
    V, origin, truth = voxel_plant() # stem top cut by the volume, like a short bounding box
    vs = truth.voxel_size
    s = voxel_plant_skeleton(V, origin, truth)
    a = ROMIAnglesAndInternodes(s)
    n = length(truth.heights)
    ids = [t.node_id for t in s.tip_ids]
    bps = copy(s.branchpoints_arclength)
    zof(s, i) = s.vb.vox_grid[s.vb.coords[i]][3]

    @testset "Topology" begin
        @test length(s.branchpoints) == n
        @test length(s.branch_curve) == n == length(s.tip_ids)
        @test issorted(s.branchpoints_arclength) # fruits ordered from the root up
        @test norm(s.stem_root[1:2]) < 2vs && s.stem_root[3] < 2vs # root at the stem base
        @test norm(s.stem_top[1:2]) < 2vs && s.stem_top[3] > maximum(truth.heights) # top on the stem
    end

    @testset "Measurements" begin
        # internodes: exact up to the voxel size
        @test all(abs.(a.internodes .- truth.internode) .≤ vs)
        # branch points: same spacing as the truth (absolute offset = root position + junction bias)
        offset = s.branchpoints_arclength .- truth.heights
        @test maximum(offset) - minimum(offset) ≤ vs
        @test all(abs.(a.lengths .- truth.branch_length) .≤ 2vs)
        @test a.orientation == :ccw
        @test !any(a.flagged)
        # azimuths within the accuracy found against manual measurements (median 6–17° per stem)
        @test all(angdist.(rad2deg.(a.azimuth), rad2deg.(truth.azimuths)) .< 20)
        @test sum(a.div_angles) / length(a.div_angles) ≈ truth.div_angle atol = 10
        @test all(d -> angdist(d, truth.div_angle) < 30, a.div_angles)

        # The fruit curve starts on the stem axis and crosses the stem perpendicularly: fitting the
        # emergence tangent at s = 0 keeps the azimuth but pulls the polar angle towards 90°. The
        # default offset (2σbranch along the fruit) reduces both errors, without removing the polar
        # bias entirely (≈ 7° left at a true 73°; horizontal fruits are exact in both cases).
        a0 = ROMIAnglesAndInternodes(s; offset = 0)
        err(x) = angdist.(rad2deg.(x.azimuth), rad2deg.(truth.azimuths))
        @test sum(err(a)) < sum(err(a0))
        @test all(err(a) .< 10)
        bias(x) = sum(rad2deg.(x.polar) .- rad2deg(truth.polar)) / n
        @test 0 < bias(a) < bias(a0) - 5
        Vh, oh, th = voxel_plant(tilt = 0.0)
        sh = voxel_plant_skeleton(Vh, oh, th)
        for ah in (ROMIAnglesAndInternodes(sh; offset = 0), ROMIAnglesAndInternodes(sh))
            @test all(abs.(rad2deg.(ah.polar) .- 90) .< 3)
        end
        # the offset is moved back for fruits too short to hold the fitting window
        @test !any(ROMIAnglesAndInternodes(s; offset = 100).flagged)
    end

    @testset "Clockwise phyllotaxis" begin
        Vcw, ocw, tcw = voxel_plant(chirality = -1)
        acw = ROMIAnglesAndInternodes(voxel_plant_skeleton(Vcw, ocw, tcw))
        @test acw.orientation == :cw
        @test all(angdist.(rad2deg.(acw.azimuth), rad2deg.(tcw.azimuths)) .< 20)
        # divergence angles are measured in the direction of the spiral
        @test sum(acw.div_angles) / length(acw.div_angles) ≈ tcw.div_angle atol = 10
    end

    @testset "Voxel size scaling" begin
        # the same plant twice as large at twice the voxel size: identical voxels, doubled lengths
        V2, o2, t2 = voxel_plant(H = 150.0, rs = 2.2, rb = 1.0, L = 24.0, voxel_size = 1.0)
        @test V2 == V
        # smoothing scales are world lengths (mm) and must follow; the prominence counts voxel steps
        p2 = ROMISkeletonParams(σstem = 2 * s.params.σstem, σbranch = 2 * s.params.σbranch)
        s2 = voxel_plant_skeleton(V2, o2, t2; params = p2)
        a2 = ROMIAnglesAndInternodes(s2)
        @test length(s2.branchpoints) == n
        @test a2.internodes ≈ 2 .* a.internodes rtol = 1e-6
        @test a2.lengths ≈ 2 .* a.lengths rtol = 1e-6
        @test a2.div_angles ≈ a.div_angles
        @test a2.polar ≈ a.polar
    end

    @testset "Editing" begin
        # remove then re-add a fruit: identical result
        s1 = voxel_plant_skeleton(V, origin, truth)
        R.remove_fruit_tip!(s1, ids[3])
        @test length(s1.branchpoints) == n - 1
        @test ids[3] ∉ [t.node_id for t in s1.tip_ids]
        R.remove_fruit_tip!(s1, -1) # unknown node: no-op
        @test length(s1.branchpoints) == n - 1
        quiet(() -> R.add_fruit_tip!(s1, ROMITipID(node_id = ids[3])))
        @test [t.node_id for t in s1.tip_ids] == ids
        @test s1.branchpoints_arclength ≈ bps
        quiet(() -> R.add_fruit_tip!(s1, ROMITipID(node_id = ids[3]))) # already present: no-op
        @test length(s1.branchpoints) == n

        # replace the whole set of tips
        s1 = voxel_plant_skeleton(V, origin, truth)
        quiet(() -> R.update_fruit_tips!(s1, s1.tip_ids[[1, 4]]))
        @test [t.node_id for t in s1.tip_ids] == ids[[1, 4]]
        @test s1.branchpoints_arclength ≈ bps[[1, 4]]

        # a stem target forces where the fruit joins the stem
        s1 = voxel_plant_skeleton(V, origin, truth)
        target = s1.stem[findfirst(i -> zof(s1, i) > 30, s1.stem)]
        quiet(() -> R.add_fruit_tip!(s1, ROMITipID(node_id = ids[1], stem_target = target)))
        @test length(s1.branchpoints) == n
        i1 = findfirst(t -> t.node_id == ids[1], s1.tip_ids)
        @test s1.tip_ids[i1].stem_target == target
        @test s1.branchpoints_arclength[i1] > bps[1] + 15 # moved up the stem
        @test issorted(s1.branchpoints_arclength) # and re-sorted

        # lowering the stem top shortens the stem; fruits above it attach to the new top
        s1 = voxel_plant_skeleton(V, origin, truth)
        mid = s1.stem[findfirst(i -> zof(s1, i) > 25, s1.stem)]
        quiet(() -> R.update_stem_top!(s1, mid))
        @test s1.stem_top_id == mid
        @test s1.stem_curve.L_smooth < 30
        @test s1.branchpoints_arclength[1:2] ≈ bps[1:2] atol = vs
        @test all(s1.branchpoints_arclength[3:end] .≈ s1.stem_curve.L_smooth)

        # moving the root up the stem shifts all branch points by the same amount
        s1 = voxel_plant_skeleton(V, origin, truth)
        new_root = s1.stem[11]
        quiet(() -> R.update_stem_root!(s1, new_root))
        @test s1.vb.root_id == new_root
        @test length(s1.branchpoints) == n
        shift = bps .- s1.branchpoints_arclength
        @test all(shift .≈ zof(s1, new_root) - zof(s, s.stem[1]))
    end

    @testset "Parameters" begin
        # prominence: a huge value removes every fruit, the default brings them back
        s1 = voxel_plant_skeleton(V, origin, truth)
        quiet(() -> R.update_prominence!(s1, 1000.0))
        @test isempty(s1.branchpoints)
        quiet(() -> R.update_prominence!(s1, 15.0))
        @test length(s1.branchpoints) == n

        # smoothing changes the curves, not the topology
        s1 = voxel_plant_skeleton(V, origin, truth)
        quiet(() -> R.update_smoothing!(s1, 3.0, 1.0))
        @test (s1.params.σstem, s1.params.σbranch) == (3.0, 1.0)
        @test length(s1.branchpoints) == n
        @test s1.branchpoints_arclength ≈ bps atol = vs
        @test s1.stem_curve.L_smooth ≈ s.stem_curve.L_smooth rtol = 0.02

        p = ROMISkeletonParams(h = 3.0, σstem = 4.0)
        q = copy(p)
        @test q !== p
        @test all(getfield(q, f) == getfield(p, f) for f in fieldnames(ROMISkeletonParams))
    end

    @testset "Rebuild from saved state" begin
        # what reopening a saved plant does: root, top and tips are enough to get the same skeleton
        s0 = voxel_plant_skeleton(V, origin, truth)
        quiet(() -> R.remove_fruit_tip!(s0, s0.tip_ids[2].node_id))
        s1 = voxel_plant_skeleton(V, origin, truth; root = s0.vb.root_id)
        quiet(() -> R.update_stem_top!(s1, s0.stem_top_id))
        quiet(() -> R.update_fruit_tips!(s1, s0.tip_ids))
        @test s1.tip_ids == s0.tip_ids
        a0, a1 = ROMIAnglesAndInternodes(s0), ROMIAnglesAndInternodes(s1)
        @test a1.internodes ≈ a0.internodes
        @test a1.div_angles ≈ a0.div_angles
    end

    @testset "Stem top inside the volume" begin
        # when the stem top is not cut by the volume border it is found as a local maximum
        Vf, of, tf = voxel_plant(top_truncated = false)
        sf = voxel_plant_skeleton(Vf, of, tf)
        @test length(sf.branchpoints) == n
        @test norm(sf.stem_top[1:2]) < 3vs && sf.stem_top[3] > 75

        # re-detecting the tips must not turn the stem end into a fruit
        quiet(() -> R.update_prominence!(sf, 16.0))
        @test sf.stem_top_id ∉ [t.node_id for t in sf.tip_ids]
        @test length(sf.branchpoints) == n
        sf = voxel_plant_skeleton(Vf, of, tf)
        quiet(() -> R.update_stem_root!(sf, sf.stem[3]))
        @test length(sf.branchpoints) == n

        # nor when the stem top was placed by hand a few voxels below the stem end
        sf = voxel_plant_skeleton(Vf, of, tf)
        quiet(() -> R.update_stem_top!(sf, sf.stem[end - 5]))
        @test length(sf.branchpoints) == n
        for update! in (s -> R.update_prominence!(s, 14.0), s -> R.update_stem_root!(s, s.stem[3]))
            quiet(() -> update!(sf))
            @test length(sf.branchpoints) == n
            @test all(p -> p[3] < maximum(tf.heights) + 15, sf.tip_points) # no tip at the stem end
        end
    end

    @testset "Wrong automatic stem top" begin
        # a long top fruit reaches farther from the root than the stem end: it is taken for the stem
        # and the stem end for a fruit
        Vw, ow, tw = voxel_plant(H = 60.0, L = 30.0, top_truncated = false)
        sw = voxel_plant_skeleton(Vw, ow, tw)
        @test norm(sw.stem_top[1:2]) > 10 # the top is the end of a fruit
        @test any(p -> norm(p[1:2]) < 2vs, sw.tip_points) # and the stem end is a fruit
        # moving the top to the stem end (highest node on the stem axis) removes that fake fruit
        axis_ids = [id for id in 1:sw.vb.nv if norm(sw.vb.vox_grid[sw.vb.coords[id]][1:2]) < vs]
        quiet(() -> R.update_stem_top!(sw, argmax(id -> zof(sw, id), axis_ids)))
        @test norm(sw.stem_top[1:2]) < 2vs
        @test !any(p -> norm(p[1:2]) < 2vs, sw.tip_points)
        @test length(sw.branchpoints) == n - 1 # the top fruit is added back by hand (Space)
    end

    @testset "Input safety" begin
        @test_throws AssertionError R.ROMIBinaryVolume(falses(5, 5, 5), 2.0, 0.01)
        # small debris are dropped (only the largest component is kept)
        Vd = copy(V)
        Vd[1:2, 1:2, end-1:end] .= true
        sd = voxel_plant_skeleton(Vd, origin, truth)
        @test length(sd.branchpoints) == n
    end
end
