@testset "Geometry" begin
    cam = ROMICamera(64, 48, 50.0, 50.0, 32.0, 24.0, 0.0, 0.0, 0.0, 0.0)
    # camera at the origin looking down +z
    frame = ROMIFrame(Mat3d(1, 0, 0, 0, 1, 0, 0, 0, 1), Vec3d(0, 0, 0), cam)

    @testset "point2frame" begin
        # the optical axis hits the principal point (1-based pixel indices)
        @test R.point2frame(Point3d(0, 0, 10), frame) == CartesianIndex(25, 33)
        @test R._point2frame(Point3d(0, 0, 10), frame) ≈ Point2d(32, 24)

        # x is the column, y the row
        @test R.point2frame(Point3d(1, 0, 10), frame) == CartesianIndex(25, 38)
        @test R.point2frame(Point3d(0, 1, 10), frame) == CartesianIndex(30, 33)

        # behind the camera or outside the sensor
        @test R.point2frame(Point3d(0, 0, -1), frame) === nothing
        @test R.point2frame(Point3d(100, 0, 10), frame) === nothing

        # the projection follows X_c = R * X_w + t
        θ = π / 5
        Rz = Mat3d(cos(θ), sin(θ), 0, -sin(θ), cos(θ), 0, 0, 0, 1)
        t = Vec3d(0.3, -0.2, 5)
        moved = ROMIFrame(Rz, t, cam)
        p = Point3d(0.4, -0.1, 3)
        @test R._point2frame(p, moved) ≈ R._point2frame(Point3d(Rz * p + t), frame)

        # radial distortion pushes points outward (k1 > 0)
        dcam = ROMICamera(64, 48, 50.0, 50.0, 32.0, 24.0, 0.1, 0.0, 0.0, 0.0)
        dframe = ROMIFrame(frame.R, frame.t, dcam)
        p = Point3d(2, 1, 10)
        @test norm(R._point2frame(p, dframe) - Point2d(32, 24)) > norm(R._point2frame(p, frame) - Point2d(32, 24))
    end

    @testset "ROMIVoxelGrid" begin
        bbox = Rect3d(Point3d(-1, 2, 0), Vec3d(4, 2, 6))
        g = R.ROMIVoxelGrid(bbox, 0.5)
        @test size(g) == (9, 5, 13)
        @test g[1, 1, 1] ≈ Point3d(-0.75, 2.25, 0.25) # voxel centers
        @test step(g.x) ≈ 0.5

        g2 = R.ROMIVoxelGrid(Point3d(-1, 2, 0), (8, 4, 12), 0.5)
        @test g2[1, 1, 1] ≈ g[1, 1, 1]
        @test g2[end, end, end] ≈ Point3d(2.75, 3.75, 5.75)
        @test R.Rect3(g2).origin ≈ bbox.origin
        @test R.Rect3(g2).widths ≈ bbox.widths
    end

    @testset "clip_segment" begin
        # fully inside: unchanged
        a, b = Point2d(1, 1), Point2d(3, 2)
        @test R.clip_segment(a, b, 0, 4, 0, 4) == (a, b)
        # fully outside
        @test R.clip_segment(Point2d(5, 5), Point2d(6, 7), 0, 4, 0, 4) === nothing
        # crossing the whole rectangle
        c = R.clip_segment(Point2d(-2, 2), Point2d(6, 2), 0, 4, 0, 4)
        @test c[1] ≈ Point2d(0, 2) && c[2] ≈ Point2d(4, 2)
        # diagonal entering from a corner side
        c = R.clip_segment(Point2d(-1, -1), Point2d(2, 2), 0, 4, 0, 4)
        @test c[1] ≈ Point2d(0, 0) && c[2] ≈ Point2d(2, 2)
    end

    @testset "project_bbox" begin
        # a box well in front of the camera projects to 12 edges, all inside the image
        bbox = Rect3d(Point3d(-0.5, -0.5, 9), Vec3d(1, 1, 1))
        segs = R.project_bbox(bbox, frame)
        @test length(segs) == 12
        @test all(s -> all(p -> (0 ≤ p[1] ≤ 64) && (0 ≤ p[2] ≤ 48), s), segs)
        # a box behind the camera has no visible edge
        @test isempty(R.project_bbox(Rect3d(Point3d(-0.5, -0.5, -5), Vec3d(1, 1, 1)), frame))
        # a huge box gets clipped to the image borders
        segs = R.project_bbox(Rect3d(Point3d(-50, -50, 5), Vec3d(100, 100, 10)), frame)
        @test all(s -> all(p -> (0 ≤ p[1] ≤ 64) && (0 ≤ p[2] ≤ 48), s), segs)
    end

    @testset "Bounding box parameters" begin
        p = ROMIBboxParams(x_offset = 1, y_offset = -2, z_offset = -30, x_span = 10, y_span = 20, z_span = 40)
        q = copy(p)
        @test q !== p
        @test all(getfield(q, f) == getfield(p, f) for f in fieldnames(ROMIBboxParams))
    end
end
