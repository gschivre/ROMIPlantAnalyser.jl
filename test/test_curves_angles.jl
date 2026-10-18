@testset "Curves" begin
    # straight line sampled every 0.5 along a known direction
    dir = normalize(Vec3d(1, 2, 2))
    line = [Point3d(1, 0, 0) + s * dir for s in 0:0.5:30]

    @testset "arclength" begin
        t = R.arclength(line)
        @test t[1] == 0 && t[end] ≈ 30
        @test all(diff(t) .≈ 0.5)
    end

    @testset "NW smoothing of a straight line is exact" begin
        c = R.ArcLengthNWSmoothedCurve(line, 2.0)
        @test c.L_smooth ≈ 30 rtol = 1e-6
        @test c(0.0) ≈ line[1]
        @test c(c.L_smooth) ≈ line[end]
        @test c(12.3) ≈ line[1] + 12.3 * dir atol = 1e-6
        @test R.tangent(c, 7.0) ≈ dir atol = 1e-6
        @test R.s_to_t(c, R.t_to_s(c, 11.0)) ≈ 11.0 atol = 1e-6

        pts = R.sample_uniform(c, 1.0)
        @test pts[1] == line[1] && pts[end] == line[end] # pinned end points
        @test all(d -> isapprox(d, 30 / (length(pts) - 1); rtol = 1e-4), norm.(diff(pts)))

        # closest point of an off-curve target is its orthogonal projection
        target = line[1] + 10 * dir + 3 * normalize(cross(dir, Vec3d(0, 0, 1)))
        @test R.t_to_s(c, R.closest_point_t(c, Point3d(target))) ≈ 10 atol = 1e-4
        # targets beyond the ends map to the ends
        @test R.closest_point_t(c, Point3d(line[1] - 5 * dir)) == c.curve.t_start
        @test R.closest_point_t(c, Point3d(line[end] + 5 * dir)) == c.curve.t_end
    end

    @testset "NW smoothing removes noise but keeps the length" begin
        # helix of radius 10, pitch 2π·2: exact length known
        ts = range(0, 4π; length = 800)
        helix = [Point3d(10cos(t), 10sin(t), 2t) for t in ts]
        exact = 4π * sqrt(10^2 + 2^2)
        c = R.ArcLengthNWSmoothedCurve(helix, 0.5)
        @test c.L_smooth ≈ exact rtol = 0.01

        # zig-zag noise of ±0.3 on a straight line: the raw length is inflated, the smoothed one is not
        zig = [Point3d(0.3 * (-1)^i, 0, 0.5i) for i in 0:200]
        cz = R.ArcLengthNWSmoothedCurve(zig, 2.0)
        @test R.arclength(zig)[end] > 1.3 * 100
        @test cz.L_smooth ≈ 100 rtol = 0.02
    end
end

@testset "Angles" begin
    @testset "_theta_phi" begin
        stem = Vec3d(0, 0, 1)
        ref = Vec3d(1, 0, 0)
        for θ in (0.3, π / 2, 2.0), φ in (0.1, 2.4, 5.5)
            b = Vec3d(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
            θ̂, φ̂ = R._theta_phi(stem, b, ref)
            @test θ̂ ≈ θ
            @test φ̂ ≈ φ
        end
        # azimuth is undefined for a branch parallel to the stem
        @test isnan(R._theta_phi(stem, stem, ref)[2])
        # or when the stem is parallel to the reference direction
        @test isnan(R._theta_phi(ref, stem, ref)[2])
    end

    @testset "stem_chirality" begin
        golden = π * (3 - sqrt(5))
        ccw = [mod2pi(i * golden) for i in 1:6]
        @test R.stem_chirality(ccw) == :ccw
        @test R.stem_chirality(mod2pi.(-ccw)) == :cw
        @test R.stem_chirality([0.0, 1.0, 0.0]) == :ambiguous
        # NaN azimuths (flagged branches) are ignored
        @test R.stem_chirality([ccw[1], NaN, ccw[2], ccw[3], ccw[4]]) == :ccw
    end

    @testset "emergence_angles on ideal curves" begin
        vs = 0.5
        stem_pts = [Point3d(0, 0, z) for z in 0:vs:60]
        stem = R.ArcLengthNWSmoothedCurve(stem_pts, 2.0)
        for (θ, φ) in ((deg2rad(45), deg2rad(30)), (deg2rad(70), deg2rad(200)), (deg2rad(100), deg2rad(310)))
            d = Vec3d(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
            bp = Point3d(0, 0, 25)
            branch = R.ArcLengthNWSmoothedCurve([bp + s * d for s in 0:vs:20], 1.0)
            res = R.emergence_angles(branch, 25.0, stem, vs)
            @test !res.flagged
            @test res.polar ≈ θ atol = deg2rad(1)
            @test res.azimuth ≈ φ atol = deg2rad(1)
            @test res.se_polar < 1 && res.se_azimuth < 1 # in degrees
        end
        # a branch along the stem cannot have an azimuth
        along = R.ArcLengthNWSmoothedCurve([Point3d(0, 0, 25 + s) for s in 0:vs:20], 1.0)
        @test R.emergence_angles(along, 25.0, stem, vs).flagged
    end
end
