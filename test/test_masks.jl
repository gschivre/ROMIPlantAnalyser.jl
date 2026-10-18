@testset "Masks" begin
    @testset "Structuring elements" begin
        @test R.linPath(CartesianIndex(1, 1), CartesianIndex(1, 1)) == [CartesianIndex(1, 1)]
        path = R.linPath(CartesianIndex(0, 0), CartesianIndex(3, 6))
        @test first(path) == CartesianIndex(0, 0) && last(path) == CartesianIndex(3, 6)
        @test length(path) == 7

        # horizontal line of half-length 2: 4 offsets, the origin excluded
        se = R.strel_line(2, 0)
        @test sort(se) == sort([CartesianIndex(0, d) for d in (-2, -1, 1, 2)])
        @test sort(R.strel_line(2, 90)) == sort([CartesianIndex(d, 0) for d in (-2, -1, 1, 2)])
        @test R.strel_line(0, 30) == [CartesianIndex(0, 0)]
    end

    @testset "Features" begin
        # dark background, a thin bright line and a large bright square
        img = fill(Gray{N0f8}(0.1), 40, 40)
        img[:, 10:11] .= Gray{N0f8}(0.9)
        img[20:35, 22:37] .= Gray{N0f8}(0.9)
        feat = R.thin_lines_enhancement(img, 2)
        @test minimum(feat[5:15, 10:11]) > 0.5 # thin line kept
        @test maximum(feat[24:31, 26:33]) < 0.05 # bulk of the square removed
        @test maximum(feat[1:5, 30:40]) < 0.05 # background stays dark

        # excess green separates green from gray and dark pixels
        rgb = [RGB{N0f8}(0.1, 0.6, 0.1) RGB{N0f8}(0.5, 0.5, 0.5) RGB{N0f8}(0.01, 0.02, 0.01)]
        exg = R.get_ExG!(similar(rgb, Gray{N0f8}), rgb, 0.2)
        g = Float64.(gray.(exg))
        @test g[1] > 0.7
        @test g[2] ≈ 1 / 3 atol = 0.01
        @test g[3] == 0.0 # below the brightness threshold
    end

    @testset "Connected components" begin
        m = falses(10, 10)
        m[1:2, 1:2] .= true # 4 px
        m[5:9, 5:9] .= true # 25 px
        m[1, 9] = true # 1 px
        m2 = copy(m)
        R.remove_small_component!(m2, 4)
        @test sum(m2) == 29 && !m2[1, 9]
        R.keep_largest_component!(m)
        @test sum(m) == 25 && m[7, 7]
    end

    @testset "Mask parameters" begin
        feat = zeros(Gray{N0f8}, 20, 20)
        feat[5:15, 10] .= Gray{N0f8}(0.8)
        feat[2, 2] = Gray{N0f8}(0.8) # isolated noise
        p = ROMIMaskParams(t = 0.5, m = 3, d = 0)
        mask = R.get_mask(feat, p)
        @test sum(mask) == 11 && !mask[2, 2]
        dilated = R.get_mask(feat, ROMIMaskParams(t = 0.5, m = 3, d = 3))
        @test sum(dilated) > sum(mask) && all(dilated[mask])

        q = copy(p)
        @test q !== p
        @test all(getfield(q, f) == getfield(p, f) for f in fieldnames(ROMIMaskParams))
    end
end
