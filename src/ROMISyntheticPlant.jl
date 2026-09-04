# =====================================================================================
# Synthetic plant scan generator
# =====================================================================================
#
# A convenience fixture for precompile.jl's warm-up workloads, not a general-purpose
# demo-data tool — so the design goal here is different from a typical renderer: instead
# of choosing world-space sizes and hoping they land at a sensible pixel size, every
# geometric parameter is *solved backwards* from a fixed target in PIXELS (stem/branch
# width, stem/branch length as a fraction of the frame). That's what line-enhancement
# masking actually cares about, and it means resolution (w, h) can change freely without
# ever having to retune anything else — a small imaged object always renders at the same
# few pixels wide, however many pixels the image itself has.
#
# Stem and branches are capsules — cylinders with hemispherical (not flat) end caps.
# Round cross-section means apparent width is the same from every azimuthal viewing
# angle; rounded ends (vs. a flat-capped cylinder) mean no flare from the cap's rim
# becoming visible at oblique angles, which a real stem/branch tip wouldn't show either.
# Renders from a single ring of cameras above the plant, tilted down toward it, and
# writes a real ROMIScan-compatible directory (images/ + metadata/images/*.json). No
# shipped image assets: procedural, generated at runtime in milliseconds at these sizes.

struct SynthCapsule
    center::Point3d
    axis::Vec3d      # unit vector along the capsule's length
    radius::Float64
    halflength::Float64  # half-length of the cylindrical portion only (caps add radius beyond this)
end

"""
    _ray_capsule_hit(origin, dir, cap)

Ray/capsule intersection, as a boolean hit test (silhouette rendering only, no depth
needed). A capsule's surface is the cylindrical side wall clipped to its axial range,
plus a sphere of the same radius at each pole. Testing the *full* sphere at each pole
(not just its outer hemisphere) is safe: the inner half already overlaps the solid
cylinder body, so it can't introduce any hit outside the true capsule shape.
"""
function _ray_capsule_hit(origin::Point3d, dir::Vec3d, cap::SynthCapsule)
    a = cap.axis
    oc = origin - cap.center
    oc_z = dot(oc, a)
    oc_perp = oc - oc_z .* a
    d_z = dot(dir, a)
    d_perp = dir - d_z .* a

    # cylindrical side wall, clipped to the axial range (no flat caps here)
    A = dot(d_perp, d_perp)
    if A > 1e-12
        B = 2 * dot(oc_perp, d_perp)
        C = dot(oc_perp, oc_perp) - cap.radius^2
        disc = B^2 - 4A * C
        if disc >= 0
            sq = sqrt(disc)
            for t in ((-B - sq) / (2A), (-B + sq) / (2A))
                if t >= 0 && abs(oc_z + t * d_z) <= cap.halflength
                    return true
                end
            end
        end
    end

    # hemispherical caps (ray-sphere at each pole; dir is unit length, so A=1 below)
    for s in (-1.0, 1.0)
        pole = cap.center + (s * cap.halflength) .* a
        oc2 = origin - pole
        b = 2 * dot(dir, oc2)
        c = dot(oc2, oc2) - cap.radius^2
        disc = b^2 - 4c
        if disc >= 0
            sq = sqrt(disc)
            t1 = (-b - sq) / 2
            t2 = (-b + sq) / 2
            (t1 >= 0 || t2 >= 0) && return true
        end
    end

    return false
end

"""
    _phyllotaxis_shapes(; n_branches, stem_height, stem_radius, branch_length, branch_radius)

A vertical stem capsule plus `n_branches` branch capsules spiraling up it at the golden
angle (≈137.5°), each tilted slightly upward off horizontal.
"""
function _phyllotaxis_shapes(; n_branches::Int, stem_height::Real, stem_radius::Real,
                              branch_length::Real, branch_radius::Real)
    shapes = SynthCapsule[SynthCapsule(Point3d(0, 0, stem_height / 2), Vec3d(0, 0, 1), stem_radius, stem_height / 2)]
    golden_angle = π * (3 - sqrt(5))
    for i in 1:n_branches
        h = stem_height * (0.15 + 0.75 * (i - 1) / max(n_branches - 1, 1))
        θ = i * golden_angle
        out_dir = normalize(Vec3d(cos(θ), sin(θ), 0.3))
        center = Point3d(0, 0, h) + Vec3d((stem_radius + branch_length / 2) .* out_dir)
        push!(shapes, SynthCapsule(center, out_dir, branch_radius, branch_length / 2))
    end
    return shapes
end

"""
    _shapes_bbox(shapes; pad)

Exact axis-aligned bounding box of a set of capsules. Unlike a plain cylinder, a
capsule's extent along world axis `e_k` is simply `L·|a·e_k| + r` — the hemispherical
caps mean the radius contributes in full regardless of the axis' orientation, rather
than the `r·√(1-(a·e_k)²)` term a flat-capped cylinder would need.
"""
function _shapes_bbox(shapes::Vector{SynthCapsule}; pad::Real)
    lo = fill(Inf, 3)
    hi = fill(-Inf, 3)
    for cap in shapes, k in 1:3
        half_extent = cap.halflength * abs(cap.axis[k]) + cap.radius
        lo[k] = min(lo[k], cap.center[k] - half_extent)
        hi[k] = max(hi[k], cap.center[k] + half_extent)
    end
    lo_pt = Point3d(lo[1] - pad, lo[2] - pad, lo[3] - pad)
    hi_pt = Point3d(hi[1] + pad, hi[2] + pad, hi[3] + pad)
    return Rect3d(lo_pt, hi_pt .- lo_pt)
end

"""
    _render_view(...)

2×2 supersampled (4 sub-rays per pixel, OR-combined) rather than one ray through the
pixel center — at the 1-2px target widths this fixture aims for, a single center ray per
pixel risks aliasing a thin capsule into a dotted rather than solid line, which would
undermine the actual point of rendering at a controlled pixel width.
"""
function _render_view(eye::Point3d, target::Point3d, w::Int, h::Int, fx::Real, fy::Real,
                       cx::Real, cy::Real, shapes::Vector{SynthCapsule})
    fwd = Vec3d(normalize(target - eye))
    right = normalize(cross(fwd, Vec3d(0, 0, 1)))
    down = normalize(cross(fwd, right))
    img = fill(RGB{N0f8}(0.08, 0.05, 0.03), h, w) # soil-brown background
    subs = (-0.25, 0.25)
    for v in 1:h, u in 1:w
        hit = false
        for du in subs, dv in subs
            dir = normalize(((u - 0.5 + du - cx) / fx) .* right .+ ((v - 0.5 + dv - cy) / fy) .* down .+ fwd)
            for cap in shapes
                if _ray_capsule_hit(eye, dir, cap)
                    hit = true
                    break
                end
            end
            hit && break
        end
        hit && (img[v, u] = RGB{N0f8}(0.15, 0.55, 0.15))
    end
    return img, right, down, fwd
end

"""
    generate_synthetic_plant_scan(dir::String; n_frames = 6, w = 32, h = 64, n_branches = 4)

Render a synthetic stem-and-branches "plant" as a precompile-warm-up fixture and write it
to `dir` as a real ROMIScan directory.

Every geometric quantity — stem/branch length and radius, camera distance, target point —
is solved from fixed PIXEL targets rather than being an independent world-space guess:
the stem always renders ~3px wide and spans most of the frame height; branches always
render ~1-2px wide and stay short relative to the frame width (keeping the reconstructed
bounding box compact, so carving stays cheap). `n_frames`/`w`/`h`/`n_branches` are the
only knobs — everything else is an internal constant, since this isn't meant to be tuned
per use, just fast and reliable as a fixture.

Returns `(scan, vol_params, bbox_params)`. `bbox_params` is a `ROMIBboxParams` solved to
exactly reproduce this scan's true bbox through `initialize_bbox`'s own formula — the
*default* `ROMIBboxParams()` assumes cameras orbit above the plant (real scanner
geometry), which doesn't hold for this fixture's side-view ring, so pass `bbox_params`
explicitly rather than relying on the default when loading synthetic scans interactively.

Note on perspective: `elev_deg` (how far the camera looks down) and `fov_deg`/
`stem_px_len_frac` (how much of the frame the stem fills) jointly control how much
foreshortening is visible along the stem's length — the near part (toward the camera's
own height) will always render a bit wider than the far part for a perspective camera at
a steep angle, same as a real photograph would show. Lowering `fov_deg` (more
"telephoto", requiring proportionally more `cam_dist` to keep the same framing, which the
pixel-target solve does automatically) reduces this; `cam_dist` alone does not, since
world size scales proportionally with it.
"""
function generate_synthetic_plant_scan(dir::String;
                                        n_frames::Int = 12, w::Int = 32, h::Int = 64,
                                        n_branches::Int = 4)
    fov_deg = 53.0          # horizontal FOV; a "normal lens" figure
    cam_dist = 100.0        # camera-to-target distance — an arbitrary unit scale, doesn't
                             # affect voxel count (world size and voxel_size both derive
                             # from the same pixel targets, so it cancels out)
    elev_deg = 0.0          # level ring — confirmed to render best; see the note above on
                             # why this means bbox_params (not the ROMIBboxParams default)
                             # is needed to frame it correctly
    stem_px_width = 2.0
    branch_px_width = 1.5
    stem_px_len_frac = 0.8  # stem spans this fraction of image height
    branch_px_len_frac = 0.3 # branches span this fraction of image width

    half_fov = deg2rad(fov_deg) / 2
    fx = w / (2 * tan(half_fov))
    fy = fx # square pixels — matches COLMAP's own camera models and real hardware
    cx, cy = w / 2, h / 2

    # solve world-space sizes from the pixel targets (inverse of s_px = s_world * fx / d)
    stem_height = stem_px_len_frac * h * cam_dist / fx
    stem_radius = stem_px_width * cam_dist / (2fx)
    branch_length = branch_px_len_frac * w * cam_dist / fx
    branch_radius = branch_px_width * cam_dist / (2fx)

    shapes = _phyllotaxis_shapes(; n_branches, stem_height, stem_radius, branch_length, branch_radius)
    bbox = _shapes_bbox(shapes; pad = 2 * branch_radius)

    target = Point3d(0, 0, stem_height / 2)
    elev = deg2rad(elev_deg)

    images_dir = mkpath(joinpath(dir, "images"))
    metadata_dir = mkpath(joinpath(dir, "metadata", "images"))
    cam = ROMICamera(w, h, fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0)

    scan_frames = Vector{ROMIFrame{Int, Float64}}(undef, n_frames)
    for i in 1:n_frames
        θ = 2π * (i - 1) / n_frames
        eye = target + cam_dist .* Vec3d(cos(elev) * cos(θ), cos(elev) * sin(θ), sin(elev))
        img, right, down, fwd = _render_view(eye, target, w, h, fx, fy, cx, cy, shapes)

        name = lpad(i - 1, 5, '0') * "_rgb"
        save(joinpath(images_dir, name * ".jpg"), img)
        open(joinpath(metadata_dir, name * ".json"), "w") do io
            JSON.print(io, Dict("approximate_pose" => [eye[1], eye[2], eye[3], 0.0]))
        end

        R = Mat3d(right[1], down[1], fwd[1], right[2], down[2], fwd[2], right[3], down[3], fwd[3])
        t = -R * Vec3d(eye[1], eye[2], eye[3])
        scan_frames[i] = ROMIFrame(R, t, cam)
    end

    scan = ROMIScan(dir)
    scan.frames .= scan_frames
    # voxel_size resolves the thinnest feature (branches) at a few voxels across; coarser
    # than that trades topology fidelity for speed, finer buys little and costs a lot —
    # count grows with the cube of the inverse.
    vol_params = ROMIVolumeParams(bbox = bbox, voxel_size = 1.2 * branch_radius, λ = 1.0)

    # ROMIBboxParams' *defaults* assume pose_centroid sits above the plant (a real
    # scanner's cameras orbit above the pot); at elev_deg=0 our ring sits at the plant's
    # own mid-height instead, so those defaults don't apply here — no choice of angle
    # reconciles this without inflating stem_height past what a fixed 250-unit z_span
    # can hold, since stem_height and camera height both scale with the same cam_dist.
    # Solve ROMIBboxParams backwards from initialize_bbox's own formula instead, so it
    # reproduces the true bbox exactly for THIS scan's actual camera placement.
    pcentroid = Point3d(sum(-f.R' * f.t for f in scan_frames) / n_frames)
    bbox_params = ROMIBboxParams(
        x_offset = round(Int, bbox.origin[1] - pcentroid[1] + bbox.widths[1] / 2),
        y_offset = round(Int, bbox.origin[2] - pcentroid[2] + bbox.widths[2] / 2),
        z_offset = round(Int, bbox.origin[3] - pcentroid[3] + bbox.widths[3]),
        x_span = ceil(Int, bbox.widths[1]),
        y_span = ceil(Int, bbox.widths[2]),
        z_span = ceil(Int, bbox.widths[3]),
    )

    return scan, vol_params, bbox_params
end