"""
    _local_poly_fit(curve, s0, h, degree, N)

Weighted local polynomial fit of `curve` around `s0` in the dimensionless coordinate
u = (s-s0)/h with Gaussian kernel weights exp(-u²/2). Returns the tangent estimate `T`
(unit vector), its angular standard error `se_T` (radians).
"""
function _local_poly_fit(curve::ArcLengthNWSmoothedCurve{F}, s0::F, h::F, degree::Int, N::Int) where {F <: AbstractFloat}
    s_min = max(zero(F), s0 - 3 * h)
    s_max = min(curve.L_smooth, s0 + 3 * h)
 
    if (s_max - s_min) < (h / 10)
        return (T = F(NaN), se_T = F(NaN), is_degenerate = true)
    end
 
    s_samples = range(s_min, s_max; length = N)
    ncols = degree + 1
 
    X = Matrix{F}(undef, N, ncols)
    W_diag = Vector{F}(undef, N)
    Y = Matrix{F}(undef, N, 3)
 
    for (i, s) in enumerate(s_samples)
        x = (s - s0) / h
        for d in 0:degree
            X[i, d + 1] = x ^ d
        end
        W_diag[i] = exp(-(x ^ 2) / 2)
        pos = curve(s)
        Y[i, 1], Y[i, 2], Y[i, 3] = pos[1], pos[2], pos[3]
    end
 
    sqrtW = sqrt.(W_diag)
    Xw = sqrtW .* X
    Yw = sqrtW .* Y
    V = Xw \ Yw
 
    β1 = Vec3{F}(V[2, 1], V[2, 2], V[2, 3])
    norm_β1 = sqrt(sum(abs2, β1))
    T = β1 / norm_β1
 
    residuals = Yw .- (Xw * V)
    RSS = sum(abs2, residuals) # Σ w_i * ‖Y_i - Ŷ_i‖², summed over the 3 coordinates jointly

    # ν = Σ w_i (1 - H̃_ii) — leverages from the already-whitened Xw
    XtX_inv = inv(Xw' * Xw)
    leverages = vec(sum((Xw * XtX_inv) .* Xw; dims = 2))
    ν = max(sum(i -> W_diag[i] * (1 - leverages[i]), 1:N), one(F))
 
    # x, y, z share the same design/weights/leverages, so each contributes ν dof — pooling
    # all 3 coordinates' residuals (assuming isotropic noise) gives a more data-efficient
    # σ² estimate than fitting each coordinate's variance separately from 1/3 the data.
    σ_res² = RSS / (3 * ν)
    se_β1 = sqrt(σ_res² * XtX_inv[2, 2])

    # β1's SE is isotropic per-axis; angular deviation lives in the 2D plane perpendicular
    # to T (noise along T itself only rescales ‖β1‖, not direction), giving 2 independent
    # perpendicular components each with variance se_β1² — hence the √2.
    se_T = F(sqrt(2)) * se_β1 / norm_β1
 
    return (T = T, se_T = se_T, is_degenerate = false)
end

"""
    emergence_tangent(curve::ArcLengthNWSmoothedCurve, voxel_size; s0 = 0.0, h = 2 * voxel_size,
                       N = round(Int, 1 + 15 * h / voxel_size))

Fit a local 2nd-order vector polynomial p(s) ≈ p0 + v1*(s-s₀) + v2*(s-s₀)² over an 
exponentially weighted neighborhood near s = s₀ to isolate pure emergence direction t₀.
"""
function emergence_tangent(curve::ArcLengthNWSmoothedCurve{F}, voxel_size::F, s0::Real, h::F, N::Int) where {F <: AbstractFloat}
    fit = _local_poly_fit(curve, F(s0), h, 2, N)
    fit.is_degenerate && @warn "emergence_tangent: fitting window too narrow near s0=$s0 (h=$h) — segment may be too short for this voxel size."
    return fit
end


"""
    _theta_phi(t0_stem, t0_branch, ref_dir)
 
Polar angle θ and azimuth φ of `t0_branch` relative to `t0_stem`, measured
against `ref_dir`.
"""
function _theta_phi(t0_stem::Vec3, t0_branch::Vec3, ref_dir::Vec3)
    # Compute polar angle
    cos_θ = dot(t0_stem, t0_branch)
    cos_θ = clamp(cos_θ, -one(cos_θ), one(cos_θ))
    θ = acos(cos_θ)

    # Project t0_branch onto plane normal to t0_stem
    proj = t0_branch - cos_θ * t0_stem
    proj_norm = sqrt(sum(abs2, proj))
    (proj_norm < sqrt(eps(θ))) && return (θ, oftype(θ, NaN))
    u = proj ./ proj_norm

    # Construct orthogonal basis (u_ref, v_ref) in transverse plane
    dot_ref = dot(t0_stem, ref_dir)
    (dot_ref > (1 - sqrt(eps(θ)))) && return (θ, oftype(θ, NaN))
    ref_proj = ref_dir - dot_ref * t0_stem
    u_ref = ref_proj ./ sqrt(sum(abs2, ref_proj))
    v_ref = cross(t0_stem, u_ref)

    # Compute azimuth angle
    ϕ = atan(dot(u, v_ref), dot(u, u_ref))
    ϕ = (ϕ < 0 ? ϕ + 2π : ϕ)
    return (θ, ϕ)
end
 
"""
    _perp_basis(v)
 
An orthonormal basis (e1, e2) spanning the plane perpendicular to unit vector `v` — used
to parametrize small perturbations of `v` that keep it (to first order) a unit vector.
"""
function _perp_basis(v::Vec3{F}) where {F <: AbstractFloat}
    i = argmin(abs.(v))
    seed = (i == 1 ? Vec3{F}(1, 0, 0) : (i == 2 ? Vec3{F}(0, 1, 0) : Vec3{F}(0, 0, 1)))
    e1 = seed - dot(v, seed) * v
    e1 = e1 ./ sqrt(sum(abs2, e1))
    e2 = cross(v, e1)
    return (e1, e2)
end
 
"""
    _theta_phi_jacobian(t0_stem, t0_branch, ref_dir)
 
Jacobian of (θ, φ) at zero perturbation, with respect to 4 independent tangent-plane
perturbation coordinates (2 for t0_stem, 2 for t0_branch), via `ForwardDiff.jacobian`. 
Returns a 2×4 Matrix, columns ordered (δs1, δs2, δb1, δb2).
"""
function _theta_phi_jacobian(t0_stem::Vec3{F}, t0_branch::Vec3{F}, ref_dir::Vec3{F}) where {F <: AbstractFloat}
    es1, es2 = _perp_basis(t0_stem)
    eb1, eb2 = _perp_basis(t0_branch)
 
    function f(δ)
        s = t0_stem + δ[1] * es1 + δ[2] * es2
        s = s ./ sqrt(sum(abs2, s))
        b = t0_branch + δ[3] * eb1 + δ[4] * eb2
        b = b ./ sqrt(sum(abs2, b))
        θ, ϕ = _theta_phi(s, b, ref_dir)
        return [θ, ϕ]
    end
 
    return ForwardDiff.jacobian(f, zeros(F, 4))
end
 
"""
    emergence_angles(branch_curve::ArcLengthNWSmoothedCurve{F}, s_branch::F, stem_curve::ArcLengthNWSmoothedCurve{F},
                      voxel_size::F; ref_dir::Vec3{F} = Vec3{F}(1, 0, 0), h::F = 2 * voxel_size,
                      N::Int = round(Int, 1 + 15 * h / voxel_size))
 
Compute the polar angle θ ∈ [0, π] and the azimuthal orientation φ ∈ [0, 2π) between the
stem and emerging branch and the propagated uncertainty of (θ,φ) from both the stem and branch tangent standard errors, 
via the delta method with a `ForwardDiff`-computed Jacobian.
 
If either fit is too degenerate to trust, all angle/uncertainty fields come back `NaN` with `flagged = true`.
"""
function emergence_angles(branch_curve::ArcLengthNWSmoothedCurve{F}, s_branch::F, stem_curve::ArcLengthNWSmoothedCurve{F},
                            voxel_size::F; ref_dir::Vec3{F} = Vec3{F}(1, 0, 0),
                            h::F = 2 * voxel_size, N::Int = round(Int, 1 + 15 * h / voxel_size)) where {F <: AbstractFloat}
    _nan_result = (polar = F(NaN), azimuth = F(NaN), flagged = true,
                   se_polar = F(NaN), se_azimuth = F(NaN), cov_θϕ = F(NaN))
 
    # Stem tangent at junction s_branch
    stem_fit = emergence_tangent(stem_curve, voxel_size, s_branch, h, N)
    stem_fit.is_degenerate && return _nan_result
    t0_stem = stem_fit.T
    σ_s = stem_fit.se_T
 
    # Branch emergence tangent at s = 0
    branch_fit = emergence_tangent(branch_curve, voxel_size, zero(F), h, N)
    branch_fit.is_degenerate && return _nan_result
    t0_branch = branch_fit.T
    σ_b = branch_fit.se_T
 
    θ, ϕ = _theta_phi(t0_stem, t0_branch, ref_dir)
    if isnan(ϕ)
        # stem ‖ branch or stem ‖ ref_dri: azimuth (and its uncertainty) undefined
        return (polar = θ, azimuth = F(NaN), flagged = true,
                se_polar = rad2deg(σ_s), se_azimuth = F(NaN), cov_θϕ = F(NaN))
    end
 
    J = _theta_phi_jacobian(t0_stem, t0_branch, ref_dir)
    Σ_diag = [(σ_s ^ 2) / 2, (σ_s ^ 2) / 2, (σ_b ^ 2) / 2, (σ_b ^ 2) / 2]
    Cov_rad = J * Diagonal(Σ_diag) * J'
    rad2deg_sq = (180 / F(π)) ^ 2
 
    return (polar = θ, azimuth = ϕ, flagged = false,
            se_polar = rad2deg(sqrt(Cov_rad[1, 1])), se_azimuth = rad2deg(sqrt(Cov_rad[2, 2])), cov_θϕ = Cov_rad[1, 2] * rad2deg_sq)
end

"""
    stem_chirality(θ::Vector{T})

Heuristic to get the stem chirality by counting the number of times the shortest arc is clockwise or counter-clockwise.
NaN entries (from degenerate branches) compare `false` to both `>(0)` and `<(0)` in Julia,
so they're silently excluded from the vote rather than corrupting it — no special-casing needed.
"""
function stem_chirality(θ::Vector{T}) where {T <: AbstractFloat}
    δ = (rem2pi(θ[i + 1] - θ[i], RoundNearest) for i in 1:(length(θ) - 1)) # signed minimal arc ∈ (-π,π]
    n₊, n₋ = count(>(0), δ), count(<(0), δ)
    return (n₊ > n₋ ? :ccw : (n₋ > n₊ ? :cw : :ambiguous))
end

struct ROMIAnglesAndInternodes
    # angles in radian
    polar::Vector{Float64}
    azimuth::Vector{Float64}

    # internodes in mm
    internodes::Vector{Float64}

    # divergence angles in °
    div_angles::Vector{Float64}
    orientation::Symbol

    # tangent-estimation quality (in ° for interpretability!)
    se_polar::Vector{Float64} # polar angle standard error in °
    se_azimuth::Vector{Float64} # azimuth angle standard error in °
    cov_θϕ::Vector{Float64} # polar/azimuth covariance in °²
    flagged::Vector{Bool}
end

# constructor from ROMISkeleton
function ROMIAnglesAndInternodes(s::ROMISkeleton)
    voxel_size = Float64(s.vb.voxel_size)

    # internodes
    internodes = diff(s.branchpoints_arclength)

    # raw angles + tangent quality diagnostics
    n = length(s.branchpoints)
    polar = zeros(n)
    azimuth = zeros(n)
    se_polar = zeros(n)
    se_azimuth = zeros(n)
    cov_θϕ = zeros(n)
    flagged = falses(n)
    for i in eachindex(s.branchpoints)
        res = emergence_angles(s.branch_curve[i], s.branchpoints_arclength[i], s.stem_curve, voxel_size)
        polar[i] = res.polar
        azimuth[i] = res.azimuth
        se_polar[i] = res.se_polar
        se_azimuth[i] = res.se_azimuth
        cov_θϕ[i] = res.cov_θϕ
        flagged[i] = res.flagged
    end

    # divergence angles
    orientation = stem_chirality(azimuth)
    if orientation == :cw
        div_angles = rad2deg.(mod2pi.(-diff(azimuth)))
    else # ambiguous case default to :ccw
        div_angles = rad2deg.(mod2pi.(diff(azimuth)))
    end
    return ROMIAnglesAndInternodes(polar, azimuth, internodes, div_angles, orientation, se_polar, se_azimuth, cov_θϕ, flagged)
end