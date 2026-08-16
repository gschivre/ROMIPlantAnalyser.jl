"""
    emergence_tangent(curve::ArcLengthNWSmoothedCurve; s0::Real = 0.0, λ::Real = 0.5, N::Int = 30)

Fit a local 2nd-order vector polynomial p(s) ≈ p0 + v1*(s-s₀) + v2*(s-s₀)² over an 
exponentially weighted neighborhood near s = s₀ to isolate pure emergence direction t₀.
"""
function emergence_tangent(curve::ArcLengthNWSmoothedCurve{F}; s0::Real = 0.0, λ::Real = 0.5, N::Int = 30) where {F <: AbstractFloat}
    # Sample dense points near s = 0 (e.g. within 3 / λ)
    λF = convert(F, λ)
    s0F = convert(F, s0)
    half_width = 3 / λF
    s_min = max(zero(F), s0F - half_width)
    s_max = min(curve.L_smooth, s0F + half_width)
    s_samples = range(s_min, s_max; length = N)
    
    # Design matrix X = [1  s  s²] and weight matrix W
    X = Matrix{F}(undef, N, 3)
    W_diag = Vector{F}(undef, N)
    Y = Matrix{F}(undef, N, 3) # Positions [x y z]

    for (i, s) in enumerate(s_samples)
        X[i, 1] = one(F)
        X[i, 2] = s
        X[i, 3] = s ^ 2
        
        W_diag[i] = exp(-λF * s)
        
        pos = curve(s)
        Y[i, 1] = pos[1]
        Y[i, 2] = pos[2]
        Y[i, 3] = pos[3]
    end

    # V is (3 x 3) where row 1 = p0, row 2 = v1 (tangent), row 3 = v2 (curvature)
    sqrtW = sqrt.(W_diag)
    Xw = sqrtW .* X
    Yw = sqrtW .* Y
    V = Xw \ Yw
    
    v1 = Vec3{F}(V[2, 1], V[2, 2], V[2, 3])
    return v1 ./ sqrt(sum(abs2, v1))
end

"""
    emergence_angles(branch_curve::ArcLengthNWSmoothedCurve{F}, s_branch::F, stem_curve::ArcLengthNWSmoothedCurve{F}; ref_dir::Vec3{F} = Vec3{F}(1, 0, 0))

Compute the polar angle θ ∈ [0, π] and the azimuthal orientation φ ∈ [0, 2π) between the stem and emerging pedicel.
"""
function emergence_angles(branch_curve::ArcLengthNWSmoothedCurve{F}, s_branch::F, stem_curve::ArcLengthNWSmoothedCurve{F};
                            ref_dir::Vec3{F} = Vec3{F}(1, 0, 0)) where {F <: AbstractFloat}
    # Stem tangent at junction s_branch
    t0_stem = emergence_tangent(stem_curve; s0 = s_branch)
    
    # Branch emergence tangent at s = 0
    t0_branch = emergence_tangent(branch_curve)
    
    # Compute polar angle
    cos_θ = dot(t0_stem, t0_branch)

    # Project t0_branch onto plane normal to t0_stem
    proj = t0_branch - cos_θ * t0_stem
    proj_norm = sqrt(sum(abs2, proj))
    (proj_norm < eps(F)) && return (acos(clamp(cos_θ, -one(F), one(F))), zero(F))
    u = proj ./ proj_norm

    # Construct orthogonal basis (u_ref, v_ref) in transverse plane
    ref_proj = ref_dir - dot(t0_stem, ref_dir) * t0_stem
    u_ref = ref_proj ./ sqrt(sum(abs2, ref_proj))
    v_ref = cross(t0_stem, u_ref)

    # Compute azimuth angle
    ϕ = atan(dot(u, v_ref), dot(u, u_ref))
    return (acos(clamp(cos_θ, -one(F), one(F))), (ϕ < 0 ? ϕ + 2 * F(π) : ϕ))
end

"""
    stem_chirality(θ::Vector{T})

Heuristic to get the stem chirality by counting the number of times the shortest arc is clockwise or counter-clockwise.
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
end

# constructor from ROMISkeleton
function ROMIAnglesAndInternodes(s::ROMISkeleton)
    # internodes
    internodes = diff(s.branchpoints_arclength)

    # raw angles
    polar = zeros(length(s.branchpoints))
    azimuth = zeros(length(s.branchpoints))
    for i in eachindex(s.branchpoints)
        polar[i], azimuth[i] = emergence_angles(s.branch_curve[i], s.branchpoints_arclength[i], s.stem_curve)
    end

    # divergence angles
    if stem_chirality(azimuth) == :cw
        div_angles = rad2deg.(mod2pi.(-diff(azimuth)))
    else # ambiguous case default to :ccw
        div_angles = rad2deg.(mod2pi.(diff(azimuth)))
    end
    return ROMIAnglesAndInternodes(polar, azimuth, internodes, div_angles)
end