@kwdef mutable struct ROMIMaskParams
    # use line enhancement
    use_line::Bool = true

    # gaussian blur
    σ::Float64 = 0.0

    # strel_line half-length (3 ⟹ 7 pixels) (used when use_line == true)
    lh::Int = 2

    # brightness treshold (used when use_line == false)
    l::Float64 = 127 / 255

    # features (lines or ExG) treshold
    t::Float64 = 15 / 255
    
    # minimum size in pixel
    m::Int = 0

    # dilatation factor
    d::Int = 3
end
function Base.copyto!(dest::ROMIMaskParams, src::ROMIMaskParams)
    dest.use_line = src.use_line
    dest.σ = src.σ
    dest.lh = src.lh
    dest.l = src.l
    dest.t = src.t
    dest.m = src.m
    dest.d = src.d
    return dest
end
function Base.copy(p::ROMIMaskParams)
    pnew = ROMIMaskParams()
    copyto!(pnew, p)
end

struct ROMIMaskedFrames <: AbstractArray{Bool, 3}
    # frames and corresponding masks
    frames::Vector{ROMIFrame{Int, Float64}}
    masks::BitArray{3}
    features::Array{Gray{N0f8}, 3}

    # store used parameters
    params::ROMIMaskParams

    function ROMIMaskedFrames(dataset::ROMIScan, params::ROMIMaskParams)
        frames = dataset.frames
        (; width, height) = frames[1].camera
        feats = Array{Gray{N0f8}}(undef, height, width, length(frames))
        masks = falses(size(feats))
        get_features!(feats, dataset, params)
        get_masks!(masks, feats, params)
        new(frames, masks, feats, params)
    end
end
Base.size(m::ROMIMaskedFrames) = size(m.masks)
Base.getindex(m::ROMIMaskedFrames, I...) = m.masks[I...]

# Bresenham's linear path in nD
function linPath(p1::CartesianIndex{N}, p2::CartesianIndex{N}) where N
    m = maximum(abs, Tuple(p2 - p1))
    (m == 0) && return [p1]
    ptslist = Vector{CartesianIndex{N}}(undef, m + 1)
    @inbounds for i in 0:m
        t = i / m
        ptslist[i + 1] = CartesianIndex{N}(
            ntuple(d -> round(Int, p1[d] + t * (p2[d] - p1[d]), RoundNearestTiesAway), N)
        )
    end
    return ptslist
end

"""
    strel_line(lh::Int, θ::Real)

Create a 2D line structuring element of half-length `lh` and orientation `θ` in °.
"""
function strel_line(lh::Int, θ::Real)
    lh == 0 && return [CartesianIndex(0, 0)]
    c, s = cosd(θ), sind(θ)
    scale = lh / max(abs(c), abs(s))

    # switch x and y to match y indexing row
    dx = round(Int, scale * s)
    dy = round(Int, scale * c)

    p1 = CartesianIndex(-dx, -dy)
    p2 = CartesianIndex(dx, dy)

    return filter!(!=(CartesianIndex(0, 0)), linPath(p1, p2))
end

"""
    thin_lines_enhancement(im::AbstractMatrix, lh::Int)

Return an image with enhanced thin lines (smaller than `lh`) of the input a gray image `im`.
This works by taking the max opening with every rotated line structuring element and 
substracting the opening from a box of matching size.
"""
function thin_lines_enhancement!(out::AbstractMatrix, im::AbstractMatrix, lh::Int)
    buf1 = similar(im)
    buf2 = similar(im)
    opening!(out, im, strel_line(lh, 0), buf1)
    for se in (strel_line(lh, -i * 45 / lh) for i in 1:(4 * lh - 1))
        opening!(buf2, im, se, buf1)
        for i in eachindex(out)
            out[i] = max(out[i], buf2[i])
        end
    end
    opening!(buf2, im, strel_box((2 * lh + 1, 2 * lh + 1)), buf1)
    out .-= buf2
    return out
end
thin_lines_enhancement(im::AbstractMatrix, lh::Int) = thin_lines_enhancement!(similar(im), im, lh)

"""
    get_ExG(im::AbstractMatrix, l::Real)

Compute ExG image with `l` the brightness threshold.
"""
function get_ExG!(out, im::AbstractMatrix{T}, l::Real) where {T}
    for i in CartesianIndices(out)
        (; r, g, b) = im[i]
        r = float(r)
        g = float(g)
        b = float(b)
        s = r + g + b
        out[i] = (s ≥ l ? g / s : zero(T)) # ExG normalized to (0, 1)
    end
    return out
end
get_ExG(im, l) = get_ExG!(similar(im), im, l)

"""
    get_ExR(im::AbstractMatrix, l::Real)

Compute ExR image with `l` the brightness threshold.
"""
function get_ExR!(out, im::AbstractMatrix{T}, l::Real) where {T}
    for i in CartesianIndices(out)
        (; r, g, b) = im[i]
        r = float(r)
        g = float(g)
        b = float(b)
        s = r + g + b
        out[i] = (s ≥ l ? (r + b / 2.4) / s : zero(T)) # ExR normalized to (0, 1)
    end
    return out
end
get_ExR(im, l) = get_ExR!(similar(im), im, l)

"""
    get_ExGmExR(im::AbstractMatrix, l::Real)

Compute ExG - ExR image with `l` the brightness threshold.
"""
function get_ExGmExR!(out, im::AbstractMatrix{T}, l::Real) where {T}
    for i in CartesianIndices(out)
        (; r, g, b) = im[i]
        r = float(r)
        g = float(g)
        b = float(b)
        s = r + g + b
        out[i] = (s ≥ l ? (g + 1.4 * b / 5.4) / s : zero(T)) # ExG - ExR normalized to (0, 1)
    end
    return out
end
get_ExGmExR(im, l) = get_ExGmExR!(similar(im), im, l)

"""
    remove_small_component!(mask::AbstractArray{Bool}, min_size::Int)

Remove all connected component smaller than a given size.
"""
function remove_small_component!(mask::AbstractArray{Bool}, min_size::Int)
    # Compute connected components
    labels = label_components(mask)
    
    # Get pixel counts for each component label
    counts = component_lengths(labels)
    
    # Filter the mask: retain only pixels belonging to large enough components
    for i in eachindex(mask)
        mask[i] = mask[i] && (counts[labels[i]] ≥ min_size)
    end
    return nothing
end

"""
    keep_largest_component!(mask::AbstractArray{Bool})

Keep the largest connected component.
"""
function keep_largest_component!(mask::AbstractArray{Bool})
    # Compute connected components
    labels = label_components(mask)
    ix = sortperm(component_lengths(labels); rev = true)[1]
    
    # Filter the mask: retain only pixels belonging to large enough components
    for i in eachindex(mask)
        mask[i] = mask[i] && (labels[i] == ix)
    end
    return nothing
end

"""
    get_features!(feats::Array{Gray{N0f8}, 3}, dataset::ROMIScan, p::ROMIMaskParams)

Return the feature needed to mask frames.
"""
function get_features!(feats::Array{Gray{N0f8}, 3}, dataset::ROMIScan, p::ROMIMaskParams)
    @info "Computing features using CPU with $(Threads.nthreads()) threads"
    Threads.@threads for i in axes(feats, 3)
        im = load(joinpath(dataset.images_dir, dataset.images_list[i]))
        img = similar(im)
        copyto!(img, im)
        (p.σ > 0) && imfilter!(img, im, Kernel.gaussian(p.σ))
        if p.use_line
            thin_lines_enhancement!(selectdim(feats, 3, i), Gray.(img), p.lh)
        else
            get_ExG!(selectdim(feats, 3, i), img, p.l)
        end
    end
end

"""
    get_masks!(masks::BitArray{3}, feats::Array{Gray{N0f8}, 3}, p::ROMIMaskParams)

Apply mask parameters and return masked frames.
"""
function get_masks!(masks::BitArray{3}, feats::Array{Gray{N0f8}, 3}, p::ROMIMaskParams)
    @info "Computing masks using CPU with $(Threads.nthreads()) threads"
    Threads.@threads for i in axes(masks, 3)
        mask = selectdim(masks, 3, i)
        mask .= selectdim(feats, 3, i) .> p.t
        remove_small_component!(mask, p.m)
        (p.d > 0) && dilate!(mask, mask, strel_diamond((p.d, p.d)))
    end
    return nothing
end

"""
    get_feat(im::AbstractMatrix, p::ROMIMaskParams)

Compute features from an image.
"""
function get_feat(im::AbstractMatrix, p::ROMIMaskParams)
    feat = similar(Matrix{Gray{N0f8}}, axes(im))
    img = similar(im)
    copyto!(img, im)
    (p.σ > 0) && imfilter!(img, im, Kernel.gaussian(p.σ))
    if p.use_line
        thin_lines_enhancement!(feat, Gray.(img), p.lh)
    else
        get_ExG!(feat, img, p.l)
    end
    return feat
end

"""
    get_mask(feat::AbstractMatrix{Gray{N0f8}}, p::ROMIMaskParams)

Apply mask parameters to a given feature frame.
"""
function get_mask(feat::AbstractMatrix{Gray{N0f8}}, p::ROMIMaskParams)
    mask = feat .> p.t
    (p.m > 0) && remove_small_component!(mask, p.m)
    (p.d > 0) && dilate!(mask, mask, strel_diamond((p.d, p.d)))
    return mask
end

"""
    update_maskedframes!(m::ROMIMaskedFrames, params::ROMIMaskParams, dataset::ROMIScan)

Update the ROMIMaskedFrames `m` given new parameters.
"""
function update_maskedframes!(m::ROMIMaskedFrames, params::ROMIMaskParams, dataset::ROMIScan)
    params_old = m.params
    recompute = (params_old.use_line ? ((params_old.lh != params.lh) || (params_old.σ != params.σ)) : params.use_line)
    if recompute # avoid recomputation of features if possible
        get_features!(m.features, dataset, params)
    end
    recompute |= (params_old.t != params.t) || (params_old.m != params.m) || (params_old.d != params.d)
    if recompute
        get_masks!(m.masks, m.features, params) # do not forget to update the masks
        copyto!(m.params, params) # update the parameters
    end
    return nothing
end