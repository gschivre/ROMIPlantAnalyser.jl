# Convert `*_ROMIAnglesAndInternodes.jls` results to and from a plain format that only holds Base types
# (Dict, Vector, Tuple, numbers, String, Symbol), so that saved results survive changes of the package
# types (added/removed fields).
#
# Usage (from the package folder):
#   julia --project scripts/results_io.jl dump    <experiment_folder_or_jls> [more ...]
#   julia --project scripts/results_io.jl restore <experiment_folder_or_plain_jls> [more ...]
#
# `dump` writes `<name>_ROMIAnglesAndInternodes.plain.jls` next to each results file (and checks that it
# converts back to identical results). Run it with the package version that wrote the results.
# `restore` rebuilds `<name>_ROMIAnglesAndInternodes.jls` from the plain file with the current package
# version; an existing results file is first moved to `.jls.bak` (an existing backup is never overwritten).
# Fields missing from the plain file take their default value (keyword constructors of `@kwdef` types);
# fields that no longer exist are dropped with a warning.
#
# The conversion only uses reflection (fieldnames/getfield), so this script does not depend on the
# definition of the types it converts.

ENV["ROMI_SKIP_WARMUP"] = "true" # no GLMakie window needed
using ROMIPlantAnalyser, Serialization, GeometryBasics, Dates
const RPA = ROMIPlantAnalyser

const FORMAT = "ROMIResults-plain"
const FORMAT_VERSION = 1
const TYPE_KEY = "__type__"

# ------------------------------------------------------------------------------------------------
# package types -> plain
to_plain(x::Union{Number, AbstractString, Symbol, Nothing}) = x
to_plain(x::Tuple) = map(to_plain, x)
to_plain(x::AbstractArray) = [to_plain(v) for v in x] # comprehension: always a Base Array (map keeps static arrays)
to_plain(x::AbstractDict) = Dict{Any, Any}(to_plain(k) => to_plain(v) for (k, v) in x)
to_plain(x::Rect3) = Dict{String, Any}(TYPE_KEY => "Rect3d", "origin" => Vector{Float64}(x.origin),
                                       "widths" => Vector{Float64}(x.widths))
function to_plain(x::T) where {T}
    parentmodule(T) === RPA || error("don't know how to store a $(T) as plain data")
    return Dict{String, Any}(TYPE_KEY => String(nameof(T)),
                             (String(f) => to_plain(getfield(x, f)) for f in fieldnames(T))...)
end

# the plain data must only hold Base/Core types so it can be read back without this package
function check_plain(x, path = "")
    if x isa AbstractDict
        foreach(((k, v),) -> check_plain(v, "$path.$k"), x)
    elseif x isa Union{Tuple, AbstractArray}
        foreach(((i, v),) -> check_plain(v, "$path[$i]"), enumerate(x))
    end
    parentmodule(typeof(x)) in (Base, Core) || error("non-Base value at $path :: $(typeof(x))")
    return nothing
end

# ------------------------------------------------------------------------------------------------
# plain -> package types
from_plain(x) = x
from_plain(x::Tuple) = map(from_plain, x)
from_plain(x::AbstractArray) = map(from_plain, x)
function from_plain(d::AbstractDict)
    haskey(d, TYPE_KEY) || return Dict(from_plain(k) => from_plain(v) for (k, v) in d)
    name = d[TYPE_KEY]
    name == "Rect3d" && return Rect3d(Vec3d(d["origin"]...), Vec3d(d["widths"]...))

    T = getfield(RPA, Symbol(name))
    fields = fieldnames(T isa UnionAll ? Base.unwrap_unionall(T) : T)
    stored = Dict(Symbol(k) => from_plain(v) for (k, v) in d if k != TYPE_KEY)
    for k in setdiff(keys(stored), fields)
        @warn "$name: dropping field `$k` that the current type no longer has"
        delete!(stored, k)
    end
    missing_fields = setdiff(fields, keys(stored))
    if hasmethod(T, Tuple{}, Tuple(keys(stored))) # @kwdef type: missing fields take their default
        isempty(missing_fields) || @info "$name: field(s) $(join(missing_fields, ", ")) set to their default value"
        return T(; stored...)
    end
    isempty(missing_fields) || error("$name: field(s) $(join(missing_fields, ", ")) missing and the type has no default values")
    return T((stored[f] for f in fields)...)
end

# ------------------------------------------------------------------------------------------------
const SUFFIX = "_ROMIAnglesAndInternodes.jls"
const PLAIN_SUFFIX = "_ROMIAnglesAndInternodes.plain.jls"

results_path(path, suffix) = isdir(path) ? joinpath(path, basename(rstrip(path, ('/', '\\'))) * suffix) : path

function dump_results(path::String)
    f = results_path(path, SUFFIX)
    (isfile(f) && endswith(f, SUFFIX)) || error("no results file at $f")
    results = load_results(f)
    plain = Dict{String, Any}(id => to_plain(r) for (id, r) in results)

    check_plain(plain)

    # round-trip check: the plain data must convert back to the same results
    back = Dict{String, Any}(id => to_plain(from_plain(p)) for (id, p) in plain)
    isequal(back, plain) || error("round-trip check failed for $f, nothing written")

    out = replace(f, SUFFIX => PLAIN_SUFFIX)
    serialize(out, Dict{String, Any}("format" => FORMAT, "format_version" => FORMAT_VERSION,
        "package_version" => string(pkgversion(RPA)), "date" => string(now()), "results" => plain))
    @info "Dumped $(length(plain)) plant(s) to $out"
end

# results from the content of a plain results file
function results_from_plain(data)
    (data isa AbstractDict && get(data, "format", nothing) == FORMAT) || error("not plain results data")
    return Dict{String, ROMIResults}(id => from_plain(p) for (id, p) in data["results"])
end

function restore_results(path::String)
    f = results_path(path, PLAIN_SUFFIX)
    (isfile(f) && endswith(f, PLAIN_SUFFIX)) || error("no plain results file at $f")
    results = results_from_plain(deserialize(f))

    out = replace(f, PLAIN_SUFFIX => SUFFIX)
    if isfile(out)
        bak = out * ".bak"
        isfile(bak) ? @warn("Backup $bak already exists, overwriting $out without a new backup") : mv(out, bak)
    end
    serialize(out, results)
    @info "Restored $(length(results)) plant(s) to $out"
end

function main(args)
    (length(args) >= 2 && args[1] in ("dump", "restore")) ||
        error("usage: julia --project scripts/results_io.jl dump|restore <experiment_folder_or_file> ...")
    action = (args[1] == "dump" ? dump_results : restore_results)
    for p in args[2:end]
        try
            action(p)
        catch err
            @error "$(args[1]) failed for $p" exception = err
        end
    end
end

(abspath(PROGRAM_FILE) == @__FILE__()) && main(ARGS)
