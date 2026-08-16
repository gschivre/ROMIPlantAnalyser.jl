struct ROMICamera{I <: Integer, F <: AbstractFloat}
    # sensor width and height
    width::I
    height::I

    # focal length
    fx::F
    fy::F

    # principal point 
    cx::F
    cy::F

    # radial distortion
    k1::F
    k2::F

    # tangential distortion
    p1::F
    p2::F
end

struct ROMIFrame{I <: Integer, F <: AbstractFloat}
    # 3×3 Camera Rotation matrix (sensor_from_world)
    R::Mat3{F} # use GeometryBasics Mat ≡ SMatrix

    # 3-element Translation vector
    t::Vec3{F} # use GeometryBasics Vec ≡ SVector

    # camera model
    camera::ROMICamera{I, F}
end

"""
    is_valid_scan_folder(dir::String) -> Bool

Checks if a single folder complies with the expected ROMI scan architecture:
  - Contains `images/` and `metadata/images/` directories.
  - Contains at least one `00000_rgb.jpg` file.
  - Every matching image has a corresponding `xxxxx_rgb.json` in `metadata/images/`.
"""
function is_valid_scan_folder(dir::String)
    !isdir(dir) && return false

    images_dir = joinpath(dir, "images")
    metadata_dir = joinpath(dir, "metadata", "images")
    (!isdir(images_dir) || !isdir(metadata_dir)) && return false

    pattern = r"^\d{5}_rgb\.jpg$"
    images = filter(f -> occursin(pattern, f), readdir(images_dir))

    # Must contain at least one valid image file
    isempty(images) && return false

    # Verify that every single image has its matching .json metadata file
    return all(images) do img
        json_name = replace(img, ".jpg" => ".json")
        isfile(joinpath(metadata_dir, json_name))
    end
end

struct ROMIScan
    # project folder architecture
    project_dir::String
    plant_id::String
    images_dir::String
    metadata_dir::String

    # images list and CNC approximate pose
    images_list::Vector{String}
    cnc_xyz::Vector{Point{3, Float64}}

    # colmap files and directories
    db_path::String
    pairs_path::String
    poses_path::String
    sparse_dir::String

    # colmap parameters
    camera_model::String
    std_xyz::NTuple{3, Float64} # x, y, and z estimated pose error in mm

    # frames
    frames::Vector{ROMIFrame{Int, Float64}}

    function ROMIScan(project_dir::AbstractString;
                camera_model::String = "OPENCV",
                std_xyz::NTuple{3, Float64} = (1.0, 1.0, 1.0))
        # check that the folder is a valid ROMI scan folder
        @assert is_valid_scan_folder(project_dir) "$project_dir is not a valid ROMI scan folder!"
        
        # get the plant id
        abs_project_dir = abspath(project_dir)
        plant_id = basename(rstrip(abs_project_dir, ('/', '\\')))

        # read all files in the images directory
        images_dir = joinpath(abs_project_dir, "images")
        all_files = readdir(images_dir)
        
        # filter for specific pattern: 5 digits followed by _rgb.jpg
        pattern = r"^\d{5}_rgb\.jpg$"
        images_list = filter(f -> occursin(pattern, f), all_files)
        sort!(images_list)

        # extract the corresponding CNC approximate poses
        metadata_dir = joinpath(abs_project_dir, "metadata", "images")
        cnc_xyz = Vector{Point3d}(undef, length(images_list))
        colmap_dir = mkpath(joinpath(abs_project_dir, "colmap"))
        open(joinpath(colmap_dir, "$(plant_id)_poses.txt"), "w") do io
            for (i, img_name) in Iterators.enumerate(images_list)
                json_file = joinpath(metadata_dir, replace(img_name, ".jpg" => ".json"))
                metadata = JSON.parsefile(json_file)
                cnc_xyz[i] = Point3d(metadata["approximate_pose"][1:3])
                println(io, join((img_name, cnc_xyz[i].data...), " "))
            end
        end

        new(
            abs_project_dir,
            plant_id,
            images_dir,
            metadata_dir,
            images_list,
            cnc_xyz,
            joinpath(colmap_dir, "$(plant_id)_colmap_database.db"),
            joinpath(colmap_dir, "$(plant_id)_images_pairs.txt"),
            joinpath(colmap_dir, "$(plant_id)_poses.txt"),
            joinpath(colmap_dir, "sparse"),
            camera_model,
            std_xyz,
            Vector{ROMIFrame{Int, Float64}}(undef, length(images_list))
        )
    end
end
Base.length(d::ROMIScan) = length(d.frames)
Base.show(io::IO, d::ROMIScan) = print(io, "ROMIScan with ", length(d), " frames")