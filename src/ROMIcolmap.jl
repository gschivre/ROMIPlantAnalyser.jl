using GeometryBasics, Serialization

# Initialize Python dependencies via PythonCall
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/home/admingeoffrey/Documents/Datas/Python/Miniconda3/envs/pycolmap/bin/python3"
ENV["JULIA_CONDAPKG_EXE"] = "/home/admingeoffrey/Documents/Datas/Python/Miniconda3/envs/pycolmap/bin"
using PythonCall
const pycolmap = pyimport("pycolmap")
const np = pyimport("numpy")

# load ROMI types as this code will be launch on a separate thread
include("ROMITypes.jl")
using .ROMITypes

"""
    write_image_pairs(dataset::ROMIScan)

Write a .txt file of pairs of images to restrict the colmap features matching to neighboring frames.
"""
function write_image_pairs(dataset::ROMIScan)
    @info "Writing image pairs for colmap feature matching to $(basename(dataset.pairs_path))"
    images_list = dataset.images_list
    num_images = length(images_list)
    open(dataset.pairs_path, "w") do io
        for i in eachindex(images_list)
            current_img = images_list[i]
            next_idx = (i % num_images) + 1
            next_img = images_list[next_idx]
            println(io, "$current_img $next_img")
        end
    end
end

"""
    extract_pose_priors(dataset::ROMIScan)

Inject the camera pose priors into the COLMAP database using `pycolmap` bindings.
"""
function extract_pose_priors(dataset::ROMIScan)
    @info "Add pose priors to $(basename(dataset.db_path))"

    # Open the pycolmap database context manager safely
    PythonCall.GC.disable()
    db = pycolmap.Database.open(dataset.db_path)
    try
        # Construct diagonal covariance matrix in NumPy
        std_xyz = dataset.std_xyz
        cov_matrix = np.diag([std_xyz[1]^2, std_xyz[2]^2, std_xyz[3]^2])

        db.clear_pose_priors() # Remove existing records
        
        for (i, img_name) in Iterators.enumerate(dataset.images_list)            
            if !pyconvert(Bool, db.exists_image(img_name))
                @warn "Image '$img_name' not found in database. Skipping prior assignment."
                continue
            end

            # Retrieve core target context matching Python structural design
            image = db.read_image_with_name(img_name)

            # Convert CNC xyz to NumPy
            position_np = np.array(dataset.cnc_xyz[i].data).reshape(3, 1)
            
            # Create the PosePrior instance
            pp = pycolmap.PosePrior()
            pp.corr_data_id = image.data_id
            pp.position = position_np
            pp.position_covariance = cov_matrix
            pp.coordinate_system = pycolmap.PosePriorCoordinateSystem(1) # cartesian coordinate system
            
            # Write to the colmap database
            db.write_pose_prior(pp)
        end
    finally
        db.close()
        PythonCall.GC.enable()
    end
end

"""
    run_colmap(dataset::ROMIScan; use_GPU::Bool = true)

Run colmap and align the colmap model to approximate pose.
"""
function run_colmap(dataset::ROMIScan; use_GPU::Bool = true)
    @info "Starting colmap CLI workflow for $(dataset.plant_id)"
    
    # Feature Extraction
    run(`colmap feature_extractor \
            --database_path $(dataset.db_path) \
            --image_path $(dataset.images_dir) \
            --ImageReader.camera_model $(dataset.camera_model) \
            --ImageReader.single_camera 1 \
            --FeatureExtraction.use_gpu $use_GPU`)

    # Inject pose priors
    extract_pose_priors(dataset)

    # Write image pairs
    write_image_pairs(dataset)

    # Matching
    run(`colmap matches_importer \
            --database_path $(dataset.db_path) \
            --match_list_path $(dataset.pairs_path) \
            --FeatureMatching.use_gpu $use_GPU \
            --FeatureMatching.guided_matching 1`)

    # Sparse Reconstruction mapping using priors
    mkpath(dataset.sparse_dir)
    run(`colmap pose_prior_mapper \
            --database_path $(dataset.db_path) \
            --image_path $(dataset.images_dir) \
            --output_path $(dataset.sparse_dir) \
            --use_robust_loss_on_prior_position 1`)
end

"""
    extract_frames!(dataset::ROMIScan)

Extract rotation matrix and translation vector and construct frames.
"""
function extract_frames!(dataset::ROMIScan)
    @info "Extracting pose information from colmap for $(dataset.plant_id)"
    PythonCall.GC.disable()
    rec = pycolmap.Reconstruction()
    rec.read(joinpath(dataset.sparse_dir, argmax(d -> parse(Int, d), readdir(dataset.sparse_dir))))
    try
        # extract camera information
        colmap_cam = rec.camera(1)
        cam_model = pyconvert(String, rec.camera(1).model_name)
        w = pyconvert(Int, colmap_cam.width)
        h = pyconvert(Int, colmap_cam.height)
        fx = pyconvert(Float64, colmap_cam.focal_length_x)
        fy = pyconvert(Float64, colmap_cam.focal_length_y)
        cx = pyconvert(Float64, colmap_cam.principal_point_x)
        cy = pyconvert(Float64, colmap_cam.principal_point_y)
        if pyconvert(Bool, colmap_cam.is_undistorted())
            cam = ROMICamera(w, h, fx, fy, cx, cy,
                        0.0, 0.0, 0.0, 0.0) # the camera distortion is 0
        elseif cam_model == "SIMPLE_RADIAL"
            k1 = pyconvert(Float64, colmap_cam.params[colmap_cam.extra_params_idxs()][0])
            cam = ROMICamera(w, h, fx, fy, cx, cy,
                        k1, 0.0, 0.0, 0.0)
        elseif cam_model == "RADIAL"
            k1, k2 = pyconvert(Vector{Float64}, colmap_cam.params[colmap_cam.extra_params_idxs()])
            cam = ROMICamera(w, h, fx, fy, cx, cy,
                        k1, k2, 0.0, 0.0)
        elseif cam_model == "OPENCV"
            k1, k2, p1, p2 = pyconvert(Vector{Float64}, colmap_cam.params[colmap_cam.extra_params_idxs()])
            cam = ROMICamera(w, h, fx, fy, cx, cy,
                        k1, k2, p1, p2)
        end

        # extract frames
        for (i, img_name) in Iterators.enumerate(dataset.images_list)
            image = rec.find_image_with_name(img_name)
            colmap_frame = rec.frame(image.frame_id)
            R = pyconvert(Mat3d, colmap_frame.sensor_from_world(colmap_cam.sensor_id).rotation.matrix())
            t = pyconvert(Vec3d, colmap_frame.sensor_from_world(colmap_cam.sensor_id).translation)
            dataset.frames[i] = ROMIFrame(R, t, cam)
        end
    finally
        PythonCall.GC.enable()
    end
end

# main execution
if length(ARGS) == 1 # run colmap and the pose extraction
    dataset = ROMIScan(ARGS[1])
    run_colmap(dataset)
    extract_frames!(dataset)
    serialize(joinpath(dataset.project_dir, "$(dataset.plant_id)_ROMIScan.jls"), dataset)
elseif length(ARGS) > 1
    xyz_error = (length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 1.0)
    dataset = ROMIScan(ARGS[1]; std_xyz = ntuple(i -> xyz_error, 3))
    if parse(Bool, ARGS[2])
        use_gpu = (length(ARGS) > 2 ? parse(Bool, ARGS[3]) : true)
        run_colmap(dataset; use_GPU = use_gpu)
    end
    extract_frames!(dataset)
    serialize(joinpath(dataset.project_dir, "$(dataset.plant_id)_ROMIScan.jls"), dataset)
else
    println("Usage: julia ROMIcolmap.jl /path/to/project_folder [runcolamp = 1] [use_gpu = 1] [xyz_error = 1.0]")
    exit(1)
end