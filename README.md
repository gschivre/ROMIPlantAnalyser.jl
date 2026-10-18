# ROMIPlantAnalyser

[![Build Status](https://github.com/gschivre/ROMIPlantAnalyser.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gschivre/ROMIPlantAnalyser.jl/actions/workflows/CI.yml?query=branch%3Amain)

ROMIPlantAnalyser is an interactive desktop app, written in [Julia](https://julialang.org) with
[GLMakie](https://docs.makie.org), for measuring plant architecture (**phyllotaxis**:
divergence angles, internode lengths, fruit orientation) from scans made with the
[ROMI](https://github.com/romi) setup.

It reads the images recorded by the [plant-imager](https://github.com/romi/plant-imager) and
reworks the [plant-3d-vision](https://github.com/romi/plant-3d-vision) reconstruction pipeline so
that you can check and correct every step by hand:

1. **Camera poses**: taken from an existing plant-3d-vision `Colmap` task, or computed by a local
   [COLMAP](https://colmap.github.io) install using the CNC poses as priors.
2. **Bounding box and masks**: set interactively on the images.
3. **Volume**: Bayesian voxel carving with optional GMRF smoothing, on the GPU if one is available.
4. **Skeleton**: the stem and fruits are extracted automatically, then you can move the stem
   root and top and add, remove or re-route fruits.
5. **Measurements**: divergence angles, internodes, fruit lengths and angles, exported as CSV/JSON.

A second app, `romi_launch_skeledit()`, skips steps 1–3 and only does skeleton editing on top of a
volume (`Voxels` task) that plant-3d-vision has already computed.

---

## Contents

- [Installation](#installation)
  - [1. Install Julia](#1-install-julia)
  - [2. Install ROMIPlantAnalyser](#2-install-romiplantanalyser)
  - [3. (Optional) Install COLMAP](#3-optional-install-colmap)
  - [4. (Optional) NVIDIA GPU](#4-optional-nvidia-gpu)
- [Input data](#input-data)
- [Usage](#usage)
  - [Starting the app](#starting-the-app)
  - [Full pipeline: `romi_launch`](#full-pipeline-romi_launch)
  - [Skeleton editing only: `romi_launch_skeledit`](#skeleton-editing-only-romi_launch_skeledit)
  - [Results and export](#results-and-export)
- [Troubleshooting](#troubleshooting)
- [Running the tests](#running-the-tests)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Installation

### 1. Install Julia

**ROMIPlantAnalyser needs Julia 1.12 or newer.** Install Julia with **juliaup**, the official
installer and version manager, by following the instructions for your system on the Julia
website: **<https://julialang.org/install/>**. More details on juliaup are in its repository:
<https://github.com/JuliaLang/juliaup>.

Then check the version in a new terminal:

```bash
julia --version # should print 1.12.x or newer
```

> **New to Julia?** Typing `julia` in a terminal starts the *REPL* (an interactive prompt,
> `julia>`). Pressing `]` at the start of a line switches it to the *package manager* prompt
> (`pkg>`), and pressing `Backspace` switches back. Type `exit()` or press `Ctrl+D` to quit.

### 2. Install ROMIPlantAnalyser

The package is not in the Julia General registry yet, so you install it from GitHub. We suggest
putting it in its own named environment (`@romi` here) so it doesn't clash with other Julia
projects:

```bash
julia --project=@romi
```

then, at the `julia>` prompt, press `]` and run:

```
pkg> add https://github.com/gschivre/ROMIPlantAnalyser.jl
```

You can also do it in one command from the terminal:

```bash
julia --project=@romi -e 'using Pkg; Pkg.add(url="https://github.com/gschivre/ROMIPlantAnalyser.jl")'
```

The first install takes a while (several minutes to half an hour): Julia downloads and
precompiles every dependency (GLMakie, CUDA, …). This happens only once.

The Python packages the app needs are installed **automatically** into a private environment
through [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) the first time the package is
loaded. You don't need an existing Python install.

To update later:

```
pkg> update ROMIPlantAnalyser
```

### 3. (Optional) Install COLMAP

You only need this step if you want the app to **compute the camera poses itself**. Skip it if you
already ran the `Colmap` task of plant-3d-vision on your scans (use `from_romi = true`, see below)
or if you only use `romi_launch_skeledit()`.

Follow the official COLMAP installation instructions for your system:
**<https://colmap.github.io/install.html>**. The `colmap` executable must be on your `PATH`. Check
with:

```bash
colmap help
```

The pipeline uses `pose_prior_mapper` and pose priors, which only exist in **recent COLMAP
releases**. Use a current version: an old distribution package (for example from `apt`) will
probably not work. Build with CUDA if you want GPU feature extraction and matching.

### 4. (Optional) NVIDIA GPU

If [CUDA.jl](https://cuda.juliagpu.org) finds a working NVIDIA GPU (`CUDA.functional() == true`),
the volume reconstruction runs on it automatically. Otherwise it runs on the CPU, which is slower
but works. You only need a recent NVIDIA driver: CUDA.jl downloads the CUDA toolkit itself.

---

## Input data

The app works on **plant-imager scan folders**, one per stem. A scan folder is valid if it
contains:

```
<plant_id>/
├── images/
│   ├── 00000_rgb.jpg
│   ├── 00001_rgb.jpg
│   └── ...
└── metadata/
    └── images/
        ├── 00000_rgb.json      # must contain "approximate_pose" (CNC x, y, z, ...)
        ├── 00001_rgb.json
        └── ...
```

You can drop into the app either:

- **one scan folder**: a single stem is analysed;
- an **experiment folder**: a folder that contains several scan folders. You then go through the
  stems one by one, and all results are saved in a single file.

Some modes need extra plant-3d-vision outputs in each scan folder:

| Mode                                | Extra content needed in the scan folder                                  |
|-------------------------------------|--------------------------------------------------------------------------|
| `romi_launch()` (local COLMAP)      | none                                                                      |
| `romi_launch(from_romi = true)`     | a `Colmap_*/` folder with `cameras.json` and `images.json`               |
| `romi_launch_skeledit()`            | a `Voxels_*/Voxels.tiff` file and `metadata/Voxels_*.json`               |

---

## Usage

### Starting the app

Start Julia **with several threads** (COLMAP and the reconstruction run in the background, which
keeps the window responsive) in the environment you installed into:

```bash
julia --project=@romi
```

then:

```julia
julia> using ROMIPlantAnalyser
julia> romi_launch()
```

The first `using ROMIPlantAnalyser` in a session takes some time: the package runs a small
synthetic reconstruction to warm up the GPU kernels and the OpenGL shaders, so the app is fast
afterwards.

When the window opens, **drag and drop** a scan folder or an experiment folder onto it, then click
**Start Computation**. If the experiment already has saved results, the app opens the first stem
that hasn't been analysed yet.

### Full pipeline: `romi_launch`

```julia
romi_launch(; from_romi = false, use_pairs = false, xyz_error = 3.0)
```

| Keyword     | Default | Meaning                                                                                                   |
|-------------|---------|-----------------------------------------------------------------------------------------------------------|
| `from_romi` | `false` | `true`: read camera poses from the plant-3d-vision `Colmap_*` folder. `false`: run the local COLMAP.  |
| `use_pairs` | `false` | `true`: only match consecutive images (faster). `false`: exhaustive matching (more robust).            |
| `xyz_error` | `3.0`   | Expected error (mm) of the CNC positions, used as the pose-prior standard deviation in COLMAP.          |

COLMAP runs only the first time you open a scan. Its outputs are kept in `<plant_id>/colmap/` and
the recovered poses in `<plant_id>/<plant_id>_ROMIScan.jls`, so opening the scan again is
immediate. If COLMAP fails on the GPU, it is retried automatically on the CPU.

#### Navigation bar

When a scan is loaded, a navigation bar sits at the top of the window. What it shows depends on
what you dropped:

- **A single scan folder**: only **Save** and **Exit**.
- **An experiment folder**: also a **stem selector** (drop-down list of the scans), **← Prev**,
  the current position (`Plant i / n`) and **Next →**, to move between the stems of the
  experiment.

Below it, three tabs, which you normally go through in order.

#### Tab 1: Bounding Box & Mask

- **Left / Right arrow keys**: go to the previous / next image.
- **Bounding Box Parameters**: move (origin) and resize (width) the reconstruction box around the
  camera centre. Its projection is drawn in yellow on the current image. Set the **voxel size**
  (mm) in the text box.
- **Features Parameters**: pick how the plant is detected, either **Line Enhancement** (good for
  thin stems on a dark background) or **Excess Green** (colour-based), and tune it. Turn on
  **Show features** to see the result, then click **Update Features**.
- **Mask Parameters**: threshold `t`, minimum object size `m` and dilation `d`. Turn on
  **Masks overlay** to see the masks in red, then click **Update Masks**.
- **Update Volume** recomputes the 3D volume from the current box and masks.

#### Tab 2: Volume

- **Smoothing penalty λ**: strength of the GMRF smoothing of the occupancy probabilities
  (0 = no smoothing). Click **Apply Smoothing**.
- **Isovalue t**: occupancy-probability threshold that defines the plant surface. Click
  **Update Skeleton** to rebuild the mesh and the skeleton.

#### Tab 3: Skeleton

The sliders set the smoothing of the stem and fruit curves (`σstem`, and `σbranch` for the fruits)
and the **tip prominence `h`**, which controls how readily fruit tips are detected. The sidebar
shows the number of fruits found (labelled *Branches* in the app).

Editing uses keyboard modes. Every edit is confirmed with **Alt + Left click** (a plain click
still rotates the view):

| Key            | Mode   | Alt + Left click does…                                                         |
|----------------|--------|---------------------------------------------------------------------------------|
| `r`            | Root   | move the root (base) of the stem                                              |
| `t`            | Top    | move the top of the stem                                                       |
| `Space`        | Add    | add a fruit from the clicked tip                                               |
|                |        | `Ctrl + Alt + click`: first choose where the fruit joins the stem             |
|                |        | `Shift + Alt + click`: add waypoints the fruit path must go through           |
| `Backspace`    | Delete | remove the fruit under the cursor                                              |
| `Esc`          | —      | go back to viewing mode                                                        |

Pressing the same key again also goes back to viewing mode. A preview of the edit is shown while
you hover the mesh.

**Edit in this order: stem root first, then stem top, then fruits.** Moving the root recomputes
the stem and re-detects all fruit tips, so any fruit you added, removed or re-routed before is
lost. Moving the top re-attaches every fruit to the new stem. Check the root and the top first,
then correct the fruits.

#### Saving

- **Save** stores the measurements *and* every parameter (box, masks, volume, skeleton, stem
  root/top, fruit tips) for the current stem. When you reopen it later, you get exactly the same
  reconstruction back and can keep correcting it.
- **Next →** (experiment folders only) saves the current stem and opens the next one.
- If the stem already has results, the app asks whether to overwrite them.

### Skeleton editing only: `romi_launch_skeledit`

If you already ran plant-3d-vision up to the `Voxels` task (on a cluster, for example), you can
skip pose estimation, masking and volume reconstruction:

```julia
julia> using ROMIPlantAnalyser
julia> romi_launch_skeledit()
```

The app reads the bounding box, voxel size and binary volume from the ROMI outputs and opens
straight on the **Skeleton** tab described above, with the same navigation bar. It uses the same
results file as `romi_launch`. To protect your work, a stem that was analysed with the full
`romi_launch` pipeline is **not** overwritten from this app.

### Results and export

Results are saved in the folder you dropped into the app (the scan folder or the experiment
folder), as:

```
<folder_name>_ROMIAnglesAndInternodes.jls
```

This is a Julia binary file. To export it to readable formats:

```julia
julia> export_results("/path/to/experiment_folder") # or the path to the .jls file
```

This writes `<folder_name>_ROMIAnglesAndInternodes.csv` and `.json` next to it. The CSV has one
row per fruit, ordered from the bottom of the stem upwards:

| Column                         | Meaning                                                           |
|--------------------------------|-------------------------------------------------------------------|
| `plant_id`                     | scan folder name (only on each stem's first row)                  |
| `fruit_id`                     | rank of the fruit along the stem                                  |
| `fruit_length (mm)`            | length of the fruit                                               |
| `polar (rad)`, `azimuth (rad)` | fruit orientation, with standard errors (°) and their covariance  |
| `flagged`                      | `true` if the fit was too poor to trust (angles are then `NaN`)   |
| `internode (mm)`               | arc length along the stem from the previous fruit                 |
| `div_angle (°)`                | divergence angle from the previous fruit                          |
| `chirality`                    | phyllotaxis direction: `cw`, `ccw` or `ambiguous` (first row only)|

Fruit angles are measured slightly away from the stem, at twice the fruit smoothing scale
(`2 × σbranch`, 3 mm with the default settings), because the reconstructed fruit curve first
runs across the stem. The azimuth is accurate, but the polar angle of fruits that are not at 90°
from the stem is still biased towards 90°. Results saved by earlier versions (measured at the
stem) still open: reopen a stem and click **Save** to update its angles.

To work with the results directly in Julia:

```julia
julia> res = load_results("/path/to/experiment_folder") # Dict{String, ROMIResults}
julia> res["plant_id"].result.div_angles
```

---

## Troubleshooting

- **`colmap` executable was not found in system PATH**: install COLMAP (see
  [step 3](#3-optional-install-colmap)) and make sure `colmap help` works in the same terminal you
  start Julia from. Or use `romi_launch(from_romi = true)` if your scans have a plant-3d-vision
  `Colmap_*` folder.
- **The window freezes while COLMAP or the viewer is loading**: start Julia with
  `--threads=auto`.
- **"Invalid folder structure!" after dropping a folder**: check the layout described in
  [Input data](#input-data). Images must be named `NNNNN_rgb.jpg`, and each one needs a matching
  `.json` file.
- **Slow `using ROMIPlantAnalyser`**: set `ROMI_SKIP_WARMUP=true` to skip the warm-up. Loading
  is faster, but the first actions in the app will be slower.
- **No window / OpenGL error on a remote machine**: GLMakie needs a real display and OpenGL 3.3+.
  It does not work over plain SSH without X forwarding or a virtual display.

---

## Acknowledgements

This README and the test suite (`test/`) were written with the help of
[Claude Code](https://claude.com/claude-code) (Anthropic's AI coding assistant).

## License

MIT, see [LICENSE](LICENSE).
