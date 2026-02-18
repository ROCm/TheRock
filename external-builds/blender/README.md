# Blender HIP Cycles Rendering Tests

This directory contains tests that verify AMD GPU rendering works correctly
through Blender's Cycles engine using the HIP backend.

## Overview

The test runner (`run_blender_tests.py`) does the following for each scene
listed in `scenes.txt`:

1. Renders the scene using `blender --engine CYCLES -- --cycles-device HIP`
1. Verifies Blender exits successfully and produces output
1. Optionally compares the rendered image against a reference using SSIM and
   MSE metrics

## Prerequisites

- **Blender** (>= 4.0): Official Linux release from <https://www.blender.org/download/>
- **ROCm** with HIP support installed
- **GPU access**: The container or host must have `/dev/kfd` and `/dev/dri` device access
- **Python packages**: `opencv-python-headless`, `scikit-image`, `numpy` (listed in `requirements.txt`)

## Directory Layout

```
external-builds/blender/
├── run_blender_tests.py   # Main test runner
├── requirements.txt       # Python dependencies for image comparison
├── scenes.txt             # Scene manifest (which scenes to render)
└── README.md              # This file
```

During execution, scenes and references are expected at:

```
<scenes-dir>/
├── <scene_name>.blend         # Scene files
├── ref/
│   └── <scene_name>_001.png   # Reference images (optional)
└── out/
    └── <scene_name>_001.png   # Rendered output (created by the runner)
```

## Usage

### Local Testing with Docker

GPU rendering requires proper device access and memory limits:

```bash
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --ulimit memlock=-1:-1 \
  --ipc host \
  <image> bash
```

> **Important**: `--ulimit memlock=-1:-1` is required for HSA (ROCm runtime)
> initialization. Without it, Blender will fail with
> `HSA_STATUS_ERROR_OUT_OF_RESOURCES`.

### Smoke Test (Render Only)

Verify that Blender can render with HIP without comparing images:

```bash
pip install -r external-builds/blender/requirements.txt

python external-builds/blender/run_blender_tests.py \
  --blender-dir /path/to/blender-4.5.6-linux-x64 \
  --scenes-dir /path/to/scenes
```

### With Reference Image Comparison

```bash
python external-builds/blender/run_blender_tests.py \
  --blender-dir /path/to/blender-4.5.6-linux-x64 \
  --scenes-dir /path/to/scenes \
  --ref-dir /path/to/scenes/ref
```

## Scene Manifest Format

`scenes.txt` lists one scene per line:

```
# Comments start with #
<scene_name_without_extension> <frame_number>
```

Example:

```
290skydemo_release 1
classroom 1
bmw27 1
```

## Image Comparison

When a reference directory is provided, rendered images are compared using two
metrics:

- **SSIM** (Structural Similarity Index): Measures perceptual similarity.
  Default threshold: `>= 0.9` (range 0-1, where 1 is identical).
- **MSE** (Mean Squared Error): Measures pixel-level difference.
  Default threshold: `<= 1000` (lower is better).

Both thresholds must be satisfied for a scene to pass. Thresholds can be
adjusted via `--ssim-threshold` and `--mse-threshold` command-line arguments.

## Adding New Scenes

1. Download or create a `.blend` file that uses the Cycles render engine
1. Place it in the scenes directory
1. Add a line to `scenes.txt`: `<filename_without_ext> <frame_to_render>`
1. Optionally generate a reference image from a known-good render and place it
   in `ref/<filename>_<frame_padded>.png`
