# Installing Artifacts

This document provides instructions for installing ROCm artifacts from TheRock builds.

## Command Options

The script supports the following command-line options, organized by category:

### Source Options

Choose one of these options to specify where to install from:

| Option             | Type   | Description                                                       |
| ------------------ | ------ | ----------------------------------------------------------------- |
| `--input-dir`      | String | Existing TheRock directory to copy from                           |
| `--latest-release` | Flag   | Install the latest nightly release (built daily from main branch) |
| `--release`        | String | Release version from nightly or dev tarballs                      |
| `--run-id`         | String | GitHub CI workflow run ID to install from                         |

### Repository Options

| Option              | Type   | Description                                                                                                    |
| ------------------- | ------ | -------------------------------------------------------------------------------------------------------------- |
| `--amdgpu-family`   | String | AMD GPU family target (required). See [therock_amdgpu_targets.cmake](../../cmake/therock_amdgpu_targets.cmake) |
| `--output-dir`      | Path   | Output directory for TheRock installation (default: `./therock-build`)                                         |
| `--run-github-repo` | String | GitHub repository for CI run ID (default: `GITHUB_REPOSITORY` env var or `ROCm/TheRock`)                       |

### Component Selection

| Option          | Type | Description                                        |
| --------------- | ---- | -------------------------------------------------- |
| `--base-only`   | Flag | Include only base artifacts (minimal installation) |
| `--blas`        | Flag | Include BLAS artifacts                             |
| `--debug-tools` | Flag | Include ROCm debugging tools artifacts             |
| `--fft`         | Flag | Include FFT artifacts                              |
| `--hipdnn`      | Flag | Include hipDNN artifacts                           |
| `--libhipcxx`   | Flag | Include libhipcxx artifacts                        |
| `--miopen`      | Flag | Include MIOpen artifacts                           |
| `--prim`        | Flag | Include primitives artifacts                       |
| `--rand`        | Flag | Include random number generator artifacts          |
| `--rccl`        | Flag | Include RCCL artifacts                             |
| `--rocwmma`     | Flag | Include rocWMMA artifacts                          |
| `--tests`       | Flag | Include test artifacts for enabled components      |

### Utility Options

| Option      | Type | Description                                                |
| ----------- | ---- | ---------------------------------------------------------- |
| `--dry-run` | Flag | Show what would be downloaded without actually downloading |

### Default Behavior

By default for CI workflow retrieval (`--run-id`), all artifacts (excluding test artifacts) will be downloaded. To customize:

- Use `--base-only` for minimal installation (core ROCm only)
- Use component flags (`--blas`, `--fft`, etc.) to select specific libraries
- Add `--tests` to include test artifacts for enabled components

### Selecting Your GPU Family

Select your AMD GPU family from [therock_amdgpu_targets.cmake](https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake#L44-L81). Common families include:

- `gfx110X-all` - RDNA 3 consumer GPUs (RX 7000 series)
- `gfx94X-dcgpu` - MI300 series datacenter GPUs
- `gfx90X-dcgpu` - MI200 series datacenter GPUs

### Finding GitHub Run IDs

To use the `--run-id` option, you need to find the GitHub Actions workflow run ID:

1. Navigate to the [TheRock Actions page](https://github.com/ROCm/TheRock/actions)
1. Click on the "CI" workflow
1. Find a successful run (green checkmark)
1. Click on the run to view details
1. The run ID is the number in the URL: `https://github.com/ROCm/TheRock/actions/runs/[RUN_ID]`

For example, if the URL is `https://github.com/ROCm/TheRock/actions/runs/15575624591`, then the run ID is `15575624591`.

### Finding Release Versions

#### Finding Release Versions Manually

TheRock provides two types of release tarballs:

##### Nightly Tarballs

Nightly tarballs are built daily and follow the naming pattern: `MAJOR.MINOR.aYYYYMMDD`

**To find and use a nightly release:**

1. Visit the [nightly tarball S3 bucket](https://therock-nightly-tarball.s3.amazonaws.com/index.html)
1. Look for files matching your GPU family. Files are named: `therock-dist-linux-{GPU_FAMILY}-{VERSION}.tar.gz`
   - Example: `therock-dist-linux-gfx110X-all-7.11.0a20251124.tar.gz`
1. Extract the version from the filename (the part after the last hyphen, before `.tar.gz`)
   - In the example above, the version is: `7.11.0a20251124`
1. Use this version string with `--release`:
   ```bash
   python build_tools/install_rocm_from_artifacts.py \
       --release 7.11.0a20251124 \
       --amdgpu-family gfx110X-all
   ```

**Version format:** `X.Y.ZaYYYYMMDD`

- `X.Y.Z` = ROCm version (e.g., `7.11.0`)
- `a` = alpha version
- `YYYYMMDD` = build date (e.g., `20251124` = November 24, 2025)

##### Dev Tarballs

Dev tarballs are built from specific commits and follow the naming pattern: `MAJOR.MINOR.PATCH.dev0+{COMMIT_HASH}`

**To find and use a dev release:**

1. Visit the [dev tarball S3 bucket](https://therock-dev-tarball.s3.amazonaws.com/index.html)
1. Look for files matching your GPU family. Files are named: `therock-dist-linux-{GPU_FAMILY}-{VERSION}.tar.gz`
   - Example: `therock-dist-linux-gfx94X-dcgpu-6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9.tar.gz`
1. Extract the version from the filename (the part after the last hyphen, before `.tar.gz`)
   - In the example above, the version is: `6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9`
1. Use this version string with `--release`:
   ```bash
   python build_tools/install_rocm_from_artifacts.py \
       --release 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9 \
       --amdgpu-family gfx94X-dcgpu
   ```

**Version format:** `X.Y.Z.dev0+{HASH}`

- `X.Y.Z` = ROCm version (e.g., `6.4.0`)
- `dev0` = development build indicator
- `{HASH}` = full Git commit hash (40 characters)

> [!TIP]
> You can browse the S3 buckets directly in your browser to see all available versions and GPU families.
> The version string to use with `--release` is always the portion of the filename between the GPU family and `.tar.gz`.

#### Using The Latest Release

To automatically install the latest nightly release without manually finding a version string, use the `--latest-release` flag:

```bash
python build_tools/install_rocm_from_artifacts.py \
    --latest-release \
    --amdgpu-family gfx110X-all
```

To preview what would be downloaded without actually downloading:

```bash
python build_tools/install_rocm_from_artifacts.py \
    --latest-release \
    --amdgpu-family gfx110X-all \
    --dry-run
```

### Fetching Artifacts from Other Repositories

By default, the script fetches artifacts from the repository defined in the `GITHUB_REPOSITORY` environment variable. If that variable is unset, it defaults to `ROCm/TheRock`.

You can specify a different repository using the `--run-github-repo` argument. For example, to fetch artifacts from the `ROCm/rocm-libraries` repository:

```bash
python build_tools/install_rocm_from_artifacts.py \
    --run-id [RUN_ID] \
    --amdgpu-family gfx110X-dgpu \
    --run-github-repo ROCm/rocm-libraries
```

## Installing Per-Commit CI Artifacts Manually

For advanced use cases, you can manually download and flatten CI artifacts using AWS CLI and the `fileset_tool.py` script.

1. Find the CI workflow run that you want to install from. For example, search
   through recent successful runs of the `ci.yml` workflow for `push` events on
   the `main` branch
   [using this page](https://github.com/ROCm/TheRock/actions/workflows/ci.yml?query=branch%3Amain+is%3Asuccess+event%3Apush)
   (choosing a build that took more than a few minutes - documentation only
   changes skip building and uploading).

1. Download the artifacts for that workflow run from S3 using either the
   [AWS CLI](https://aws.amazon.com/cli/) or
   [AWS SDK for Python (Boto3)](https://aws.amazon.com/sdk-for-python/):

   ```bash
   export LOCAL_ARTIFACTS_DIR=~/therock-artifacts
   export LOCAL_INSTALL_DIR=${LOCAL_ARTIFACTS_DIR}/install
   mkdir -p ${LOCAL_ARTIFACTS_DIR}
   mkdir -p ${LOCAL_INSTALL_DIR}

   # Example: https://github.com/ROCm/TheRock/actions/runs/15575624591
   export RUN_ID=15575624591
   export OPERATING_SYSTEM=linux # or 'windows'
   aws s3 cp s3://therock-artifacts/${RUN_ID}-${OPERATING_SYSTEM}/ \
     ${LOCAL_ARTIFACTS_DIR} \
     --no-sign-request --recursive --exclude "*" --include "*.tar.xz"
   ```

1. Flatten the artifacts:

   ```bash
   python build_tools/fileset_tool.py artifact-flatten \
     ${LOCAL_ARTIFACTS_DIR}/*.tar.xz -o ${LOCAL_INSTALL_DIR}
   ```

> [!NOTE]
> The `install_rocm_from_artifacts.py` script automates this process and is the recommended approach for most use cases.

## Usage Examples

### Install from CI Run with BLAS Components

```bash
python build_tools/install_rocm_from_artifacts.py \
    --run-id 19588907671 \
    --amdgpu-family gfx110X-all \
    --blas --tests
```

### Install from Nightly Tarball with Multiple Components

Install RCCL and FFT components from a nightly build for gfx94X:

```bash
python build_tools/install_rocm_from_artifacts.py \
    --release 6.4.0rc20250416 \
    --amdgpu-family gfx94X-dcgpu \
    --rccl --fft --tests
```

### Install from rocm-libraries Repository

Download artifacts from the `ROCm/rocm-libraries` repository:

```bash
python build_tools/install_rocm_from_artifacts.py \
    --run-id 19644138192 \
    --amdgpu-family gfx94X-dcgpu \
    --tests \
    --run-github-repo ROCm/rocm-libraries
```

### Install from Dev Tarball

Install a development build using a commit hash version:

```bash
python build_tools/install_rocm_from_artifacts.py \
    --release 6.4.0.dev0+e015c807437eaf32dac6c14a9c4f752770c51b14 \
    --amdgpu-family gfx110X-all
```

### Preview Before Downloading

Use `--dry-run` with any source to see what would be downloaded:

```bash
python build_tools/install_rocm_from_artifacts.py \
    --run-id 15052158890 \
    --amdgpu-family gfx94X-dcgpu \
    --blas --tests \
    --dry-run
```

## Adding Support for New Components

When you add a new component to TheRock, you will need to update `install_rocm_from_artifacts.py` to allow users to selectively install it.

> [!NOTE]
> You only need to modify `install_rocm_from_artifacts.py` when adding an entirely new component to TheRock.<br>
> Typically if you are adding a new .toml file you will need to add support to `install_rocm_from_artifacts.py`.<br>
> Adding libraries to existing components, (such as including a new library in the `blas` component) requires no script changes.

### Step-by-Step Guide

Here's how to add support for a hypothetical component called `newcomponent`:

#### Step 1: Verify the Artifact is Built

Ensure your component's artifact is properly defined in CMake and built:

```bash
# Check that the artifact is created during build
cmake --build build
ls build/artifacts/newcomponent_*
```

You should see artifacts like:

- `newcomponent_lib_gfx110X`
- `newcomponent_test_gfx110X`
- etc.

#### Step 2: Add Command-Line Argument

Open `build_tools/install_rocm_from_artifacts.py` and add a new argument in the `artifacts_group`:

```python
    artifacts_group.add_argument(
        "--rccl",
        default=False,
        help="Include 'rccl' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--newcomponent",
        default=False,
        help="Include 'newcomponent' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--tests",
        default=False,
        help="Include all test artifacts for enabled libraries",
        action=argparse.BooleanOptionalAction,
    )
```

#### Step 3: Add to Artifact Selection Logic

In the `retrieve_artifacts_by_run_id` function, add your component to the conditional logic:

```python
# filepath: \home\bharriso\Source\TheRock\build_tools\install_rocm_from_artifacts.py
    if args.base_only:
        argv.extend(base_artifact_patterns)
    elif any([args.blas, args.fft, args.miopen, args.prim, args.rand, args.rccl, args.newcomponent]):
        argv.extend(base_artifact_patterns)

        extra_artifacts = []
        if args.blas:
            extra_artifacts.append("blas")
        if args.fft:
            extra_artifacts.append("fft")
        if args.miopen:
            extra_artifacts.append("miopen")
        if args.prim:
            extra_artifacts.append("prim")
        if args.rand:
            extra_artifacts.append("rand")
        if args.rccl:
            extra_artifacts.append("rccl")
        if args.rocprofiler_compute:
            extra_artifacts.append("rocprofiler-compute")
        if args.rocprofiler_systems:
            extra_artifacts.append("rocprofiler-systems")
        if args.newcomponent:
            extra_artifacts.append("newcomponent")

        extra_artifact_patterns = [f"{a}_lib" for a in extra_artifacts]
```

#### Step 4: Update Documentation

Add your new component to the command options table in this document (see the table above).

#### Step 5: Test Your Changes

Test that artifacts can be fetched with your new flag:

```bash
# Test with a CI run
python build_tools/install_rocm_from_artifacts.py \
    --run-id YOUR_RUN_ID \
    --amdgpu-family gfx110X-all \
    --newcomponent --tests
```

#### Step 6: Update Test Configuration (Optional)

If you want to add tests for your component in CI, also update `build_tools/github_actions/fetch_test_configurations.py`. See [Adding Tests](./adding_tests.md) for details.
