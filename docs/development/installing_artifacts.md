# Installing Artifacts

This document provides instructions for installing ROCm artifacts from TheRock builds.

## Command Options

The script supports the following command-line options:

| Option            | Type   | Description                                                            |
| ----------------- | ------ | ---------------------------------------------------------------------- |
| `--amdgpu-family` | String | AMD GPU family target (required)                                       |
| `--base-only`     | Flag   | Include only base artifacts (minimal installation)                     |
| `--blas`          | Flag   | Include BLAS artifacts                                                 |
| `--fft`           | Flag   | Include FFT artifacts                                                  |
| `--input-dir`     | String | Existing TheRock directory to copy from                                |
| `--miopen`        | Flag   | Include MIOpen artifacts                                               |
| `--output-dir`    | Path   | Output directory for TheRock installation (default: `./therock-build`) |
| `--prim`          | Flag   | Include primitives artifacts                                           |
| `--rand`          | Flag   | Include random number generator artifacts                              |
| `--rccl`          | Flag   | Include RCCL artifacts                                                 |
| `--release`       | String | Release version from nightly or dev tarballs                           |
| `--run-id`        | String | GitHub CI workflow run ID to install from                              |
| `--tests`         | Flag   | Include test artifacts for enabled components                          |

### Finding GitHub Run IDs

To use the `--run-id` option, you need to find the GitHub Actions workflow run ID:

1. Navigate to the [TheRock Actions page](https://github.com/ROCm/TheRock/actions)
1. Click on the "CI" workflow
1. Find a successful run (green checkmark)
1. Click on the run to view details
1. The run ID is the number in the URL: `https://github.com/ROCm/TheRock/actions/runs/[RUN_ID]`

For example, if the URL is `https://github.com/ROCm/TheRock/actions/runs/15575624591`, then the run ID is `15575624591`.

### Finding Release Versions

TheRock provides two types of release tarballs:

#### Nightly Tarballs

Nightly tarballs are built daily and follow the naming pattern: `MAJOR.MINOR.PATCHrcYYYYMMDD`

**To find and use a nightly release:**

1. Visit the [nightly tarball S3 bucket](https://therock-nightly-tarball.s3.amazonaws.com/)
1. Look for files matching your GPU family. Files are named: `therock-dist-linux-{GPU_FAMILY}-{VERSION}.tar.gz`
   - Example: `therock-dist-linux-gfx110X-dgpu-6.4.0rc20250514.tar.gz`
1. Extract the version from the filename (the part after the last hyphen, before `.tar.gz`)
   - In the example above, the version is: `6.4.0rc20250514`
1. Use this version string with `--release`:
   ```bash
   python build_tools/install_rocm_from_artifacts.py \
       --release 6.4.0rc20250514 \
       --amdgpu-family gfx110X-dgpu
   ```

**Version format:** `X.Y.ZrcYYYYMMDD`

- `X.Y.Z` = ROCm version (e.g., `6.4.0`)
- `rc` = release candidate indicator
- `YYYYMMDD` = build date (e.g., `20250514` = May 14, 2025)

#### Dev Tarballs

Dev tarballs are built from specific commits and follow the naming pattern: `MAJOR.MINOR.PATCH.dev0+{COMMIT_HASH}`

**To find and use a dev release:**

1. Visit the [dev tarball S3 bucket](https://therock-dev-tarball.s3.amazonaws.com/)
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

## Usage Examples

### Install from CI Run with BLAS Components

```bash
python build_tools/install_rocm_from_artifacts.py \
    --run-id 15575624591 \
    --amdgpu-family gfx110X-dgpu \
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
    --amdgpu-family gfx110X-dgpu \
    --newcomponent --tests
```

#### Step 6: Update Test Configuration (Optional)

If you want to add tests for your component in CI, also update `build_tools/github_actions/fetch_test_configurations.py`. See [Adding Tests](./adding_tests.md) for details.
