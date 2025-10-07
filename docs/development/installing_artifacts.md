# Installing Artifacts

This document provides instructions for installing ROCm artifacts from TheRock builds.

## Command Options

The script supports the following command-line options:

| Option | Type | Description |
|--------|------|-------------|
| `--output-dir` | Path | Output directory for TheRock installation (default: `./therock-build`) |
| `--amdgpu-family` | String | AMD GPU family target (required) |
| `--run-id` | String | GitHub CI workflow run ID to install from |
| `--release` | String | Release version from nightly or dev tarballs |
| `--input-dir` | String | Existing TheRock directory to copy from |
| `--blas` | Flag | Include BLAS artifacts |
| `--fft` | Flag | Include FFT artifacts |
| `--miopen` | Flag | Include MIOpen artifacts |
| `--prim` | Flag | Include primitives artifacts |
| `--rand` | Flag | Include random number generator artifacts |
| `--rccl` | Flag | Include RCCL artifacts |
| `--tests` | Flag | Include test artifacts for enabled components |
| `--base-only` | Flag | Include only base artifacts (minimal installation) |

### Finding GitHub Run IDs

To use the `--run-id` option, you need to find the GitHub Actions workflow run ID:

1. Navigate to the [TheRock Actions page](https://github.com/ROCm/TheRock/actions)
2. Click on the "CI" workflow
3. Find a successful run (green checkmark)
4. Click on the run to view details
5. The run ID is the number in the URL: `https://github.com/ROCm/TheRock/actions/runs/[RUN_ID]`

For example, if the URL is `https://github.com/ROCm/TheRock/actions/runs/15575624591`, then the run ID is `15575624591`.

### Finding Release Versions

TheRock provides two types of release tarballs:

#### Nightly Tarballs

1. Visit the [nightly tarball S3 bucket](https://therock-nightly-tarball.s3.amazonaws.com/)
2. Find the version you want (versions are date-stamped)
3. Use the full version string with `--release`

#### Dev Tarballs

1. Visit the [dev tarball S3 bucket](https://therock-dev-tarball.s3.amazonaws.com/)
2. Find the version corresponding to the commit you want
3. Use the full version string with `--release`

## Usage Examples

### Install from CI Run with BLAS Components

````bash
python build_tools/install_rocm_from_artifacts.py \
    --run-id 15575624591 \
    --amdgpu-family gfx110X-dgpu \
    --blas --tests
````

### Install from Nightly Tarball with Multiple Components

Install RCCL and FFT components from a nightly build for gfx94X:

````bash
python build_tools/install_rocm_from_artifacts.py \
    --release 6.4.0rc20250416 \
    --amdgpu-family gfx94X-dcgpu \
    --rccl --fft --tests
````

## Adding Support for New Components

When you add a new component to TheRock, you will need to update `install_rocm_from_artifacts.py` to allow users to selectively install it.

> [!NOTE]
> You only need to modify `install_rocm_from_artifacts.py` when adding an entirely new component to TheRock.
> Typically if you are adding a new .toml file you will need to add support to `install_rocm_from_artifacts.py`.
> Adding libraries to existing components, (such as including a new library in the `blas` component) requires no script changes.

### Step-by-Step Guide

Here's how to add support for a hypothetical component called `newcomponent`:

#### Step 1: Verify the Artifact is Built

Ensure your component's artifact is properly defined in CMake and built:

````bash
# Check that the artifact is created during build
cmake --build build
ls build/artifacts/newcomponent_*
````

You should see artifacts like:
- `newcomponent_lib_gfx110X`
- `newcomponent_test_gfx110X`
- etc.

#### Step 2: Add Command-Line Argument

Open `build_tools/install_rocm_from_artifacts.py` and add a new argument in the `artifacts_group`:

````python
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
````

#### Step 3: Add to Artifact Selection Logic

In the `retrieve_artifacts_by_run_id` function, add your component to the conditional logic:

````python
# filepath: \home\bharriso\Source\TheRock\build_tools\install_rocm_from_artifacts.py
    if args.base_only:
        argv.extend(base_artifact_patterns)
    elif any([args.blas, args.fft, args.miopen, args.prim, args.rand, args.rccl, args.new_component]):
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
        if args.new_component:
            extra_artifacts.append("newcomponent")

        extra_artifact_patterns = [f"{a}_lib" for a in extra_artifacts]
````

#### Step 4: Update Documentation

Add your new component to the command options table in this document (see the table above).

#### Step 5: Test Your Changes

Test that artifacts can be fetched with your new flag:

````bash
# Test with a CI run
python build_tools/install_rocm_from_artifacts.py \
    --run-id YOUR_RUN_ID \
    --amdgpu-family gfx110X-dgpu \
    --newcomponent --tests
````

#### Step 6: Update Test Configuration (Optional)

If you want to add tests for your component in CI, also update `build_tools/github_actions/fetch_test_configurations.py`. See [Adding Tests](./adding_tests.md) for details.
