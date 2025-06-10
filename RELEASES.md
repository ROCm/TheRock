# Releases

This page describes how to install and use our release artifacts for ROCm and
external builds like PyTorch. We produce build artifacts as part of our
Continuous Integration (CI) build/test workflows as well as release artifacts as
part of Continuous Delivery (CD) nightly releases. See also the
[Roadmap for support](ROADMAP.md) and
[Build artifacts overview](docs/development/artifacts.md) pages.

> [!WARNING]
> These instructions assume familiarity with how to use ROCm. Please see
> https://rocm.docs.amd.com/ for general information about the ROCm software
> platform.

Table of contents:

- [Installing releases using pip](#installing-releases-using-pip)
  - [Installing ROCm Python packages](#installing-rocm-python-packages)
  - [Installing PyTorch Python packages](#installing-pytorch-python-packages)
- [Installing from tarballs](#installing-from-tarballs)
  - [Installing release tarballs](#installing-release-tarballs)
  - [Installing per-commit CI build tarballs](#installing-per-commit-ci-build-tarballs)
  - [Installing tarballs using `install_rocm_from_artifacts.py`](#installing-tarballs-using-install_rocm_from_artifactspy)
- [Testing your installation](#testing-your-installation)

## Installing releases using pip

We recommend installing ROCm and projects like PyTorch via `pip`, the
[Python package installer](https://packaging.python.org/en/latest/guides/tool-recommendations/).

### Python packages support status

|         | ROCm Python packages | PyTorch Python packages                                             |
| ------- | -------------------- | ------------------------------------------------------------------- |
| Linux   | ✅ Supported         | 🟡 In progress ([#703](https://github.com/ROCm/TheRock/issues/703)) |
| Windows | ⚪ Planned           | ⚪ Planned                                                          |

> [!IMPORTANT]
> Known issues with the Python wheels are tracked at
> https://github.com/ROCm/TheRock/issues/808.

### Installing ROCm Python packages

We provide several Python packages which together form the complete ROCm SDK.
These packages are defined in the
[`build_tools/packaging/python/templates/`](https://github.com/ROCm/TheRock/tree/main/build_tools/packaging/python/templates)
directory.

| Package name         | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `rocm`               | Primary sdist meta package that dynamically determines other deps  |
| `rocm-sdk-core`      | OS-specific core of the ROCm SDK (e.g. compiler and utility tools) |
| `rocm-sdk-devel`     | OS-specific development tools                                      |
| `rocm-sdk-libraries` | OS-specific libraries                                              |

For now these packages are published to GPU architecture-specific index pages
and must be installed using an appropriate `--find-links` argument to `pip`.
They will later be pushed to the
[Python Package Index (PyPI)](https://pypi.org/). **Please check back regularly
as these instructions will change as we migrate to official indexes and adjust
project layouts.**

> [!TIP]
> We highly recommend working within a [Python virtual environment](https://docs.python.org/3/library/venv.html):
>
> ```bash
> python -m venv .venv
> source .venv/bin/activate
> ```
>
> Multiple virtual environments can be present on a system at a time, allowing you to switch between them at will.

> [!WARNING]
> If you _really_ want a system-wide install, you can pass `--break-system-packages` to `pip` outside a virtual enivornment.
> In this case, commandline interface shims for executables are installed to `/usr/local/bin`, which normally has precedence over `/usr/bin` and might therefore conflict with a previous installation of ROCm.

<!-- TODO: mapping from product name to gfx family -->

#### gfx94X-dcgpu

```bash
python -m pip install \
  --pre --find-links https://therock-nightly-python.s3.amazonaws.com/gfx94X-dcgpu/index.html \
  rocm[libraries,devel]
```

#### gfx110X-dgpu

```bash
python -m pip install \
  --pre --find-links https://therock-nightly-python.s3.amazonaws.com/gfx110X-dgpu/index.html \
  rocm[libraries,devel]
```

#### gfx1151

```bash
python -m pip install \
  --pre --find-links https://therock-nightly-python.s3.amazonaws.com/gfx1151/index.html \
  rocm[libraries,devel]
```

#### gfx120X-all

```bash
python -m pip install \
  --pre --find-links https://therock-nightly-python.s3.amazonaws.com/gfx120X-all/index.html \
  rocm[libraries,devel]
```

### Installing PyTorch Python packages

Coming soon!

<!-- TODO: add `torch` to install commands
       * needs new build to be compatible with 'rocm' instead of 'rocm-sdk'
       * For 'rocm-sdk', need an environment workaround -->

## Installing from tarballs

<!-- TODO: clean up these sections and confirm the instructions work -->

Here's a quick way assuming you copied the all the tar files into `${BUILD_ARTIFACTS_DIR}` to "install" TheRock into `${BUILD_ARTIFACTS_DIR}/output_dir`

### Installing release tarballs

Release tarballs are already flattened and simply need untarring, follow the below instructions.

```bash
echo "Unpacking artifacts"
pushd "${BUILD_ARTIFACTS_DIR}"
mkdir output_dir
tar -xf *.tar.gz -C output_dir
popd
```

### Installing per-commit CI build tarballs

Our CI builds artifacts which need to be "flattened" by the `build_tools/fileset_tool.py artifact-flatten` command before they can be used. You will need to have a checkout (see for example [Clone and fetch sources](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md#clone-and-fetch-sources)) in `${SOURCE_DIR}` to use this tool and a Python environment.

```bash
echo "Unpacking artifacts"
pushd "${BUILD_ARTIFACTS_DIR}"
mkdir output_dir
python "${SOURCE_DIR}/build_tools/fileset_tool.py artifact-flatten *.tar.xz -o output_dir --verbose
popd
```

### Installing tarballs using `install_rocm_from_artifacts.py`

This script installs ROCm community builds produced by TheRock from either a developer/nightly tarball, a specific CI runner build or an already existing installation of TheRock. This script is used by CI and can be used locally.

Examples:

- Downloads all gfx94X S3 artifacts from [GitHub CI workflow run 15052158890](https://github.com/ROCm/TheRock/actions/runs/15052158890) to the default output directory `therock-build`:

  ```bash
  python build_tools/install_rocm_from_artifacts.py --run-id 15052158890 --amdgpu-family gfx94X-dcgpu --tests
  ```

- Downloads the version `6.4.0rc20250516` gfx110X artifacts from GitHub release tag `nightly-tarball` to the specified output directory `build`:

  ```bash
  python build_tools/install_rocm_from_artifacts.py --release 6.4.0rc20250516 --amdgpu-family gfx110X-dgpu --output-dir build
  ```

- Downloads the version `6.4.0.dev0+e015c807437eaf32dac6c14a9c4f752770c51b14` gfx110X artifacts from GitHub release tag `dev-tarball` to the default output directory `therock-build`:

  ```bash
  python build_tools/install_rocm_from_artifacts.py --release 6.4.0.dev0+e015c807437eaf32dac6c14a9c4f752770c51b14 --amdgpu-family gfx110X-dgpu
  ```

Select your AMD GPU family from this file [therock_amdgpu_targets.cmake](https://github.com/ROCm/TheRock/blob/59c324a759e8ccdfe5a56e0ebe72a13ffbc04c1f/cmake/therock_amdgpu_targets.cmake#L44-L81)

By default for CI workflow retrieval, all artifacts (excluding test artifacts) will be downloaded. For specific artifacts, pass in the flag such as `--rand` (RAND artifacts) For test artifacts, pass in the flag `--tests` (test artifacts). For base artifacts only, pass in the flag `--base-only`

## Testing your installation

The quickest way is to run `rocminfo`

```bash
echo "Running rocminfo"
pushd "${BUILD_ARTIFACTS_DIR}"
./output_dir/bin/rocminfo
popd
```

## Where to get artifacts

- [Releases](https://github.com/ROCm/TheRock/releases): Our releases page has the latest "developer" release of our tarball artifacts and source code.

- [Packages](https://github.com/orgs/ROCm/packages?repo_name=TheRock): We currently publish docker images for LLVM targets we support (as well as a container for our build machines)

- [Per-commit CI builds](https://github.com/ROCm/TheRock/actions/workflows/ci.yml?query=branch%3Amain+is%3Asuccess): Each of our latest passing CI builds has its own artifacts you can leverage. This is the latest and greatest! We will eventually support a nightly release that is at a higher quality bar than CI. Note a quick recipe for getting all of these from the s3 bucket is to use this quick command `aws s3 cp s3://therock-artifacts . --recursive --exclude "*" --include "${RUN_ID}-${OPERATING_SYSTEM}/*.tar.xz" --no-sign-request` where ${RUN_ID} is the runner id you selected (see the URL). Check the [AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to get the aws cli.
