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
>
> **Note: these install steps are a substitute for those on that website**.

Table of contents:

- [Installing releases using pip](#installing-releases-using-pip)
  - [Installing ROCm Python packages](#installing-rocm-python-packages)
  - [Using ROCm Python packages](#using-rocm-python-packages)
  - [Installing PyTorch Python packages](#installing-pytorch-python-packages)
  - [Using PyTorch Python packages](#using-pytorch-python-packages)
- [Installing from tarballs](#installing-from-tarballs)
  - [Installing release tarballs](#installing-release-tarballs)
  - [Installing per-commit CI build tarballs](#installing-per-commit-ci-build-tarballs)
  - [Installing tarballs using `install_rocm_from_artifacts.py`](#installing-tarballs-using-install_rocm_from_artifactspy)
  - [Using installed tarballs](#using-installed-tarballs)

## Installing releases using pip

We recommend installing ROCm and projects like PyTorch via `pip`, the
[Python package installer](https://packaging.python.org/en/latest/guides/tool-recommendations/).

We currently support Python 3.11 and 3.12, with 3.13 support coming soon.

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

### Using ROCm Python packages

After installing the ROCm Python packages, you should see them in your
environment:

```bash
pip freeze | grep rocm
# rocm==6.5.0rc20250610
# rocm-sdk-core==6.5.0rc20250610
# rocm-sdk-devel==6.5.0rc20250610
# rocm-sdk-libraries-gfx110X-dgpu==6.5.0rc20250610
```

You should also see various tools on your `PATH` and in the `bin` directory:

```bash
which rocm-sdk
# .../.venv/bin/rocm-sdk

ls .venv/bin
# activate       amdclang++    hipcc      python                 rocm-sdk
# activate.csh   amdclang-cl   hipconfig  python3                rocm-smi
# activate.fish  amdclang-cpp  pip        python3.12             roc-obj
# Activate.ps1   amdflang      pip3       rocm_agent_enumerator  roc-obj-extract
# amdclang       amdlld        pip3.12    rocminfo               roc-obj-ls
```

The `rocm-sdk` tool can be used to inspect and test the installation:

```console
$ rocm-sdk --help
usage: rocm-sdk {command} ...

ROCm SDK Python CLI

positional arguments:
  {path,test,version,targets}
    path                Print various paths to ROCm installation
    test                Run installation tests to verify integrity
    version             Print version information
    targets             Print information about the GPU targets that are supported

$ rocm-sdk test
...
Ran 22 tests in 8.284s
OK

$ rocm-sdk targets
gfx1100;gfx1101;gfx1102
```

Once you have verified your installation, you can continue to use it for
standard ROCm development or install PyTorch or another supported Python ML
framework.

### Installing PyTorch Python packages

> [!WARNING]
> This is under **active** development.

Using the index pages listed above, you can install `torch==2.7.0a0` instead of
`rocm[libraries,devel]`.

For example, with gfx110X-dgpu:

```bash
python -m pip install \
  --pre --find-links https://therock-nightly-python.s3.amazonaws.com/gfx110X-dgpu/index.html \
  torch==2.7.0a0

pip freeze
# ...
# rocm-sdk==6.5.0rc20250606
# rocm-sdk-core==6.5.0rc20250606
# rocm-sdk-libraries-gfx110X-dgpu==6.5.0rc20250606
# ...
# torch==2.7.0a0+rocmsdk20250608
```

Current known issues, expected to be fixed in the next nightly (see also
https://github.com/ROCm/TheRock/issues/808):

- This currently fetches the old `rocm-sdk` package instead of the new `rocm`
  package.

- This currently needs a workaround on startup to fix import errors:

  ```python
  # Temporary workaround!
  import rocm_sdk

  rocm_sdk.preload_libraries("rocprofiler-sdk-roctx", "miopen")

  # ... then continue as normal
  import torch
  ```

### Using PyTorch Python packages

After installing the `torch` package with ROCm support, PyTorch can be used
normally:

```python
import torch

print(torch.cuda.is_available())
# True
print(torch.cuda.get_device_name(0))
# e.g. AMD Radeon Pro W7900 Dual Slot
```

See also the
[Testing the PyTorch installation](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html#testing-the-pytorch-installation)
instructions in the AMD ROCm documentation.

## Installing from tarballs

Standalone "ROCm SDK tarballs" are assembled from the same
[artifacts](docs/development/artifacts.md) as the Python packages which can be
[installed using pip](#installing-releases-using-pip), without the additional
wrapper Python wheels or utility scripts.

### Installing release tarballs

Release tarballs are automatically uploaded to
[GitHub releases](https://github.com/ROCm/TheRock/releases) and AWS S3 buckets
for archival. The S3 buckets do not yet have index pages.

| Release page                                                                      | S3 bucket                                                                    | Description                                       |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------- |
| [`nightly-tarball`](https://github.com/ROCm/TheRock/releases/tag/nightly-tarball) | [therock-nightly-tarball](https://therock-nightly-tarball.s3.amazonaws.com/) | Nightly builds from the `main` branch             |
| [`dev-tarball`](https://github.com/ROCm/TheRock/releases/tag/dev-tarball)         | [therock-dev-tarball](https://therock-dev-tarball.s3.amazonaws.com/)         | ⚠️ Development builds from project maintainers ⚠️ |

After downloading, simply extract the release tarball into place:

```bash
mkdir therock && cd therock
# For example...
wget https://github.com/ROCm/TheRock/releases/download/nightly-tarball/therock-dist-linux-gfx110X-dgpu-6.5.0rc20250610.tar.gz

mkdir install
tar -xf *.tar.gz -C install

ls install
# bin  include  lib  libexec  llvm  share

# Now test some of the installed tools:
./install/bin/rocminfo
./install/bin/test_hip_api

# You may now also want to add the install directory to your PATH.
```

### Installing per-commit CI build tarballs

<!-- TODO: Hide this section by default?
           Maybe move into artifacts.md or another developer page. -->

Our CI builds artifacts at every commit. These can be installed by "flattening"
them from the expanded artifacts down to a ROCm SDK "dist folder" using the
`artifact-flatten` command from
[`build_tools/fileset_tool.py`](https://github.com/ROCm/TheRock/blob/main/build_tools/fileset_tool.py).

1. Download TheRock's source code and setup your Python environment:

   ```bash
   # Clone the repository
   git clone https://github.com/ROCm/TheRock.git
   cd TheRock

   # Init python virtual environment and install python dependencies
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

1. Find the CI workflow run that you want to install from. For example, search
   through recent successful runs of the `ci.yml` workflow for `push` events on
   the `main` branch
   [using this page](https://github.com/ROCm/TheRock/actions/workflows/ci.yml?query=branch%3Amain+is%3Asuccess+event%3Apush).
   (choosing a build that took more than a few minutes)

<!-- TODO: working here, translate this paragraph into a code sample -->

Note a quick recipe for getting all of these from the s3 bucket is to use this quick command `aws s3 cp s3://therock-artifacts . --recursive --exclude "*" --include "${RUN_ID}-${OPERATING_SYSTEM}/*.tar.xz" --no-sign-request` where ${RUN_ID} is the runner id you selected (see the URL). Check the [AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to get the aws cli.

```bash
echo "Unpacking artifacts"
pushd "${BUILD_ARTIFACTS_DIR}"
mkdir output_dir
python "${SOURCE_DIR}/build_tools/fileset_tool.py artifact-flatten *.tar.xz -o output_dir --verbose
popd
```

### Installing tarballs using `install_rocm_from_artifacts.py`

<!-- TODO: move this above the manual `tar -xf` commands? -->

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

By default for CI workflow retrieval, all artifacts (excluding test artifacts) will be downloaded. For specific artifacts, pass in the flag such as `--rand` (RAND artifacts). For test artifacts, pass in the flag `--tests` (test artifacts). For base artifacts only, pass in the flag `--base-only`

### Using installed tarballs

The quickest way is to run `rocminfo`

```bash
echo "Running rocminfo"
pushd "${BUILD_ARTIFACTS_DIR}"
./output_dir/bin/rocminfo
popd
```

## Where to get artifacts

<!-- TODO: move Dockerfiles to their own top level section -->

- [Packages](https://github.com/orgs/ROCm/packages?repo_name=TheRock): We currently publish docker images for LLVM targets we support (as well as a container for our build machines)

<!-- TODO: delete, merged into other sections -->

- [Per-commit CI builds](https://github.com/ROCm/TheRock/actions/workflows/ci.yml?query=branch%3Amain+is%3Asuccess): Each of our latest passing CI builds has its own artifacts you can leverage. This is the latest and greatest! We will eventually support a nightly release that is at a higher quality bar than CI. Note a quick recipe for getting all of these from the s3 bucket is to use this quick command `aws s3 cp s3://therock-artifacts . --recursive --exclude "*" --include "${RUN_ID}-${OPERATING_SYSTEM}/*.tar.xz" --no-sign-request` where ${RUN_ID} is the runner id you selected (see the URL). Check the [AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to get the aws cli.
