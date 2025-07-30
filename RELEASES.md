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
  - [Installing per-commit CI build tarballs manually](#installing-per-commit-ci-build-tarballs-manually)
  - [Installing tarballs using `install_rocm_from_artifacts.py`](#installing-tarballs-using-install_rocm_from_artifactspy)
  - [Using installed tarballs](#using-installed-tarballs)
- [Using Dockerfiles](#using-dockerfiles)

## Installing releases using pip

We recommend installing ROCm and projects like PyTorch via `pip`, the
[Python package installer](https://packaging.python.org/en/latest/guides/tool-recommendations/).

We currently support Python 3.11, 3.12, and 3.13.

### Python packages release status

> [!IMPORTANT]
> Known issues with the Python wheels are tracked at
> https://github.com/ROCm/TheRock/issues/808.
>
> ⚠️ Windows packages are new and may be unstable! ⚠️

| Platform |                                                                                                                                                                                                                                         ROCm Python packages |                                                                                                                                                                                                                                               PyTorch Python packages |
| -------- | -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Linux    | [![Release portable Linux packages](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_packages.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_packages.yml?query=branch%3Amain) | [![Release Linux PyTorch Wheels](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_pytorch_wheels.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_pytorch_wheels.yml?query=branch%3Amain) |
| Windows  |                      [![Release Windows packages](https://github.com/ROCm/TheRock/actions/workflows/release_windows_packages.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_windows_packages.yml?query=branch%3Amain) |             [![Release Windows PyTorch Wheels](https://github.com/ROCm/TheRock/actions/workflows/release_windows_pytorch_wheels.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_windows_pytorch_wheels.yml?query=branch%3Amain) |

### Installing ROCm Python packages

We provide several Python packages which together form the complete ROCm SDK.

- See [ROCm Python Packaging via TheRock](./docs/packaging/python_packaging.md)
  for information about the each package.
- The packages are defined in the
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
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx94X-dcgpu/ \
  rocm[libraries,devel]
```

#### gfx950-dcgpu

```bash
python -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx950-dcgpu/ \
  rocm[libraries,devel]
```

#### gfx110X-dgpu

```bash
python -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/ \
  rocm[libraries,devel]
```

#### gfx1151

```bash
python -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/ \
  rocm[libraries,devel]
```

#### gfx120X-all

```bash
python -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx120X-all/ \
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

Using the index pages listed above, you can install `torch` instead of
`rocm[libraries,devel]`:

#### gfx94X-dcgpu

```bash
python -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx94X-dcgpu/ \
  torch
```

#### gfx950-dcgpu

```bash
python -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx950-dcgpu/ \
  torch
```

#### gfx110X-dgpu

```bash
python -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/ \
  torch
```

#### gfx1151

```bash
python -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/ \
  torch
```

#### gfx120X-all

```bash
python -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx120X-all/ \
  torch
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

Release tarballs are automatically uploaded to AWS S3 buckets.
The S3 buckets do not yet have index pages.

| S3 bucket                                                                    | Description                                       |
| ---------------------------------------------------------------------------- | ------------------------------------------------- |
| [therock-nightly-tarball](https://therock-nightly-tarball.s3.amazonaws.com/) | Nightly builds from the `main` branch             |
| [therock-dev-tarball](https://therock-dev-tarball.s3.amazonaws.com/)         | ⚠️ Development builds from project maintainers ⚠️ |

After downloading, simply extract the release tarball into place:

```bash
mkdir therock-tarball && cd therock-tarball
# For example...
wget https://therock-nightly-tarball.s3.us-east-2.amazonaws.com/therock-dist-linux-gfx110X-dgpu-6.5.0rc20250610.tar.gz

mkdir install
tar -xf *.tar.gz -C install
```

### Installing per-commit CI build tarballs manually

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
   [using this page](https://github.com/ROCm/TheRock/actions/workflows/ci.yml?query=branch%3Amain+is%3Asuccess+event%3Apush)
   (choosing a build that took more than a few minutes - documentation only
   changes skip building and uploading).

1. Download the artifacts for that workflow run from S3 using either the
   [AWS CLI](https://aws.amazon.com/cli/) or
   [AWS SDK for Python (Boto3)](https://aws.amazon.com/sdk-for-python/):

   <!-- TODO: replace URLs with cloudfront / some other CDN instead of raw S3 -->

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

After installing (downloading and extracting) a tarball, you can test it by
running programs from the `bin/` directory:

```bash
ls install
# bin  include  lib  libexec  llvm  share

# Now test some of the installed tools:
./install/bin/rocminfo
./install/bin/test_hip_api
```

You may also want to add the install directory to your `PATH` or set other
environment variables like `ROCM_HOME`.
