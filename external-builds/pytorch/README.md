# Build PyTorch with ROCm support

This directory provides tooling for building PyTorch with ROCm Python wheels.

> [!TIP]
> If you want to install our prebuilt PyTorch packages instead of building them
> from source, see [RELEASES.md](/RELEASES.md) instead.

Table of contents:

- [Support status](#support-status)
- [Build instructions](#build-instructions)
- [Running/testing PyTorch](#runningtesting-pytorch)
- [Advanced build instructions](#advanced-build-instructions)
- [Development instructions](#development-instructions)

These build procedures are meant to run as part of ROCm CI and development flows
and thus leave less room for interpretation than in upstream repositories. Some
of this tooling is being migrated upstream as part of
[[RFC] Enable native Windows CI/CD on ROCm](https://github.com/pytorch/pytorch/issues/159520).

This incorporates advice from:

- https://github.com/pytorch/pytorch#from-source
- `.ci/manywheel/build_rocm.sh` and friends

## Support status

### Project and feature support status

| Project / feature              | Linux support                                                                     | Windows support  |
| ------------------------------ | --------------------------------------------------------------------------------- | ---------------- |
| torch                          | ✅ Supported                                                                      | ✅ Supported     |
| torchaudio                     | ✅ Supported                                                                      | ✅ Supported     |
| torchvision                    | ✅ Supported                                                                      | ✅ Supported     |
| Flash attention via [ao]triton | ✅ Supported                                                                      | ✅ Supported     |
| FBGEMM GenAI                   | ❌ Not supported (see Issue [#2056](https://github.com/ROCm/TheRock/issues/2056)) | ❌ Not supported |

### Supported PyTorch versions

We support building various PyTorch versions compatible with the latest ROCm
sources and release packages.

Support for the latest upstream PyTorch code (i.e. `main` or `nightly`) uses the
upstream projects directly, while extended support for older releases is maintained
via backported release branches in https://github.com/ROCm/pytorch. Developers can
also build variations of these versions to suite their own requirements.

> [!NOTE]
> We build "nightly" versions with alpha suffixes. Once
> PyTorch promotes a new release version, we switch from the upstream branches
> to the backported release branches in https://github.com/ROCm/pytorch without
> such version suffixes (e.g. `2.7.0a0`-> `2.7.1`).
>
> Historical builds are archived at https://rocm.nightlies.amd.com according to
> some retention policy.

Each PyTorch version uses a combination of:

- Git repository URLs for each project
- Git "repo hashtags" (branch names, tag names, or commit refs) for each project

See the following table for how each version is supported:

| PyTorch version | Linux                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Windows                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.10 alpha      | ✅ Using upstream pytorch<br><ul><li>[pytorch/pytorch `nightly` branch](https://github.com/pytorch/pytorch/tree/nightly)<ul><li>[ROCm/triton](https://github.com/ROCm/triton) - [`ci_commit_pins/triton.txt`](https://github.com/pytorch/pytorch/blob/nightly/.ci/docker/ci_commit_pins/triton.txt)</li></ul></li><li>[pytorch/audio `nightly` branch](https://github.com/pytorch/audio/tree/nightly)</li><li>[pytorch/vision `nightly` branch](https://github.com/pytorch/vision/tree/nightly)</li></ul>                                                                                                                                            | ✅ Using upstream pytorch<br><ul><li>[pytorch/pytorch `nightly` branch](https://github.com/pytorch/pytorch/tree/nightly)</li><li>[pytorch/audio `nightly` branch](https://github.com/pytorch/audio/tree/nightly)</li><li>[pytorch/vision `nightly` branch](https://github.com/pytorch/vision/tree/nightly)</li></ul>                                                                                                                                           |
| 2.9             | ✅ Using downstream ROCm/pytorch fork<br><ul><li>[ROCm/pytorch `release/2.9` branch](https://github.com/ROCm/pytorch/tree/release/2.9)<ul><li>[ROCm/triton](https://github.com/ROCm/triton) - [`ci_commit_pins/triton.txt`](https://github.com/ROCm/pytorch/blob/release/2.9/.ci/docker/ci_commit_pins/triton.txt)</li></ul></li><li>[pytorch/audio](https://github.com/pytorch/audio) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.9/related_commits)</li><li>[pytorch/vision](https://github.com/pytorch/vision) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.9/related_commits)</li></ul> | ✅ Using downstream ROCm/pytorch fork<br><ul><li>[ROCm/pytorch `release/2.9` branch](https://github.com/ROCm/pytorch/tree/release/2.9)</li><li>[pytorch/audio](https://github.com/pytorch/audio) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.9/related_commits)</li><li>[pytorch/vision](https://github.com/pytorch/vision) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.9/related_commits)</li></ul> |
| 2.9 alpha       | Previously built                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Previously built                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2.8             | ✅ Using downstream ROCm/pytorch fork<br><ul><li>[ROCm/pytorch `release/2.8` branch](https://github.com/ROCm/pytorch/tree/release/2.8)<ul><li>[ROCm/triton](https://github.com/ROCm/triton) - [`ci_commit_pins/triton.txt`](https://github.com/ROCm/pytorch/blob/release/2.8/.ci/docker/ci_commit_pins/triton.txt)</li></ul></li><li>[pytorch/audio](https://github.com/pytorch/audio) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.8/related_commits)</li><li>[pytorch/vision](https://github.com/pytorch/vision) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.8/related_commits)</li></ul> | Unsupported                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2.7             | ✅ Using downstream ROCm/pytorch fork<br><ul><li>[ROCm/pytorch `release/2.7` branch](https://github.com/ROCm/pytorch/tree/release/2.7)<ul><li>[ROCm/triton](https://github.com/ROCm/triton) - [`ci_commit_pins/triton.txt`](https://github.com/ROCm/pytorch/blob/release/2.7/.ci/docker/ci_commit_pins/triton.txt)</li></ul></li><li>[pytorch/audio](https://github.com/pytorch/audio) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.7/related_commits)</li><li>[pytorch/vision](https://github.com/pytorch/vision) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.7/related_commits)</li></ul> | Unsupported                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

See also:

- The [Alternative branches and versions](#alternative-branches-and-versions)
  section for detailed information about configurations
- The upstream PyTorch
  [release documentation](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- Workflow source code:
  - [`.github/workflows/build_portable_linux_pytorch_wheels.yml`](/.github/workflows/build_portable_linux_pytorch_wheels.yml)
  - [`.github/workflows/build_windows_pytorch_wheels.yml`](/.github/workflows/build_windows_pytorch_wheels.yml)

## Build instructions

See the comments in [`build_prod_wheels.py`](./build_prod_wheels.py) for
detailed instructions. That information is summarized here.

### Prerequisites and setup

You will need a supported Python version (3.11+) on a system which we build the
`rocm[libraries,devel]` packages for. See the
[`RELEASES.md`: Installing releases using pip](../../RELEASES.md#installing-releases-using-pip)
and [Python Packaging](../../docs/packaging/python_packaging.md) documentation
for more background on these `rocm` packages.

> [!WARNING]
> On Windows, prefer to install Python for the current user only and to a path
> **without spaces** like
> `C:\Users\<username>\AppData\Local\Programs\Python\Python312`.
>
> Several developers have reported issues building torchvision when using
> "Install Python for all users" with a default path like
> `C:\Program Files\Python312` (note the space in "Program Files"). See
> https://github.com/pytorch/vision/issues/9165 for details.

> [!WARNING]
> On Windows, when building with "--enable-pytorch-flash-attention-windows",
> Make sure to use [ninja 1.13.1](https://github.com/ninja-build/ninja/releases/tag/v1.13.1) or above.
>
> NOTE: If you use ccache and face "invalid argument" errors during the aotriton build,
> disable ccache and try again.

### Quickstart

It is highly recommended to use a virtual environment unless working within a
throw-away container or CI environment.

```bash
# On Linux
python -m venv .venv && source .venv/bin/activate

# On Windows
python -m venv .venv && .venv\Scripts\activate.bat
```

Now checkout repositories using their default branches:

- On Linux, use default paths (nested under this folder):

  ```bash
  python pytorch_torch_repo.py checkout
  python pytorch_audio_repo.py checkout
  python pytorch_vision_repo.py checkout
  ```

- On Windows, use shorter paths to avoid command length limits:

  ```batch
  python pytorch_torch_repo.py checkout --checkout-dir C:/b/pytorch
  python pytorch_audio_repo.py checkout --checkout-dir C:/b/audio
  python pytorch_vision_repo.py checkout --checkout-dir C:/b/vision
  ```

Now note the gfx target you want to build for and then...

1. Install `rocm` packages
1. Build PyTorch wheels
1. Install the built PyTorch wheels

...all in one command. See the
[advanced build instructions](#advanced-build-instructions) for ways to
mix/match build steps.

- On Linux:

  ```bash
  python build_prod_wheels.py build \
    --install-rocm --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
    --output-dir $HOME/tmp/pyout
  ```

- On Windows:

  ```batch
  python build_prod_wheels.py build ^
    --install-rocm --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ ^
    --pytorch-dir C:/b/pytorch ^
    --pytorch-audio-dir C:/b/audio ^
    --pytorch-vision-dir C:/b/vision ^
    --output-dir %HOME%/tmp/pyout
  ```

## Running/testing PyTorch

### Prerequisites

On Linux we run automated tests under our
[`no_rocm_image_ubuntu24_04.Dockerfile`](dockerfiles/no_rocm_image_ubuntu24_04.Dockerfile)
container. Docker is optional for developers and users. If you want to use our
test image, run it like so:

```bash
sudo docker run -it \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --group-add=video --group-add=render --group-add=110 \
  ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest
```

### Running ROCm and PyTorch sanity checks

The simplest tests for a working PyTorch with ROCm install are:

```bash
rocm-sdk test
# Should show passing tests

python -c "import torch; print(torch.cuda.is_available())"
# Should print "True"
```

### Running PyTorch smoketests

We have additional smoketests in [smoke-tests](./smoke-tests/) that run some
sample computations. To run these tests:

```bash
# Basic usage (no wrapper script)
pytest -v smoke-tests

# Wrapper script, passing through some useful pytest args:
python run_pytorch_smoke_tests.py -- \
  --log-cli-level=INFO \
  -v
```

### Running full PyTorch tests

We have a [`run_pytorch_tests.py`](run_pytorch_tests.py) script
which runs PyTorch unit tests using pytest with additional test exclusion
capabilities tailored for AMD ROCm GPUs. See the script for detailed
instructions. Here are a few examples:

```bash
# Basic usage (auto-detect everything, no extra args):
python run_pytorch_tests.py

# Typical usage on CI, passing through some useful pytest args:
python run_pytorch_tests.py -- \
  --continue-on-collection-errors \
  --import-mode=importlib \
  -v

# Custom test selection with pytest -k:
python run_pytorch_tests.py -k "test_nn and not test_dropout"

# Explicit pytorch repo path (for test sources) and GPU family (for filtering)
python run_pytorch_tests.py --pytorch-dir=/tmp/pytorch --amdgpu-family=gfx950
```

Tests can also be run by following the ROCm documentation at
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#testing-the-pytorch-installation.
For example:

```bash
PYTORCH_TEST_WITH_ROCM=1 python pytorch/test/run_test.py --include test_torch
```

## Nightly releases

### Gating releases with Pytorch tests

With passing builds we upload `torch`, `torchvision`, `torchaudio`, and `triton` wheels to subfolders of the "v2-staging" directory in the nightly release s3 bucket with a public URL at https://rocm.nightlies.amd.com/v2-staging/

Only with passing Torch tests we promote passed wheels to the "v2" directory in the nightly release s3 bucket with a public URL at https://rocm.nightlies.amd.com/v2/

If no runner is available: Promotion is blocked by default. Set `bypass_tests_for_releases=true` for exceptional cases under [`amdgpu_family_matrix.py`](/build_tools/github_actions/amdgpu_family_matrix.py)

## Advanced build instructions

### Other ways to install the rocm packages

The `rocm[libraries,devel]` packages can be installed in multiple ways:

- (As above) during the `build_prod_wheels.py build` subcommand

- Using the more tightly scoped `build_prod_wheels.py install-rocm` subcommand:

  ```bash
  build_prod_wheels.py
      --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
      install-rocm
  ```

- Manually installing from a release index:

  ```bash
  # From therock-nightly-python
  python -m pip install \
    --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
    rocm[libraries,devel]

  # OR from therock-dev-python
  python -m pip install \
    --index-url https://rocm.devreleases.amd.com/v2/gfx110X-all/ \
    rocm[libraries,devel]
  ```

- Building the rocm Python packages from artifacts fetched from a CI run:

  <!-- TODO: teach scripts to look up latest stable run and mkdir themselves -->

  ```bash
  # From the repository root
  mkdir $HOME/.therock/17123441166
  mkdir $HOME/.therock/17123441166/artifacts
  python ./build_tools/fetch_artifacts.py \
    --run-id=17123441166 \
    --target=gfx110X-all \
    --output-dir=$HOME/.therock/17123441166/artifacts

  python ./build_tools/build_python_packages.py \
    --artifact-dir=$HOME/.therock/17123441166/artifacts \
    --dest-dir=$HOME/.therock/17123441166/packages
  ```

- Building the rocm Python packages from artifacts built from source:

  ```bash
  # From the repository root
  cmake --build build --target therock-archives

  python ./build_tools/build_python_packages.py \
    --artifact-dir=build/artifacts \
    --dest-dir=build/packages
  ```

### Bundling PyTorch and ROCm together into a "fat wheel"

By default, Python wheels produced by the PyTorch build do not include ROCm
binaries. Instead, they expect those binaries to come from the
`rocm[libraries,devel]` packages. A "fat wheel" bundles the ROCm binaries into
the same wheel archive to produce a standalone install including both PyTorch
and ROCm, with all necessary patches to shared library / DLL loading for out of
the box operation.

To produce such a fat wheel, see
[`windows_patch_fat_wheel.py`](./windows_patch_fat_wheel.py) and a future
equivalent script for Linux.

## Development instructions

This section covers recommended practices for making changes to PyTorch and
other repositories for use with the build scripts and integration with
version control systems.

If you want to make changes to PyTorch source code, prefer in this order:

1. Contributing to upstream `main` branches
1. Contributing to upstream `release/` branches while within the release window
1. Contributing to downstream `release/` branches in forked repositories

> [!NOTE]
> We used to support applying git patches as part of checkout out PyTorch
> repositories. This system has been removed as ROCm-specific changes are now
> maintained in the https://github.com/ROCm/pytorch/ fork rather than as patch
> files.

### Checking out PyTorch repositories

Each `pytorch_*_repo.py` script handles checkout and preparation:

1. Clone the repository as needed from `--gitrepo-origin` to the `--repo-name`
   folder under path `--repo`
1. Checkout the `--repo-hashtag` git ref/tag
1. Tag the upstream commit as `THEROCK_UPSTREAM_DIFFBASE`
1. Run 'hipify' on the repository, editing source files and committing the
   changes
1. Tag the hipify commit as `THEROCK_HIPIFY_DIFFBASE`

After running one of the `pytorch_*_repo.py` scripts you should have a
repository with history like this:

```console
$ python pytorch_torch_repo.py checkout --repo-hashtag main
$ cd pytorch
$ git log --oneline

cb53ee6fd45 (HEAD, tag: THEROCK_HIPIFY_DIFFBASE) DO NOT SUBMIT: HIPIFY
96682103026 (tag: THEROCK_UPSTREAM_DIFFBASE, origin/main) Example upstream commit 2
3f5a8e2003f Example upstream commit 1
```

Note the sequence of commits and tags that were created:

- `main` is checked out initially and is tagged `THEROCK_UPSTREAM_DIFFBASE`
- hipify is run and its changes are tagged `THEROCK_HIPIFY_DIFFBASE`

### Alternative branches and versions

#### PyTorch main branches

This checks out the `main` branches from https://github.com/pytorch, tracking
the latest (potentially unstable) code:

- https://github.com/pytorch/pytorch/tree/main
- https://github.com/pytorch/audio/tree/main
- https://github.com/pytorch/vision/tree/main

```bash
python pytorch_torch_repo.py checkout --repo-hashtag main
python pytorch_audio_repo.py checkout --repo-hashtag main
python pytorch_vision_repo.py checkout --repo-hashtag main
# Note that triton will be checked out at the PyTorch pin.
python pytorch_triton_repo.py checkout
```

#### PyTorch nightly branches

This checks out the `nightly` branches from https://github.com/pytorch,
tracking the latest pytorch.org nightly release:

- https://github.com/pytorch/pytorch/tree/nightly
- https://github.com/pytorch/audio/tree/nightly
- https://github.com/pytorch/vision/tree/nightly

```bash
python pytorch_torch_repo.py checkout --repo-hashtag nightly
python pytorch_audio_repo.py checkout --repo-hashtag nightly
python pytorch_vision_repo.py checkout --repo-hashtag nightly
# Note that triton will be checked out at the PyTorch pin.
python pytorch_triton_repo.py checkout
```

#### ROCm PyTorch release branches

Because upstream PyTorch freezes at release but AMD needs to keep updating
stable versions for a longer period of time, backport branches are maintained in
the fork at https://github.com/ROCm/pytorch.

> [!TIP]
> You are welcome to maintain your own branches that extend one of AMD's.
> Change origins and tags as appropriate.

In order to check out and build one of these, use the following instructions:

```bash
# Other common --repo-hashtag values:
#   release/2.10
#   release/2.9
#   release/2.8
#   release/2.7
python pytorch_torch_repo.py checkout \
  --gitrepo-origin https://github.com/ROCm/pytorch.git \
  --repo-hashtag release/2.9

# Backport branches have `related_commits` files that point to specific
# sub-project commits, so the main torch repo must be checked out first to
# have proper defaults.
python pytorch_audio_repo.py checkout --require-related-commit
python pytorch_vision_repo.py checkout --require-related-commit

python pytorch_triton_repo.py checkout
```
