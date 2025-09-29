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

| Project / feature              | Linux support                              | Windows support |
| ------------------------------ | ------------------------------------------ | --------------- |
| torch                          | ✅ Supported                               | ✅ Supported    |
| torchaudio                     | ✅ Supported                               | ✅ Supported    |
| torchvision                    | ✅ Supported                               | ✅ Supported    |
| Flash attention via [ao]triton | ✅ Supported                               | ✅ Supported    |
|                                | (only for versions < 2.10, disabled ≥2.10) |                 |

### Supported PyTorch versions

We support building ROCm source and nightly releases together with several
different PyTorch versions. The intent is to support the latest upstream PyTorch
code (i.e. `main` or `nightly`) as well as recently published release branches
which users depend on. Developers can also build variations of these versions to
suite their own requirements.

Each PyTorch version uses a combination of:

- Git repository URLs for each project
- Git "repo hashtags" (branch names, tag names, or commit refs) for each project
- Optional patches to be applied on top of a git checkout

See the following table for how each version is supported. Previously supported
versions are no longer being built but may have existing wheels in the [nightly build repo](https://rocm.nightlies.amd.com/v2/).

| PyTorch version      | Linux                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Windows                                                                                                                                                                                                                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| main (2.10 alpha)    | (Similar to nightly)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | ✅ Using upstream pytorch<br><ul><li>[pytorch/pytorch `main` branch](https://github.com/pytorch/pytorch/tree/main)</li><li>[pytorch/audio `main` branch](https://github.com/pytorch/audio/tree/main)</li><li>[pytorch/vision `main` branch](https://github.com/pytorch/vision/tree/main)</li></ul> |
| nightly (2.10 alpha) | ✅ Using upstream pytorch<br><ul><li>[pytorch/pytorch `nightly` branch](https://github.com/pytorch/pytorch/tree/nightly)<ul><li>[ROCm/triton](https://github.com/ROCm/triton) - [`ci_commit_pins/triton.txt`](https://github.com/pytorch/pytorch/blob/nightly/.ci/docker/ci_commit_pins/triton.txt)</li></ul></li><li>[pytorch/audio `nightly` branch](https://github.com/pytorch/audio/tree/nightly)</li><li>[pytorch/vision `nightly` branch](https://github.com/pytorch/vision/tree/nightly)</li></ul>                                                                                                                                            | (Similar to main)                                                                                                                                                                                                                                                                                  |
| 2.9 alpha            | Previously supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Previously supported                                                                                                                                                                                                                                                                               |
| 2.8                  | Unsupported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Unsupported                                                                                                                                                                                                                                                                                        |
| 2.7                  | ✅ Using downstream ROCm/pytorch fork<br><ul><li>[ROCm/pytorch `release/2.7` branch](https://github.com/ROCm/pytorch/tree/release/2.7)<ul><li>[ROCm/triton](https://github.com/ROCm/triton) - [`ci_commit_pins/triton.txt`](https://github.com/ROCm/pytorch/blob/release/2.7/.ci/docker/ci_commit_pins/triton.txt)</li></ul></li><li>[pytorch/audio](https://github/com/pytorch/audio) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.7/related_commits)</li><li>[pytorch/vision](https://github/com/pytorch/vision) - ["rocm related commit"](https://github.com/ROCm/pytorch/blob/release/2.7/related_commits)</li></ul> | Unsupported                                                                                                                                                                                                                                                                                        |

See also:

- The [Alternate Branches / Patch Sets](#alternate-branches--patch-sets) section
  for detailed information about configurations
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
  python pytorch_torch_repo.py checkout --repo C:/b/pytorch
  python pytorch_audio_repo.py checkout --repo C:/b/audio
  python pytorch_vision_repo.py checkout --repo C:/b/vision
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
    --install-rocm --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/ \
    --output-dir $HOME/tmp/pyout
  ```

- On Windows:

  ```batch
  python build_prod_wheels.py build ^
    --install-rocm --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/ ^
    --pytorch-dir C:/b/pytorch ^
    --pytorch-audio-dir C:/b/audio ^
    --pytorch-vision-dir C:/b/vision ^
    --output-dir %HOME%/tmp/pyout
  ```

## Running/testing PyTorch

### Running ROCm and PyTorch sanity checks

The simplest tests for a working PyTorch with ROCm install are:

```bash
rocm-sdk test
# Should show passing tests

python -c "import torch; print(torch.cuda.is_available())"
# Should print "True"
```

### Running PyTorch smoketests

We have additional smoketests that run some sample computations. See
[smoke-tests](./smoke-tests/) for details, or just run:

```bash
pytest -v smoke-tests
```

### Running full PyTorch tests

See https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#testing-the-pytorch-installation

<!-- TODO(erman-gurses): update docs here -->

## Nightly releases

### Gating releases with Pytorch tests

With passing builds we upload `torch`, `torchvision`, `torchaudio`, and `pytorch-triton-rocm` wheels to subfolders of the "v2-staging" directory in the nightly release s3 bucket with a public URL at https://rocm.nightlies.amd.com/v2-staging/

Only with passing Torch tests we promote passed wheels to the "v2" directory in the nightly release s3 bucket with a public URL at https://rocm.nightlies.amd.com/v2/

If no runner is available: Promotion is blocked by default. Set `bypass_tests_for_releases=true` for exceptional cases under [`amdgpu_family_matrix.py`](/build_tools/github_actions/amdgpu_family_matrix.py)

## Advanced build instructions

### Other ways to install the rocm packages

The `rocm[libraries,devel]` packages can be installed in multiple ways:

- (As above) during the `build_prod_wheels.py build` subcommand

- Using the more tightly scoped `build_prod_wheels.py install-rocm` subcommand:

  ```bash
  build_prod_wheels.py
      --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/ \
      install-rocm
  ```

- Manually installing from a release index:

  ```bash
  # From therock-nightly-python
  python -m pip install \
    --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/ \
    rocm[libraries,devel]

  # OR from therock-dev-python
  python -m pip install \
    --index-url https://d25kgig7rdsyks.cloudfront.net/v2/gfx110X-dgpu/ \
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
    --target=gfx110X-dgpu \
    --output-dir=$HOME/.therock/17123441166/artifacts \
    --all

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

### Recommendation: avoid using patch files if possible

If you want to make changes to PyTorch source code, prefer in this order:

1. Contributing to upstream `main` branches
1. Contributing to upstream `release/` branches
1. Maintaining downstream git forks with their own branches
   - This allows for standard git workflows to merge, resolve conflicts, etc.
1. Using an upstream or downstream branch with patch files
   - Patch files can be applied on top of git history and do not require commit
     access in an upstream repository but require extra effort to keep current
     and provide no path to being merged

> [!WARNING]
> We carry some patch files out of necessity while waiting on upstream consensus
> and code review, but maintaining patch files carries significant overhead. For
> long term support, we use downstream git forks like the
> [ROCm PyTorch release branches](#rocm-pytorch-release-branches).

### About patch files and patchsets

Patches are commits saved to text files generated by
[`git format-patch`](https://git-scm.com/docs/git-format-patch) that can be
replayed later using [`git am`](https://git-scm.com/docs/git-am).

We store patches in a directory hierarchy within the
[`patches/` subdirectory](./patches/):

```text
pytorch/                 <-- project name (for pytorch_torch_repo.py)
  main/                  <-- patchset name (typically a git tag or branch name)
    pytorch/             <-- folder within the project
      base/              <-- patches to apply before hipify
        0001-COMMIT_MESSAGE.patch
        0002-COMMIT_MESSAGE.patch
      hipified/          <-- patches to apply after hipify
        0001-COMMIT_MESSAGE.patch
    third_party/         <-- another folder within the project
      fbgemm/
        base/
        hipified/
          0001-COMMIT_MESSAGE.patch
  v2.7.0/                <-- a different patchset name
    pytorch/
      base/
        0001-COMMIT_MESSAGE.patch

pytorch_audio/
  main/
    ...
  v2.7.0/
    ...
```

### Checking out and applying patches

Each `pytorch_*_repo.py` scripts handle a few aspects of working with each
associated repository:

1. Clone the repository as needed from `--gitrepo-origin` to the `--repo-name`
   folder under path `--repo`
1. Checkout the `--repo-hashtag` git ref/tag
1. Apply "base" patches from the `--patchset` subfolders of `--patch-dir`
1. Run 'hipify' on the repository, editing source files and committing the
   changes
1. Apply "hipified" patches from the `--patchset` subfolders of `--patch-dir`

After running one of the `pytorch_*_repo.py` scripts you should have a
repository with history like this:

```console
$ python pytorch_torch_repo.py checkout --repo-hashtag main --patchset main
$ cd src/pytorch
$ git log --oneline

bbb55555555 (HEAD) Example 'hipified' patch 2
bbb44444444 Example 'hipified' patch 1
bbb33333333 (tag: THEROCK_HIPIFY_DIFFBASE) DO NOT SUBMIT: HIPIFY
bbb22222222 Example 'base' patch 2
bbb11111111 Example 'base' patch 1
aaa22222222 (tag: THEROCK_UPSTREAM_DIFFBASE, origin/main) Example upstream commit 2
aaa11111111 Example upstream commit 1
```

Note the sequence of commits and tags that were created:

- `main` is checked out initially and is tagged `THEROCK_UPSTREAM_DIFFBASE`
- "base" patches are added _before_ running hipify
- hipify is run and its changes are tagged `THEROCK_HIPIFY_DIFFBASE`
- "hipified" patches are added _after_ running hipify

> [!NOTE]
> For repositories with no patches, this simplifies to "clone the repo, checkout
> a branch, then run hipify".
>
> If you maintain a downstream fork like https://github.com/ROCm/pytorch,
> aim to have a branch that only depends on hipify running on it.

### Saving new patches

To create patches, modify the commits after the `THEROCK_UPSTREAM_DIFFBASE` tag
then run the `save-patches` subcommand of the relevant source management script.
If you want a patch to be applied _before_ hipify, you can put it before
`THEROCK_HIPIFY_DIFFBASE` while editing git history or you can move the patch
file from the `hipified` patch folder to the `base` patch folder.

- If a patch modifies code at a `*/cuda/*` path, it likely needs to be a 'base'
  patch that is applied before hipify is run.
- If a patch modifies code at a `*/hip/*` path, it should either be a
  'hipified' patch or should be reworked to modify `*/cuda/*` code that gets
  correctly transformed by hipify.
- Most other patches can be categorized as 'base' or 'hipified', though
  'hipified' is simplest to construct based on git history

Here is a complete example:

```bash
cd external-builds/pytorch
python -m venv .venv && source .venv/bin/activate

# Checkout, applying patches and running hipify.
python pytorch_torch_repo.py checkout --repo-hashtag main --patchset main

# Switch into the new source directory and make a change.
pushd pytorch
git log --oneline -n 5
# f3d83d2abee (HEAD) Support FLASH_ATTENTION, MEM_EFF_ATTENTION via. aotriton on windows
# b3787ab8e90 Include native_transformers srcs to fix link errors.
# cb53ee6fd45 (tag: THEROCK_HIPIFY_DIFFBASE) DO NOT SUBMIT: HIPIFY
# 96682103026 (tag: THEROCK_UPSTREAM_DIFFBASE, origin/main) Allow bypasses for Precompile when guards, etc. cannot be serialized (#160902)
# 3f5a8e2003f Fix torchaudio build when TORCH_CUDA_ARCH_LIST is not set (#161084)
touch test.txt
git add -A
git commit -m "Test commit for patch saving"
popd

# Save the patch.
python pytorch_torch_repo.py save-patches --repo-hashtag main

git status
# Untracked files:
#   (use "git add <file>..." to include in what will be committed)
#         patches/pytorch/main/pytorch/hipified/0003-Test-commit-for-patch-saving.patch
```

Since the commit was added on top of `THEROCK_HIPIFY_DIFFBASE`, it was written
to the `hipified/` patch folder. The patch could stay there or be moved to the
`base/` patch folder.

### Alternate branches / patch sets

> [!TIP]
> Each branch combination below can also use specific commits by selecting a
> patchset. For example, this will fetch PyTorch at
> [pytorch/pytorch@3e2aa4b](https://github.com/pytorch/pytorch/commit/3e2aa4b0e3e971a81f665a9a6d803683452c022d)
> using the patches from
> [`patches/pytorch/main/pytorch/`](./patches/pytorch/main/pytorch/):
>
> ```bash
> python pytorch_torch_repo.py checkout \
>   --repo-hashtag 3e2aa4b0e3e971a81f665a9a6d803683452c022d \
>   --patchset main
> ```

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
stable versions for a longer period of time, backport branches are maintained.
In order to check out and build one of these, use the following instructions:

In general, we regularly build PyTorch nightly from upstream sources and the
most recent stable backport. Generally, backports are only supported on Linux
at present.

Backport branches have `related_commits` files that point to specific
sub-project commits, so the main torch repo must be checked out first to
have proper defaults.

You are welcome to maintain your own branches that extend one of AMD's.
Change origins and tags as appropriate.

#### ROCm release v2.7.x branch

NOTE: Presently broken at runtime on a HIP major version incompatibility in the
pre-built aotriton (#1025). Must build with
`USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0` until fixed.

```bash
python pytorch_torch_repo.py checkout \
  --gitrepo-origin https://github.com/ROCm/pytorch.git \
  --repo-hashtag release/2.7 \
  --patchset rocm_2.7
python pytorch_audio_repo.py checkout --require-related-commit
python pytorch_vision_repo.py checkout --require-related-commit
python pytorch_triton_repo.py checkout
```
