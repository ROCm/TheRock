# Build PyTorch with ROCm support

This directory provides tooling for building PyTorch with ROCm Python wheels.

> [!TIP]
> If you want to install our prebuilt PyTorch packages instead of building them
> from source, see [RELEASES.md](/RELEASES.md) instead.

There is nothing special about these build procedures except that they are meant
to run as part of the ROCm CI and development flow and thus leave less room for
interpretation with respect to golden path in upstream docs.

This incorporates advice from:

- https://github.com/pytorch/pytorch#from-source
- `.ci/manywheel/build_rocm.sh` and friends

Note that the above statement is currently aspirational as we contain some
patches locally until they can be upstreamed. See the
[`patches/` directory](./patches/).

## Feature support status

| Feature                  | Linux support | Windows support                                                       |
| ------------------------ | ------------- | --------------------------------------------------------------------- |
| PyTorch                  | ✅ Supported  | ✅ Supported                                                          |
| torchaudio               | ✅ Supported  | ✅ Supported                                                          |
| torchvision              | ✅ Supported  | ✅ Supported                                                          |
| Flash attention (Triton) | ✅ Supported  | 🟡 In progress ([#1040](https://github.com/ROCm/TheRock/issues/1040)) |

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
> PyTorch builds aotriton locally, but without the kernel images.
> Make sure to copy the `aotriton.images` folder from an existing
> aotriton linux build (`<aotriton_build_dir>/lib/aotriton.images`) and copy
> that folder into your local pytorch lib directory: `<pytorch_dir>/torch/lib/`.
> This is a temporary measure for manually producing aotriton builds.
> NOTE: This will not work without the [corresponding patch](./patches/pytorch/main/pytorch/hipified/0004-Support-FLASH_ATTENTION-MEM_EFF_ATTENTION-via.-aotri.patch) for the main branch.
>
> On Windows, aotriton uses `dladdr`, which is implemented through
> [dlfcn-win32](https://github.com/dlfcn-win32/dlfcn-win32), which unfortunately
> uses `GetModuleFileNameA` (ANSI version) to get the base directory of
> `libaotriton.so`. This means, if `libaotriton.so` is put under a path with
> characters that cannot represented in current code page, the loading of GPU
> kernels will fail.
> See https://github.com/ROCm/aotriton/commit/e1be21d80b25f46139c2e3b4b0615e0279feccac
> For possible fixes. A proper fix is planned and will eventually be added.
>
> NOTE: If you use ccache and face "invalid argument" errors during the aotriton build,
> disable ccache and try again.

### Quickstart

It is highly recommended to use a virtual environment unless working within a
throw-away container or CI environment.

- On Linux:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

- On Windows:

  ```bash
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

Now checkout repositories:

- On Linux, use default paths (nested under this folder) and default branches:

  ```bash
  python pytorch_torch_repo.py checkout
  python pytorch_audio_repo.py checkout
  python pytorch_vision_repo.py checkout
  ```

- On Windows, use shorter paths to avoid command length limits and `main` branches:

  ```bash
  python pytorch_torch_repo.py checkout --repo C:/b/pytorch --repo-hashtag main
  python pytorch_audio_repo.py checkout --repo C:/b/audio --repo-hashtag main
  python pytorch_vision_repo.py checkout --repo C:/b/vision --repo-hashtag main
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
    --install-rocm --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/ \
    --output-dir $HOME/tmp/pyout
  ```

- On Windows:

  ```bash
  python build_prod_wheels.py build \
    --install-rocm --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/ \
    --pytorch-dir C:/b/pytorch \
    --pytorch-audio-dir C:/b/audio \
    --pytorch-vision-dir C:/b/vision \
    --output-dir %HOME%/tmp/pyout
  ```

## Advanced build instructions

### Other ways to install the rocm packages

The `rocm[libraries,devel]` packages can be installed in multiple ways:

- (As above) during the `build_prod_wheels.py build` subcommand

- Using the more tightly scoped `build_prod_wheels.py install-rocm` subcommand:

  ```bash
  build_prod_wheels.py
      --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/ \
      install-rocm
  ```

- Manually installing from a release index:

  ```bash
  # From therock-nightly-python
  python -m pip install \
    --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/ \
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

To produce such a fat wheel, see `windows_patch_fat_wheel.py` and a future
equivalent script for Linux.

## Running/testing PyTorch

### Running PyTorch smoketests

We have some basic smoketests to check that the build succeeded and the
environment setup is correct. See [smoke-tests](./smoke-tests/) for details, or
just run:

```bash
pytest -v smoke-tests
```

### Running full PyTorch tests

See https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#testing-the-pytorch-installation

<!-- TODO(erman-gurses): update docs here -->

## Development instructions

To create patches

1. Commit your change(s) within the relevant source folder(s)
1. Run the `save-patches` subcommand of the relevant source management script(s)

## Alternate Branches / Patch Sets

We support several PyTorch branches with associated patch sets on Linux and
Windows. The intent is to support the latest upstream PyTorch code (i.e. `main`
or `nightly`) as well as recently published release branches which users depend
on.

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

### PyTorch main

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

### PyTorch Nightly

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

### ROCm PyTorch Release Branches

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

### v2.7.x

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
