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
| torchvision              | ✅ Supported  | 🟡 In progress ([#910](https://github.com/ROCm/TheRock/issues/910))   |
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
> Windows support for these packages is _very_ new so some instructions
> may not work yet. Stay tuned!

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
  # TODO(#910): Support torchvision on Windows
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

  ```bash
  # From the repository root
  mkdir $HOME/.therock/15914707463
  mkdir $HOME/.therock/15914707463/artifacts
  python ./build_tools/fetch_artifacts.py \
    --run-id=15914707463 \
    --target=gfx110X-dgpu \
    --output-dir=$HOME/.therock/15914707463/artifacts \
    --all

  python ./build_tools/build_python_packages.py \
    --artifact-dir=$HOME/.therock/15914707463/artifacts \
    --dest-dir=$HOME/.therock/15914707463/packages
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

### PyTorch Nightly

This checks out the `nightly` branches from https://github.com/pytorch,
tracking the latest pytorch.org nightly release:

- https://github.com/pytorch/pytorch/tree/nightly
- https://github.com/pytorch/audio/tree/nightly
- https://github.com/pytorch/vision/tree/nightly

```
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

```
python pytorch_torch_repo.py checkout \
  --gitrepo-origin https://github.com/ROCm/pytorch.git \
  --repo-hashtag release/2.7 \
  --patchset rocm_2.7
python pytorch_audio_repo.py checkout --require-related-commit
python pytorch_vision_repo.py checkout --require-related-commit
python pytorch_triton_repo.py checkout
```
