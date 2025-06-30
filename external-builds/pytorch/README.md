# Build PyTorch with ROCm support

This directory provides tooling for building PyTorch compatible with TheRock's
ROCm dist packages.

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

| Feature                  | Linux support | Windows support |
| ------------------------ | ------------- | --------------- |
| PyTorch                  | ✅ Supported  | ✅ Supported    |
| torchvision              | ✅ Supported  | ⚪ Unknown      |
| torchaudio               | ✅ Supported  | ⚪ Unknown      |
| Flash attention (Triton) | ✅ Supported  | 🟡 In progress  |

## Build instructions

### Prerequisites and setup

You will need either a source build or binary distribution of the dist packages.

- For binary distributions, see [RELEASES.md](../../RELEASES.md). Both tarballs
  and Python packages should include the necessary files.

  - Note: Windows ROCm Python packages are not yet available.

- For source builds:

  1. Follow the [building from source](../../README.md#building-from-source)
     instructions.
  1. Build the `therock-dist` target:
     ```bash
     cmake --build build --target therock-dist
     ```
  1. Use the `build/dist/rocm` directory.

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

### Build PyTorch, PyTorch vision and PyTorch audio on Linux

> [!WARNING]
> This is being migrated to `build_prod_wheels.py`, using `rocm` (sdk) wheels.

```bash
./checkout_and_build_all.sh
```

### Build PyTorch on Windows (or the old way on Linux)

> [!WARNING]
> This is being migrated to `build_prod_wheels.py`, using `rocm` (sdk) wheels.

#### Step 1: Preparing sources

```bash
# Checks out the most recent stable release branch of PyTorch, hipifies and
# applies patches.
python pytorch_torch_repo.py checkout
```

#### Step 2: Install Deps

Python deps:

```bash
pip install -r pytorch/requirements.txt
pip install mkl-static mkl-include
```

#### Step 3: Setup and Build

On Linux:

```bash
export CMAKE_PREFIX_PATH="$(realpath ../../build/dist/rocm)"
(cd src && USE_KINETO=OFF python setup.py develop)
```

On Windows:

```bash
bash build_pytorch_windows.sh gfx1100
```

## Running/testing PyTorch

### Windows DLL setup

> [!IMPORTANT]
> This will no longer be necessary after `rocm` wheels are available.

On Windows, PyTorch needs to be able to find DLL files from the `dist/rocm`
directory. This can be achieved by either

- Extending your `PATH` to include that directory:

  ```bash
  set PATH=..\..\build\dist\rocm\bin;%PATH%
  ```

- Creating a "fat wheel" that bundles the files together (see the next section).

### Bundling PyTorch and ROCm together into a "fat wheel"

> [!IMPORTANT]
> This will no longer be necessary after `rocm` wheels are available.

By default, Python wheels produced by the PyTorch build do not include ROCm
binaries. Instead, they expect those binaries to be installed elsewhere on the
system. A "fat wheel" bundles the ROCm binaries into the same wheel archive to
produce a standalone install including both PyTorch and ROCm, with all necessary
patches to shared library / DLL loading for out of the box operation.

To produce such a fat wheel, see `windows_patch_fat_wheel.py` and a future
equivalent script for Linux.

### Running PyTorch smoketests

We have some basic smoketests to check that the build succeeded and the
environment setup is correct. See [smoketests](./smoke-tests/) for details, or
just run:

```bash
pytest -v smoketests
```

## Development instructions

To create patches

1. Commit your change(s) within the relevant source folder(s)
1. Run the `save-patches` subcommand of the relevant source management script(s)
