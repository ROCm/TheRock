# Build ROCM PyTorch

## Build PyTorch, PyTorch vision and PyTorch audio on Linux

- cd external-builds/pytorch
- ./checkout_and_build_all.sh

## Build PyTorch on Windows or on old way in Linux

There is nothing special about this build procedure except that it is meant
to run as part of the ROCM CI and development flow and leaves less room for
interpretation with respect to golden path in upstream docs.

This incorporates advice from:

- https://github.com/pytorch/pytorch#from-source
- `.ci/manywheel/build_rocm.sh` and friends

Note that the above statement is currently aspirational as we contain some
patches locally until they can be upstreamed. See the `patches` directory.

### Step 0: Prep venv

It is highly recommended to use a virtual environment unless if in a throw-away
container/CI environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 1: Preparing sources

```bash
# Checks out the most recent stable release branch of PyTorch, hipifies and
# applies patches.
./ptbuild.py checkout
```

### Step 2: Install Deps

Python deps:

```bash
pip install -r src/requirements.txt
pip install mkl-static mkl-include
```

### Step 3: Setup and Build

On Linux:

```bash
export CMAKE_PREFIX_PATH="$(realpath ../../build/dist/rocm)"
(cd src && USE_KINETO=OFF python setup.py develop)
```

On Windows:

```bash
# First make local edits to e.g. PYTORCH_ROCM_ARCH and PYTORCH_SRC_DIR, then...
bash build_pytorch_windows.sh
```
