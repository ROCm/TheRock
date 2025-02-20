# Build ROCM PyTorch

There is nothing special about this build procedure except that it is meant
to run as part of the ROCM CI and development flow and leaves less room for
interpretation with respect to golden path in upstream docs.

This incorporates advice from:

- https://github.com/pytorch/pytorch#from-source
- `.ci/manywheel/build_rocm.sh` and friends

Note that the above statement is currently aspirational as we contain some
patches locally until they can be upstreamed. See the `patches` directory.

## Step 0: Prep venv

It is highly recommended to use a virtual environment unless if in a throw-away
container/CI environment.

```
python -m venv .venv
source .venv/bin/activate
```

## Step 1: Preparing sources

```
# Checks out the most recent stable release branch of PyTorch, hipifies and
# applies patches.
./ptbuild checkout
```

## Step 2: Install Deps

Python deps:

```
pip install -r src/requirements.txt
pip install mkl-static mkl-include
```

## Step 3: Setup and Build

```
export CMAKE_PREFIX_PATH="$(realpath ../../build/dist/rocm)"
(cd src && USE_KINETO=OFF python setup.py develop)
```
