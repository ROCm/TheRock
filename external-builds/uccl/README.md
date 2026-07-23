# Build UCCL with ROCm support

This directory provides tooling for building UCCL with ROCm Python wheels.

Table of contents:

- [Support status](#support-status)
- [Build instructions](#build-instructions)
- [Running/testing UCCL](#runningtesting-uccl)
- [Development instructions](#development-instructions)

## Support status

| Project / feature | Linux support | Windows support  |
| ----------------- | ------------- | ---------------- |
| UCCL              | ✅ Supported  | ❌ Not Supported |

## Build instructions

Upstream UCCL has a "rocm" build target for system-wide ROCm
installations that currently builds in an Ubuntu 22.04 docker
container.

We have contributed and upstreamed a new build target "therock" for
python packaged ROCm from TheRock. This setup uses that build
target. It is adapted to run in TheRock's PyPA-based docker container
to maximize compabitility.

### Prerequisites and setup

The build script currently requires that TheRock index URL be provided
explicitly and uses that to install python packaged ROCm along with
UCCL prerequisites inside the UCCL build container.

The specific versions of the ROCm python packages are recorded in the
UCCL wheel's dependences, along with compatible pytorch and numpy
python packages needed for UCCL collective and transfer engine
modules.

### Quickstart

Building is a two-step process. First, checkout upstream UCCL sources
to a local directory using the `uccl_repo.py` script. Then execute a
build script to prepare a build container and produce the UCCL wheel
artifact. Note that you need provide an index URL for the python
packaged ROCm matching your GPU's gfx arch.

Example:

```bash
python uccl_repo.py checkout
python build_prod_wheels.py --output-dir outputs \
  --python-version 3.12 \
  --index-url https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu
```

The build script has optional arguments for the name of the directory
with previously checked out UCCL sources (default `uccl`), to target a
specific python version, or to use a specific base image for the UCCL
build container.

The resulting wheel can then be installed like so:

```bash
python3.12 -m venv venv
. venv/bin/activate
pip install --extra-index-url https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu \
  uccl-0.0.1.post4-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl[rocm]
```

Note the use of `--extra-index-url` instead of `--index-url` to
accommodate resolution of non-ROCm dependences of UCCL to be satisfied
by the default PyPI index.

## Testing UCCL

Tests are structured in two tiers, following the same pattern as
`external-builds/pytorch/`.

### Smoke tests

Quick sanity checks that verify the UCCL wheel is installed correctly,
GPU hardware is accessible, and the UCCL Python API is importable.

```bash
python run_uccl_smoke_tests.py -- --log-cli-level=INFO -v
```

### Intranode EP tests

Runs the upstream `test_intranode.py` test via `torchrun` in standalone
mode. This exercises Expert Parallelism dispatch, combine, and tuning
kernels on a single node with multiple GPUs. Requires a UCCL source
checkout for the test files.

```bash
# Checkout UCCL sources first
python uccl_repo.py checkout

# Run with all available GPUs (auto-detected)
python run_uccl_tests.py

# Or specify GPU count
python run_uccl_tests.py --nproc-per-node 4

# Dry-run to see the command without executing
python run_uccl_tests.py --dry-run
```

### CI workflow

The `test_uccl_wheels.yml` workflow runs both test tiers. It can be
triggered manually via `workflow_dispatch` or called from other
workflows via `workflow_call`. See the workflow file for the full list
of inputs.
