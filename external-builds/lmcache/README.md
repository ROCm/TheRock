# Build LMCache with ROCm support

This directory builds LMCache wheels against ROCm and PyTorch packages
produced by TheRock.

## Support status

| Project / feature | Linux support | Windows support  |
| ----------------- | ------------- | ---------------- |
| LMCache           | ✅ Supported  | ❌ Not Supported |

## Build instructions

The build runs in TheRock's pinned manylinux container and consumes packages
from the unified multi-architecture Python index. Each wheel embeds code for
the supported GPU architectures:

```text
gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1250
```

### Prerequisites

- Docker with BuildKit support
- Python 3

### Quickstart

From the root of a TheRock checkout:

```bash
python external-builds/lmcache/lmcache_repo.py checkout

python external-builds/lmcache/build_prod_wheels.py \
  --output-dir outputs
```

The command writes one `lmcache-*.whl` file to `outputs`. The default upstream
LMCache checkout and ROCm/PyTorch versions are the combination validated by
this integration. The wheel version includes the ROCm version as a local
suffix so builds of the same LMCache revision against different ROCm releases
do not overwrite one another in the package index.

To test another LMCache revision, select it during checkout:

```bash
python external-builds/lmcache/lmcache_repo.py checkout \
  --repo-hashtag <branch-tag-or-commit>
```

To use another compatible TheRock package set, override the index and both
versions together:

```bash
python external-builds/lmcache/build_prod_wheels.py \
  --output-dir outputs \
  --rocm-index-url <index-url> \
  --rocm-version <rocm-version> \
  --torch-version <torch-version>
```

### Install the wheel

Install matching TheRock PyTorch device packages before installing LMCache:

```bash
python3.12 -m venv .venv
. .venv/bin/activate

python -m pip install \
  --index-url https://rocm.prereleases.amd.com/whl-multi-arch/ \
  "torch[device-gfx942]==2.11.0+rocm7.14.0rc3"

python -m pip install \
  --extra-index-url https://rocm.prereleases.amd.com/whl-multi-arch/ \
  outputs/lmcache-*.whl
```

`--extra-index-url` allows LMCache's non-ROCm dependencies to resolve from
PyPI while the matching TheRock packages remain available.

### Build script options

| Option                | Description                                  |
| --------------------- | -------------------------------------------- |
| `--output-dir`        | Output directory for the wheel (required)    |
| `--lmcache-dir`       | Local LMCache source checkout                |
| `--build-device-arch` | Device package used by the build environment |
| `--rocm-arches`       | GPU architectures embedded in the wheel      |
| `--rocm-index-url`    | TheRock multi-arch package index             |
| `--rocm-version`      | Pinned ROCm package version                  |
| `--torch-version`     | Pinned PyTorch package version               |
| `--python-version`    | CPython version supplied by the build image  |
| `--image`             | Pinned manylinux build image                 |
| `--max-jobs`          | Maximum parallel compiler jobs               |
| `--no-cache`          | Disable the Docker build cache               |

## Test the installation

Run the smoke test on a machine with a matching AMD GPU:

```bash
python external-builds/lmcache/run_lmcache_smoke_test.py \
  --expected-arch gfx942
```

The test checks the PyTorch ROCm runtime, executes a GPU operation, and imports
the native LMCache extensions.

## Continuous integration

`build_linux_lmcache_wheels.yml` checks out the pinned LMCache source and
builds the wheel on a runner with Docker. `test_lmcache_wheels.yml` installs
that exact artifact in a no-ROCm container on a `gfx942` runner, runs package
and ROCm sanity checks, the LMCache smoke test, and the upstream GPU kernel
tests.

S3 publication is disabled by default. A trusted manual or reusable workflow
invocation can set `publish_to_s3` after selecting the release type. The tested
multi-architecture wheel is then uploaded to
`s3://therock-<release-type>-python/v4/whl/` and served through the matching
`whl-multi-arch` package index.
