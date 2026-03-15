# Build UCX with ROCm support

This directory provides tooling for building UCX (Unified Communication X)
with ROCm support.

## Support status

| Project / feature | Linux support | Windows support  |
| ----------------- | ------------- | ---------------- |
| UCX               | Supported     | Not Supported    |

## Build instructions

Building is a two-step process. First, checkout upstream UCX sources using the
`ucx_repo.py` script. Then run the build script to compile UCX with ROCm
support.

### Prerequisites

- ROCm installation (from TheRock build or system install)
- autotools (`autoconf`, `automake`, `libtool`)
- Standard build tools (`gcc`, `make`)

### Quickstart

```bash
# 1. Checkout UCX sources
python ucx_repo.py checkout

# 2. Build UCX with ROCm
python build_ucx.py \
    --rocm-path /path/to/rocm \
    --output-dir ./output
```

### Checkout options

```bash
# Checkout a specific branch/tag
python ucx_repo.py checkout --repo-hashtag v1.18.0

# Shallow clone for faster checkout
python ucx_repo.py checkout --depth 1
```

### Build options

```bash
# Specify number of parallel jobs
python build_ucx.py \
    --rocm-path /path/to/rocm \
    --output-dir ./output \
    --jobs 16
```

## Testing UCX with ROCm

After building, the UCX gtest binary (with ROCm-specific tests) is located at:

```
<ucx-source>/build/test/gtest/gtest
```

Run ROCm-specific tests:

```bash
./build/test/gtest/gtest --gtest_filter='*rocm*'
```
