# ROCm Computer Vision Libraries

This directory contains computer vision libraries for AMD GPUs.

- **RPP** (ROCm Performance Primitives) -- a comprehensive, high-performance
  computer vision library for AMD processors with a HIP backend.

## Dependencies

RPP depends on the HIP runtime, the half-precision floating-point headers,
and OpenMP.

The library can be individually controlled:

- `-DTHEROCK_ENABLE_RPP=ON`

Or disabled as a group:

- `-DTHEROCK_ENABLE_CV_LIBS=OFF`

## Platform support

RPP is built by default on Linux. On Windows it is **experimental and disabled
by default**; it can be opted into explicitly with `-DTHEROCK_ENABLE_RPP=ON`
(or `-DTHEROCK_ENABLE_CV_LIBS=ON`). The Windows CI pipeline does not build
cv-libs.

There is currently **no Windows native packaging** for RPP (consistent with the
Linux-only media-libs). Native packages are produced for Linux only
(`amdrocm-rpp`, `amdrocm-rpp-devel`, `amdrocm-rpp-test`). Windows packaging would
be a follow-up if/when RPP graduates from experimental on Windows.

## Source Layout

The source code for RPP lives in the
[rocm-libraries](https://github.com/ROCm/rocm-libraries) monorepo:

- `rocm-libraries/projects/rpp`
