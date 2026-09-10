# ROCm Computer Vision Libraries

This directory contains computer vision libraries for AMD CPUs and GPUs.

## Libraries

### RPP (ROCm Performance Primitives)

A comprehensive, high-performance computer vision library for AMD CPUs and GPUs,
with HOST and HIP backends. **Part of the default ROCm distribution** — included
in `amdrocm-core` and built by default on Linux.

Depends on the HIP runtime, the half-precision floating-point headers, and
OpenMP (used to parallelize the host/CPU code paths, not for GPU offload).

Enable flag: `-DTHEROCK_ENABLE_RPP=ON`
Disable group: `-DTHEROCK_ENABLE_CV_LIBS=OFF`

Source: [`rocm-libraries/projects/rpp`](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rpp)

### MIVisionX

A computer vision toolkit built on AMD OpenVX, providing GPU-accelerated image
processing and vision kernels. **Opt-in extension — not part of the default ROCm
distribution.** Packaged separately as `amdrocm-vision`.

Depends on RPP, the HIP runtime, and the half-precision floating-point headers.

Enable flag: `-DTHEROCK_ENABLE_MIVISIONX=ON` (Linux only, off by default)

Source: [`github.com/ROCm/MIVisionX`](https://github.com/ROCm/MIVisionX)

## Platform support

| Library   | Linux default                                 | Windows                                               |
| --------- | --------------------------------------------- | ----------------------------------------------------- |
| RPP       | ✅ Built by default                           | 🟡 Experimental; opt-in via `-DTHEROCK_ENABLE_RPP=ON` |
| MIVisionX | ⚪ Opt-in via `-DTHEROCK_ENABLE_MIVISIONX=ON` | ❌ Unsupported                                        |

The Windows CI pipeline does not build cv-libs.

Native packages are produced for Linux only. Windows packaging for RPP would be
a follow-up if/when it graduates from experimental on Windows.
