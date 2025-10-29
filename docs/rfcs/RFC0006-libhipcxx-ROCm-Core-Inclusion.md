---
author: Marco Grond (marco-grond)
created: 2025-10-16
modified: 2025-10-29
status: draft
---

# Inclusion of libhipcxx in ROCm-Core

This RFC propses the inclusion of libhipcxx into the ROCm-Core/TheRock repository and to include it as a part of future ROCm-Core releases.

## History and Current Status

[libhipcxx](https://github.com/ROCm/libhipcxx) is based on the [libcudacxx](https://github.com/NVIDIA/cccl/tree/main/libcudacxx) library,
which is the CUDA C++ Standard Library, providing an implementation of the C++ Standard Library that works in both host and device code.
This library additionally provides abstractions for CUDA-specific hardware features like synchronization primitives, cache control,
atomicss, and more.

libhipcxx was originally created as a dependency for ROCm-DS, AMD's counterpart to the NVIDIA RAPIDS data science toolkit. The library is
maintained by the AIOSS team under Vish Vadlamani.

## Dependencies

In addition to the core ROCm-DS libraries, libhipcxx has also become a dependency for rocThrust and hipCUB. These two libraries are based
on NVIDIA's Thrust and CUB respectively, which together with libcudacxx form the
[CUDA Core Compute Libraries (CCCL)](https://github.com/NVIDIA/cccl). NVIDIA has moved the backend code from CUB and Thrust into
libcudacxx, which has made it a core dependency for these libraries. Furthermore, libcudacxx has become a dependency for PyTorch through
CUB and Thrust. Similarly, future updates of rocThrust and hipCUB will be dependent on libhipcxx, and by extension so will PyTorch.

## Proposal

Currently, the development branch of libhipcxx is hosted on the AMD-AIOSS GitHub organization, while the public repo falls within ROCm.
Furthermore, libhipcxx is currently included as a part of the ROCm-DS toolkit release and has not been released as a part of ROCm. It
has had an independent development, QA, and release process from other libraries that were included in ROCm. This proposal is to have
libhipcxx be included into ROCm-Core, starting with ROCm 7.10, and to have libhipcxx be included in the release and validation pipelines
setup for ROCm-Core.

Development of the library will remain with the current team, however this move will enable rocThrust and hipCUB to keep pace with the
development of their NVIDIA counterparts, while also grouping together low level libraries as well as libraries that are dependencies
for PyTorch. Since one of the focuses of TheRock is to enable PyTorch and include all of its dependencies, the inclusion of libhipcxx
is of the utmost importance.

Testing pipelines should be updated to include libhipcxx validation and to ensure that the library integrates with other libraries
included in TheRock. Release pipelines should also be updated to include libhipcxx in all future ROCm-Core releases.

## Revision History

- 2025-10-16: marco-grond: Create skeleton
- 2025-10-20: marco-grond: Filled out content
- 2025-10-29: marco-grond: Updated RFC number
