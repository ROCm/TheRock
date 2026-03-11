# ROCM BLAS Libraries

This directory contains the suite of projects which collectively implement
ROCM BLAS APIs on AMD GPUs.

Some libraries are low-level, implementation libraries. The following are
user-level API libraries:

- hipBLASLt
- hipBLAS
- hipSOLVER
- hipSPARSE
- hipSPARSELt

Currently, all component libraries are built individually, but a future rework
will nest them all under the same CMake project for better optimization of the
build graph.
