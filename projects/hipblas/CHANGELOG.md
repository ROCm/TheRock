# Changelog for hipBLAS

Documentation for hipBLAS is available at
[https://rocm.docs.amd.com/projects/hipBLAS/en/latest/](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/).

## hipBLAS 3.3.0

### Added

* Added the `hipblasHerkEx()` API.
* Added the `hipblasSyrkEx()` API.

## hipBLAS 3.2.0 for ROCm 7.2

### Resolved issues
* Corrected client memory use counts for the `HIPBLAS_CLIENT_RAM_GB_LIMIT` environment variable.
* Fix to avoid false Clang static analysis warnings.

## hipBLAS 3.1.0 for ROCm 7.1

### Added

* `--clients-only` build option to only build clients against a prebuilt library.
* gfx1103, gfx1150, gfx1151, gfx1200, and gfx1201 support to clients.
* FORTRAN enabled for the Microsoft Windows build and tests.
* Additional reference library fallback options added.

### Changed

* Improve the build time for clients by removing `clients_common.cpp` from the hipblas-test build.

## hipBLAS 3.0.0 for ROCm 7.0

### Added

* Added the `hipblasSetWorkspace()` API
* Support for codecoverage tests

### Changed

* HIPBLAS_V2 API is now the only available API using `hipComplex` and `hipDatatype` types
* Documentation updates
* Verbose compilation for `hipblas.cpp`

### Removed

* `hipblasDatatype_t` type
* `hipComplex` and `hipDoubleComplex` types
* Support code for non-production gfx targets

### Resolved issues

* The build time `CMake` configuration for the dependency on `hipBLAS-common` is fixed
* Compiler warnings for unhandled enums have been resolved

##  hipBLAS 2.4.0 for ROCm 6.4.0

### Changed

* Updated build dependencies

### Resolved issues

* Fixed Windows reference library interface for rocSOLVER functions for hipBLAS clients

## hipBLAS 2.3.0 for ROCm 6.3.0

### Added

* Level 3 functions have an additional ILP64 API for both C and FORTRAN (_64 name suffix) with int64_t function arguments.

### Changed

* amdclang is used as the default compiler instead of g++
* Added a dependency on the hipblas-common package

## hipBLAS 2.2.0 for ROCm 6.2.0

### Additions

* Level 2 functions have additional ILP64 API for both C and FORTRAN (_64 name suffix) with int64_t function arguments
* Level 1 "_ex" functions have additional ILP64 API

### Changes

* install.sh script invokes rmake.py script, along with various improvements within the build scripts
* Library dependencies in install.sh script have been changed from "rocblas" and "rocsolver" to the development packages
  "rocblas-dev" and "rocsolver-dev"
* Linux AOCL dependency updated to release 4.2 gcc build
* Windows vcpkg dependencies updated to release 2024.02.14

## hipBLAS 2.1.0 for ROCm 6.1.0

### Additions

* New build option to automatically use hipconfig --platform to determine HIP platform
* Level 1 functions have additional ILP64 API for both C and Fortran (`_64` name
  suffix) with int64_t function arguments
* New functions hipblasGetMathMode and hipblasSetMathMode

### Deprecations

* USE_CUDA build option; use HIP_PLATFORM=amd or HIP_PLATFORM=nvidia to override hipconfig

### Changes

* Some Level 2 function argument names have changed from `m` to `n` to match legacy BLAS; there
  was no change in implementation.
* Updated client code to use YAML-based testing
* Renamed `.doxygen` and `.sphinx` folders to `doxygen` and `sphinx`, respectively
* Added CMake support for documentation

## hipBLAS 2.0.0 for ROCm 6.0.0

### Additions

* New option to define `HIPBLAS_USE_HIP_BFLOAT16` to switch API to use the `hip_bfloat16` type
* New `hipblasGemmExWithFlags` API

### Deprecations

* `hipblasDatatype_t`; use `hipDataType` instead
* `hipblasComplex`; use `hipComplex` instead
* `hipblasDoubleComplex`; use `hipDoubleComplex` instead
* Use of `hipblasDatatype_t` for `hipblasGemmEx` for compute-type; use `hipblasComputeType_t` instead

### Removals

* `hipblasXtrmm` (calculates B <- alpha * op(A) * B) has been replaced with `hipblasXtrmm` (calculates
  C <- alpha * op(A) * B)

## hipBLAS 1.1.0 for ROCm 5.7.0

### Changes

* Updated documentation requirements

### Dependencies

* rocSOLVER now depends on rocSPARSE

## hipBLAS 1.0.0 for ROCm 5.6.0

### Changes

* Added const qualifier to hipBLAS functions (swap, sbmv, spmv, symv, trsm) where missing

### Removals

* `hipblasInt8Datatype_t enum`
* `hipblasSetInt8Datatype`
* `hipblasGetInt8Datatype functions`

### Deprecations

* In-place trmm will be replaced by trmm that includes both in-place and out-of-place functionality

## hipBLAS 0.54.0 for ROCm 5.5.0

### Additions

* Optional opt-in to use the `__half for hipblasHalf` type (with C++) when you define
  `HIPBLAS_USE_HIP_HALF`
* added scripts to plot performance for multiple functions
* data driven hipblas-bench and hipblas-test execution via external yaml format data files
* client smoke test added for quick validation using command hipblas-test --yaml hipblas_smoke.yaml

### Fixes

* Data type conversion functions support more rocBLAS and cuBLAS data types
* `geqrf` now returns successfully when nullptrs are passed with n == 0 || m == 0
* `getrs` now returns successfully when given nullptrs with corresponding size = 0
* `getrs` gives info = -1 when transpose is not an expected type
* `gels` now returns successfully when given nullptrs with corresponding size = 0
* `gels` now gives info = -1 when transpose is not in ('N', 'T') for real cases and not in ('N', 'C') for
  complex cases

### Changes

* Changed reference code for Windows to OpenBLAS
* hipBLAS client executables all now begin with the `hipblas-` prefix

## hipBLAS 0.53.0 for ROCm 5.4.0

### Additions

* Allow for selection of int8 data type
* Added support for `hipblasXgels` and `hipblasXgelsStridedBatched` operations (with s,d,c,z precisions),
  only supported with rocBLAS backend
* Added support for `hipblasXgelsBatched` operations (with s,d,c,z precisions)

## hipBLAS 0.52.0 for ROCm 5.3.0

### Additions

* New `--cudapath` option in `install.sh`, which allows you to specify the CUDA build you want to use
* New `--installcuda` option in `install.sh` to install CUDA using a package manager (this can also be
  used with the new `--installcudaversion` option that allows you to specify the CUDA version you want
  to install)

### Fixes

* `#includes` now support a compiler version
* Fixed client dependency support in `install.sh`

## hipBLAS 0.51.0 for ROCm 5.2.0

### Additions

* New packages for test and benchmark executables on all supported operating systems using CPack
* Added file and folder reorganization changes with backward compatibility support for `rocm-cmake`
  wrapper functions
* Added user-specified initialization option to `hipblas-bench`

### Fixes

* Version gathering in performance-measuring script

## hipBLAS 0.50.0 for ROCm 5.1.0

### Additions

* `hipblas-test` output now has library version and device information
* New `--rocsolver-path` command line option that you can use to specify a path (absolute or relative)
  to the pre-built rocSOLVER
* Added `--cmake_install` command line option to update CMake to the minimum version
* Added `cmake-arg` parameter to pass in cmake arguments while building
* ReadtheDocs infrastructure support for the hipBLAS documentation

### Fixes

* Added `hipblasVersionMinor` (` hipblaseVersionMinor` remains for backwards compatibility)
* Doxygen warnings in `hipblas.h` header file

### Changes

* `rocblas-path` command line option can be specified as absolute or relative path
* Help message improvements in `install.sh` and `rmake.py`
* Updated GoogleTest dependency from 1.10.0 to 1.11.0

## hipBLAS 0.49.0 for ROCm 5.0.0

### Additions

- `hipblas-bench` rocSOLVER functions
- Added `ROCM_MATHLIBS_API_USE_HIP_COMPLEX` to opt-in to use `hipFloatComplex` and
  `hipDoubleComplex`
- Compilation warning for future trmm changes
- `hipblas.h` documentation
- Added option to forgo pivoting for getrf and getri when ipiv is nullptr
- Code coverage option

### Fixes

* Use of incorrect `HIP_PATH` when building from source
* Windows packaging
* Allowing negative increments in `hipblas-bench`
* Removed boost dependency

## hipBLAS 0.48.0 for ROCm 4.5.0

### Additions

- Additional support for `hipblas-bench`
- `HIPBLAS_STATUS_UNKNOWN` for unsupported backend status codes

### Fixes

* Avoid large offset overflow for `gemv` and `hemv` in `hipblas-test`

### Changes

* Packaging has been split into a runtime package (`hipblas`) and a development package
  (`hipblas-devel`):
  The development package depends on the runtime package. When installing the runtime package,
  the package manager will suggest the installation of the development package to aid users
  transitioning from the previous version's combined package. This suggestion by package manager is
  for all supported operating systems (except CentOS 7) to aid in the transition. The `suggestion`
  feature in the runtime package is introduced as a deprecated feature and will be removed in a future
  ROCm release.

## hipBLAS 0.46.0 for ROCm 4.3.0

### Additions

* `hipblasStatusToString`

### Fixes

* Added `catch()` blocks around API calls to prevent the leak of C++ exceptions

## hipBLAS 0.44.0 for ROCm 4.2.0

### Additions

* Updates for rocBLAS `gemm_ex` changes: When using the rocBLAS backend, hipBLAS queries the
  preferable layout of int8 data passed to `gemm_ex` and passes in the resulting flag (you must specify
  your preferred data format when calling `gemm_ex` with a rocBLAS backend)
* Added `hipblas-bench` with support for `copy`, `swap`, and `scal`

## hipBLAS 0.42.0 for ROCm 4.1.0

### Additions

* Added the following functions, which include batched and strided-batched support with the rocBLAS
  backend:
  * `axpy_ex`
  * `dot_ex`
  * `nrm2_ex`
  * `rot_ex`
  * `scal_ex`

### Fixes

* Complex unit test bug caused by incorrect `caxpy` and `zaxpy` function signatures

## hipBLAS 0.40.0 for ROCm 4.0.0

### Additions

* Added a changelog
* `hipblas-bench`, with support for `gemv`, `trsm`, and `gemm`
* rocSOLVER is now a CPack dependency

## hipBLAS 0.38.0 for ROCm 3.10.0

### Additions

* `hipblasSetAtomicsMode` and `hipblasGetAtomicsMode`
* Build doesn't look for CUDA backend unless `--cuda` flag is passed

## hipBLAS 0.36.0 for ROCm 3.9.0

### Additions

* Device memory reallocates on demand

## hipBLAS 0.34.0 for ROCm 3.8.0

### Additions

* `--static` build flag allows the creation of a static library

## hipBLAS 0.32.0 for ROCm 3.7.0

### Additions

* `--rocblas-path` command line option to choose path to pre-built rocBLAS
* `sgetriBatched`
* `dgetriBatched`
* `cgetriBatched`
* `zgetriBatched`
* `TrsmEx`
* `TrsmBatchedEx`
* `TrsmStridedBatchedEx`
* `hipblasSetVectorAsync`
* `hipblasGetVectorAsync`
* `hipblasSetMatrixAsync`
* `hipblasGetMatrixAsync`
* Fortran support for `getrf`, `getrs`, `geqrf`, and all variants thereof

## hipBLAS 0.30.0 for ROCm 3.6.0

### Additions

* Added the following functions, which include batched and strided-batched support with the rocBLAS
  backend:
  * `stbsv`, `dtbsv`, `ctbsv`, `ztbsv`
  * `ssymm`, `dsymm`, `csymm`, `zsymm`
  * `cgeam`, `zgeam`
  * `chemm`, `zhemm`
  * `strtri`, `dtrtri`, `ctrtri`, `ztrtri
  * `sdgmm`, `ddgmm`, `cdgmm`, `zdgmm`
* `GemmBatchedEx` and `GemmStridedBatchedEx`
* Fortran support for BLAS functions

## hipBLAS 0.28.0 for ROCm 3.5.0

### Additions

* Added the following functions, which include batched and strided-batched support with the rocBLAS
  backend:
  * `sgbmv`, `dgbmv`, `cgbmv`, `zgbmv`
  * `chemv`, `zhemv`
  * `stbmv`, `dtbmv`, `ctbmv`, `ztbmv`
  * `strmv`, `trmv`, `ctrmv`, `ztrmv`
  * `chbmv`, `zhbmv`
  * `cher`, `zher`
  * `cher2`, `zher2`
  * `chpmv`, `zhpmv`
  * `chpr`, `zhpr`
  * `chpr2`, `zhpr2`
  * `ssbmv`, `dsbmv`
  * `sspmv`, `dspmv`
  * `ssymv`, `dsymv`, `csymv`, `zsymv`
  * `stpmv`, `dtpmv`, `ctpmv`, `ztpmv`
  * `cgeru`, `cgerc`, `zgeru`, `zgerc`
  * `sspr`, `dspr`, `cspr`, `zspr`
  * `sspr2`, `dspr2`
  * `csyr`, `zsyr`
  * `ssyr2`, `dsyr2`, `csyr2`, `zsyr2`
  * `stpsv`, `dtpsv`, `ctpsv`, `ztpsv`
  * `ctrsv`, `ztrsv`
  * `cherk`, `zherk`
  * `cherkx`, `zherkx`
  * `cher2k`, `zher2k`
  * `ssyrk`, `dsyrk`, `csyrk`, `zsyrk`
  * `ssyr2k`, `dsyr2k`, `csyr2k`, `zsyr2k`
  * `ssyrkx`, `dsyrkx`, `csyrkx`, `zsyrkx`
  * `ctrmm`, `ztrmm`
  * `ctrsm`, `ztrsm`
