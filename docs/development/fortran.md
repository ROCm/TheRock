# Fortran Support

TheRock builds Fortran support as a layer on top of the minimal AMD LLVM
compiler. This keeps the base compiler usable for C, C++, HIP, and AMDGPU code
generation without also forcing Flang, MLIR, OpenMP offload, or Flang runtime
payload into the bottom of the dependency graph.

## LLVM Layers

The LLVM-related artifacts are intentionally split:

- `amd-llvm-base`: bootstrap compiler layer. Builds LLVM, Clang, LLD,
  compiler-rt, AMDGPU/X86 backends, and AMD device libraries. This is the base
  compiler used by lower runtime layers.
- `amd-llvm`: compiler infrastructure layer. Adds ROCm compiler consumers such
  as COMGR and hipcc on top of `amd-llvm-base`.
- `amd-llvm-flang`: Linux-only Flang frontend layer. It builds Flang and the
  MLIR implementation pieces needed by Flang, but MLIR is not a public
  installed artifact.
- `amd-llvm-offload`: Linux-only OpenMP/offload runtime layer. It builds
  `libomp`, `libomptarget`, device OpenMP runtime pieces, and, when Flang is
  enabled, `flang-rt`.

The full Fortran toolchain requires both `amd-llvm-flang` and
`amd-llvm-offload`: Flang supplies the compiler driver and frontend, while
`amd-llvm-offload` supplies the runtime libraries needed to link Fortran code.

## Build Flags

- `THEROCK_ENABLE_FLANG`: enables the Flang compiler layer.
- `THEROCK_ENABLE_AMD_LLVM_OFFLOAD`: enables the OpenMP/offload runtime layer.
- `THEROCK_FLAG_BUILD_FORTRAN_LIBS`: controls whether ROCm subprojects build
  their Fortran clients, wrappers, examples, or bindings.
- `ROCM_BUILD_FORTRAN_LIBS`: propagated into subprojects from
  `THEROCK_FLAG_BUILD_FORTRAN_LIBS`.

Subprojects declare their relationship to Fortran through
`therock_cmake_subproject_declare`:

- `FORTRAN_OPTIONAL`: use the built Fortran/offload toolchain when
  `THEROCK_FLAG_BUILD_FORTRAN_LIBS=ON`; otherwise configure the subproject with
  Fortran disabled.
- `FORTRAN_REQUIRED`: fail configure if Fortran libraries are disabled or the
  split Flang/offload toolchain is unavailable.

When Fortran is requested, TheRock wires the subproject to the
`amd-llvm-offload` compiler toolchain. That toolchain exposes Clang, Flang, the
OpenMP runtime, offload runtime, and Flang runtime from one staged tree.

## Developer Builds

For local Linux builds, keep LLVM and Flang concurrency inside the machine's
memory envelope. LLVM link jobs are especially large, and Flang compile jobs are
also memory-heavy.

Example minimal HIP/runtime build without Fortran:

```bash
cmake -B build -S sources/TheRock -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_FLANG=OFF \
  -DTHEROCK_ENABLE_AMD_LLVM_OFFLOAD=OFF \
  -DTHEROCK_FLAG_BUILD_FORTRAN_LIBS=OFF \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100 \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DFLANG_PARALLEL_COMPILE_JOBS=32 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

Example Fortran/offload toolchain build:

```bash
cmake -B build -S sources/TheRock -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_COMPILER=ON \
  -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_FLANG=ON \
  -DTHEROCK_ENABLE_AMD_LLVM_OFFLOAD=ON \
  -DTHEROCK_FLAG_BUILD_FORTRAN_LIBS=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100 \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DFLANG_PARALLEL_COMPILE_JOBS=32 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

Use `+expunge` when changing Fortran or offload state for a downstream project,
because CMake can cache compiler and package discovery results:

```bash
ninja -C build rocprofiler-systems+expunge
ninja -C build rocprofiler-systems
```

## Verification

Before debugging deep integration projects, verify the staged toolchain
directly:

- `build/dist/rocm/lib/llvm/bin/flang` exists when Flang is enabled.
- `build/dist/rocm/lib/llvm/lib` contains Flang runtime libraries such as
  `libFortran*`, `libflang*`, and `libflang_rt*`.
- `build/dist/rocm/lib/llvm/include/flang` contains `omp_lib.mod` and
  `omp_lib_kinds.mod` when OpenMP Fortran modules are expected.
- `build/dist/rocm/lib/llvm/lib` contains `libomp` and `libomptarget`.
- `build/dist/rocm/lib/clang/<version>/lib/amdgcn-amd-amdhsa` contains the
  device OpenMP runtime payload.

Use small standalone compile/link tests for C OpenMP offload, simple Fortran,
Fortran using `omp_lib`, and HIP compilation before enabling larger ROCm
projects such as rocprofiler-systems.

## Known Follow-Ups

- LLVM offload currently takes a source dependency on ROCr through
  `LIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH`. TheRock should eventually make this
  consume the staged ROCr/core-runtime package instead.
- rocRAND and hipRAND currently generate package configs that require
  root-level `${prefix}/rocrand/src/fortran` and
  `${prefix}/hiprand/src/fortran` source trees. The rand artifact packages
  those paths for compatibility, but the RAND projects should fix that public
  install contract upstream.
- The OpenMP generated-header patch keeps standalone runtimes builds from using
  the installed Clang resource include directory as the intermediate generated
  header directory. That patch should be evaluated for upstreaming or replaced
  with an upstream-supported split runtimes build mode.
