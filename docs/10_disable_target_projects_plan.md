# DISABLE_TARGET_PROJECTS implementation plan

## Motivation

`EXCLUDE_TARGET_PROJECTS` (in `cmake/therock_amdgpu_targets.cmake`) only strips a
gfx target from a project's `GPU_TARGETS`; the project still builds (host-only or
via `DEFAULT_GPU_TARGETS` fallback). It cannot express "this project is
meaningless for this target and must not build at all".

The `amdgcnspirv` portable target (family `gpu-generic`) is build-only: it has no
per-arch device code and is finalized to ISA at load time. Device libraries
(rocBLAS, rocFFT, MIOpen, …) make no sense for it. The current PR works around
this with an ad-hoc `therock_project_has_no_targets` gate that checks the EXCLUDE
list. This promotes that workaround to a first-class primitive.

## Two primitives, clear semantics

| Keyword                   | Effect                                              | Use when                                                                |
| ------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------- |
| `EXCLUDE_TARGET_PROJECTS` | strip arch from `GPU_TARGETS`, project still builds | project's kernels don't support the arch, but host/API parts still ship |
| `DISABLE_TARGET_PROJECTS` | project is not declared/built for the arch          | project is meaningless for the arch by design                           |

## Mechanism

- `therock_add_amdgpu_target(... DISABLE_TARGET_PROJECTS <names>)` records each
  name into global property `THEROCK_AMDGPU_PROJECT_TARGET_DISABLES_${name}`
  (list of gfx targets that disable it), mirroring the EXCLUDE storage.
- New public helper `therock_project_disabled(out_var project_name)`: TRUE when a
  per-arch target list is requested and every requested target disables the
  project (nothing remains). FALSE for ordinary gfx builds and generic stages.
- Each subproject `CMakeLists.txt` gates its `if(THEROCK_ENABLE_X ...)` block with
  `therock_project_disabled(_skip_X <declared-name>)` + `AND NOT _skip_X`.
- `therock_project_has_no_targets` is removed (introduced in this PR chain, no
  external users); all call sites move to `therock_project_disabled`.

## Partition of the amdgcnspirv project list (54)

DISABLE (device libraries + their tests/providers/samples, 35): MIOpen,
composable_kernel, hipBLAS, hipBLAS-common, hipBLASLt, hipCUB, hipDNN,
hipDNN_samples, hipFFT, hipRAND, hipSOLVER, hipSPARSE, hipSPARSELt, hipTensor,
hipblasltprovider, hipdnn_integration_tests, hipfile, hipkernelprovider,
hipthreads, miopenprovider, mxDataGenerator, rccl, rccl-tests, rocALUTION,
rocBLAS, rocFFT, rocPRIM, rocPRIM_tests, rocRAND, rocRoller, rocSOLVER,
rocSPARSE, rocThrust, rocWMMA, rocshmem.

EXCLUDE (core runtime/tools that still build host-only, 19): ROCR-Runtime,
aqlprofile, hip-clr, hip-tests, hipInfo, mirage, ocl-clr, rdc, rocdecode,
rocjitsu, rocjpeg, rocm-kpack, rocprofiler-compute, rocprofiler-sdk,
rocprofiler-systems, rocprofiler-systems-examples, rocr-debug-agent-tests,
rocrtst, roctracer.

## Edits

1. `cmake/therock_amdgpu_targets.cmake`: add `DISABLE_TARGET_PROJECTS` parsing +
   storage; split the amdgcnspirv list into EXCLUDE + DISABLE.
1. `cmake/therock_subproject.cmake`: add `therock_project_disabled`; remove
   `therock_project_has_no_targets`.
1. `math-libs`, `ml-libs`, `comm-libs`, `storage-libs`, `examples`
   `CMakeLists.txt`: switch gate calls to `therock_project_disabled`.
1. Tests: extend cmake target tests for the new keyword/helper.

## Status

Implemented. Verified:

- `python3 -m unittest build_tools.tests.cmake_amdgpu_targets_test` — 11 pass.
- Standalone `cmake -P` harness exercising `therock_add_amdgpu_target` +
  `therock_project_disabled` — 6 semantic cases pass (device lib disabled under
  spirv; core-runtime not disabled; not disabled under gfx942; not disabled in a
  mixed gfx942+spirv build; not disabled in no-targets generic stage; unknown
  project not disabled).
- rocWMMA remains in gfx900/gfx90c/gfx101X/gfx103X EXCLUDE (inner sample/test
  filter for partial-support shards unchanged); it is DISABLE'd only for
  amdgcnspirv, where its whole block is skipped.
- libhipcxx (header-only) is neither excluded nor disabled — builds under spirv.
