---
author: Stella Laurenzo (stellaraccident)
created: 2025-09-08
modified: 2025-09-08
status: draft
discussion: https://github.com/ROCm/TheRock/discussions/1425
---

# Build Tree Normalization Post Super-Repo Migrations

This RFC recommends a re-organization of project directories to meet the following goals:

1. Provide consistency between TheRock and libraries/systems submodule repositories.
1. Produce a structure whereby we can move TheRock CMake build files and artifact descriptors into the submodule repositories (vs having them spanning both).
1. Have a clear structure of the composite source tree that makes semantic sense and makes it clear how to `mkdir` related projects and components.
1. Post super-repo migration, return to a state where the composite source tree and the build tree line up.
1. Provide directory tree layering that is forward compatible with future addition of dedicated language binding layers (as opposed to the current state where language bindings are often built for direct inclusion in SDK packages as part of the hosting project).
1. Perform non-intrusive casing and punctuation normalization as possible.

Non-goals:

1. Refactoring existing project boundaries (this can be a related effort but is not strictly required).
1. Changing the organization of the compiler sub-tree or that of deprecated/misplaced base libraries.

## History and Current Status

Now that the [rocm-systems](https://github.com/ROCm/rocm-systems) and [rocm-libraries](https://github.com/ROCm/rocm-libraries) super-repos are populated, it is possible to begin the process of re-organizing the source trees to better reflect the taxonomy and adjacencies in the software.

When TheRock was started, an initial directory tree was chosen to organize the build scripts and submodules:

```
├── base
│   ├── aux-overlay
│   ├── half
│   └── rocm-cmake
├── comm-libs
│   ├── rccl
│   └── rccl-tests
├── compiler
│   ├── amd-llvm
│   └── hipify
├── core
│   ├── amdgpu-windows-interop
├── math-libs
│   ├── BLAS
│   └── support
├── ml-libs
│   └── composable_kernel
└── profiler
```

This was always seen as a first attempt to provide an overall organization to ROCm that would be revisited once various repository migrations were complete. Note that sub-project load order must be a DAG relative to project dependencies.

Current load order:

```
# Add subdirectories in dependency DAG order (which happens to be semi-alpha:
# don't be fooled).
add_subdirectory(third-party)
add_subdirectory(base)
add_subdirectory(compiler)
add_subdirectory(core)
# Note that rocprofiler-register is in base and is what higher level clients
# depend on. The profiler itself is independent.
add_subdirectory(profiler)
add_subdirectory(comm-libs)
add_subdirectory(math-libs)
add_subdirectory(ml-libs)
```

Note that prior to the migration to the libraries and systems super-repos, each individual submodule was placed under its semantic directory location above. Post migration, the `rocm-libraries/` and `rocm-systems/` directories in the root carry most of the source code, and the build files in the taxonomized directories point there.

In this prior state, it was easier to see which of the above directories contained which projects, because the sub-projects would all have a submodule/directory that was plainly visible. Now they are just artifacts in the corresponding `CMakeLists.txt` files. To keep from paging through, here are the sub-projects of each TLD:

- `base`:
  - Artifacts:
    - `base`
  - Sub-projects:
    - `therock-aux-overlay` (local directory)
    - `rocm-cmake` (submodule)
    - `rocm-core` (link to `../rocm-systems/projects/rocm-core`)
    - `rocm_smi_lib` (link to `../rocm-systems/projects/rocm-smi-lib`)
    - `rocprofiler-register` (link to `../rocm-systems/projects/rocprofiler-register`)
    - `rocm-half` (submodule)
- `compiler/`:
  - Artifacts:
    - `amd-llvm`
    - `hipify`
  - Sub-projects:
    - `amd-llvm`
    - `amd-comgr`
    - `hipcc`
    - `hipify`
- `core/`:
  - Artifacts:
    - `hip`
    - `hipinfo`
    - `core-runtime`
  - Sub-projects:
    - `ROCR-Runtime` (link to `../rocm-systems/projects/rocr-runtime`)
    - `rocminfo` (link to `../rocm-systems/projects/rocminfo`)
    - `hip-clr` (link to `../rocm-systems/projects/clr`)
    - `hipInfo` (win32 only, local directory that trampolines to `../rocm-systems/projects/hip-tests`)
- `profiler/`:
  - Artifacts:
    - `rocprofiler`
  - Sub-projects:
    - `rocprof-trace-decoder` (local directory which incorporates binary only library)
    - `aqlprofile` (link to `../rocm-systems/projects/aqlprofile`)
    - `rocprofiler-sdk` (link to `../rocm-systems/projects/rocprofiler-sdk`)
    - `roctracer` (link to \`../rocm-systems/projects/roctracer)
- `comm-libs/`:
  - Artifacts:
    - `rccl`
  - Sub-projects:
    - `rccl` (submodule)
    - `rccl-tests` (submodule)
- `math-libs/`:
  - Artifacts:
    - `fft`
    - `prim`
    - `rand`
  - Sub-projects:
    - `rocRAND` (link to `../rocm-libraries/projects/rocrand`)
    - `hipRAND` (link to `../rocm-libraries/projects/hiprand`)
    - `rocPRIM` (link to `../rocm-libraries/projects/rocprim`)
    - `hipCUB` (link to `../rocm-libraries/projects/hipcub`)
    - `rocPRIM` (link to `../rocm-libraries/projects/rocprim`)
    - `rocThrust` (link to `../rocm-libraries/projects/rocthrust`)
    - `rocFFT` (link to `../rocm-libraries/projects/rocfft`)
    - `hipFFT` (link to `../rocm-libraries/projects/hipfft`)
- `math-libs/support`:
  - Artifacts:
    - `support` (AI: this should not just be called "support" as a barename)
  - Sub-projects:
    - `mxDataGenerator` (link to `../rocm-libraries/shared/mxdatagenerator`)
- `math-libs/BLAS`:
  - Artifacts:
    - `blas`
  - Sub-projects:
    - `hipBLAS-common` (link to `../../rocm-libraries/projects/hipblas-common`)
    - `rocRoller` (link to `../../rocm-libraries/shared/rocroller`)
    - `hipBLASLt` (link to `../../rocm-libraries/projects/hipblaslt`)
    - `rocBLAS` (link to `../../rocm-libraries/projects/rocblas`)
    - `rocSPARSE` (link to `../../rocm-libraries/projects/rocsparse`)
    - `hipSPARSE` (link to `../../rocm-libraries/projects/hipsparse`)
    - `rocSOLVER` (submodule, migration in progress)
    - `hipSOLVER` (submodule, migration in progress)
    - `hipBLAS` (link to `../../rocm-libraries/projects/hipblas`)
- `ml-libs/`
  - Artifacts:
    - `composable-kernel`
    - `miopen`
    - `hipdnn` (soon)
  - Sub-projects:
    - `composable_kernel`
    - `miopen`
    - `hipdnn` (soon)

## Dependency and Layering Considerations

The constraint that sub-project declarations must be strictly ordered after declarations of all of their dependencies puts hard limits on the layout of the directory tree, artifacts and package structures. While all sub-projects have dependencies, some of them are intricate/counter-intuitive and need an explicit explainer. This is further complicated when such projects are partway through a long term re-organization.

### GEMM and LA Library Dependencies

The improperly named `math-libs/BLAS` directory contains project build rules for the GEMM and LA (linear algebra) libraries, in both dense and sparse variants. In the present configuration, these are all lumped into one `blas` artifacts. This was considered a POC/starting point during initial bringup and needs to be refactored. In addition, these libraries themselves are undergoing significant evolution, sometimes without a widely known end state defined. As such, a re-organization needs to consider both of these aspects. Here are the present sub-project dependencies of this tree with sibling projects in the GEMM/LA and math-libs stack:

- `hipBLAS-common`: None
- `rocRoller`: None
- `hipBLASLt`: `hipBLAS-common`, `rocRoller`, `rocisa` (bundled), `tensilelite` (bundled), `origami` (bundled)
- `rocBLAS`: `hipBLAS-common`, `hipBLASLt`, `tensile`
- `rocSPARSE`: `rocBLAS`, `rocPRIM`
- `hipSPARSE`: `rocSPARSE`
- `hipSPARSELt`: `tensilelite` (via hipBLASLt)
- `rocSOLVER`: `rocPRIM`, `rocBLAS`
- `hipSOLVER`: `rocBLAS`, `rocSOLVER`, `rocSPARSE`
- `hipBLAS`: `hipBLAS-common`, `rocBLAS`, `rocSOLVER`

Much of this categorization is historical. Further, the long sub-project dependency chains create significant false dependency edges in the build graph, which causes tens of minutes of additional build latency. On the surface with present names, coming up with a more sensible categorization is a bit impenetrable. However, if we apply a couple of transformations, it becomes a bit more manageable:

- Define `codegen` category and move `rocRoller`, `tensile`, `tensilelite`, `rocisa` and (potentially) `origami` into it.
- `hipBLAS-common`: Rename to `gemmla-common`.
- Consider that both `hipBLASLt` and `hipSPARSELt` contain both a backend provider and their public API in the same project, whereas others separate these concerns.
- `rocBLAS`, `hipBLASLt`: Implement dense GEMMs or provide APIs for such (think: `gemm-dense`).
- `rocSPARSE`, `hipSPARSE`, `hipSPARSELt`: Implement sparse GEMMs or provide APIs for such (think: `gemm-sparse`).
- `rocSOLVER`: Implements LA algorithms (think: `la`).
- `hipBLAS`, `hipSOLVER`: Algorithm selector APIs (think: `gemm-api`, `la-api` respectively).

We may not actually execute such renames any time soon, but thinking of them in these terms may help when considering layering, artifacts/packaging, and code organization.

We will also be promoting everything in this stack to a unified build vs a cascade of individual sub-projects in the near future, and any organization should enable us to do that cleanly with proper nesting.

### Profiler Layering

The profiler stack is mostly self contained with the exception of rocprofiler-register, which is a base rendezvous library upon which many other components of ROCm depend. As such, it must be defined before most other projects, but the profiler itself does not generally have inbound deps.

### Composable Kernel, MIOpen, and hipdnn

These three projects are potentially the most mis-categorized in the build system. Logically, composable kernel is a peer of other C++ compute-kernel authoring libraries, however historically it's inbound deps in ROCm have been via a hidden pin from MIOpen and its lack of API/implementation stability has forced these two into lock-step. These defects are in the process of being corrected, but when it was added to the build system, it was purely to support MIOpen and they were kept in lock-step.

Further, MIOpen has historically been both a kernel aggregator and a public API. It is being reclassified as a kernel provider for certain key kernels useful to ML that we need to support in the various frameworks (convolutions, \*norm, etc). hipdnn is being introduced as a public API layer that can have kernel provider backends. MIOpen will become one of these backends.

This document will propse structuring congruent with this future state.

### Others

There are other projects which have a non-trivial evolution roadmap in front of them but these changes are not yet in TheRock build system. These will be considered in a later phase of project organization:

- `rccl`: Potential aggregation with higher level distribution libraries and other implementation changes
- Decoders (`rocDECODE`, `rocJPEG`): Will be added to the `rocm-systems` super-repo and potentially re-organized into one project in concert with dependency improvements with respect to their user-mode drivers. Their language bindings will be separated to the language binding tree.
- `*smi`, `rdc`, etc: Work is ongoing to separate concerns and land these properly into `rocm-systems`
- Python bindings: Many projects contain integrated builds of various Python bindings that are distributed for public consumption. These bindings are being disaggregated from their backing C++ API projects and will be built/deployed independently (with the exception of Python projects that are build-only dependencies that are not distributed).

## Abstract Proposal

Currently, TheRock hosts all sub-project and artifact definition files, while source code and project-specific `CMakeLists.txt` files are stored in the submodules. While useful for bootstrapping, this is not a great state of affairs for managing project evolution since it creates a coupling between the super-project (TheRock) for various build system details and component projects. This can make it difficult to perform lock-step changes to the project build system.

Further, the systems and libraries super-repos have a mostly flat directory layout (under `projects/` and `shared/`) that was expedient for bootstrapping but should be reconsidered.

We propose letting TheRock continue to be the build driver, while relocating the sub-project and artifact definitions to the component repositories. We will specifically focus on the rocm-systems and rocm-libraries super-repos and ignore the remaining stand-alone repositories. In order to provide sound ordering/layering, this will force a hierarchical directory structure in the libraries/systems super-repos that corresponds to the layering of the actual software -- whereas today, all of the sources are in a flat tree under `projects/` or `shared/`.

In order to make such a large-scale change tractable, we will create this new directory tree in-place, having it *only* contain TheRock cmake and artifact files, referring to the flat `projects/` and `shared/` tree with relative paths as needed. Then in future followups, the project teams can move the contents of their project directories into the taxonomized tree using single atomic commits when ready.

## Concrete Proposal

### TheRock CMake extensions

Currently, internal to TheRock, the taxonomized trees are added with normal CMake `add_subdirectory()` calls and the contained CMakeLists.txt is expected to contain various incantations to define the sub-project setup and dependencies:

- `therock_cmake_subproject_declare()`
- `therock_cmake_subproject_activate()`
- `therock_provide_artifact()`
- etc

Consider an ideal end state for a hypothetical taxonomized tree:

```
airplane/
  CMakeLists.txt (for add_subdirectory)
  engine/
    CMakeLists.txt (for add_subdirectory)
    common/
      CMakeLists.txt
    housing/
      CMakeLists.txt
    thruster/
      CMakeLists.txt
  cockpit/
    CMakeLists.txt (for add_subdirectory)
    instruments/
      CMakeLists.txt
    seating/
      CMakeLists.txt
```

In this configuration, you want the leaf CMakeLists.txt files to be the actual sub-project top-level build files (i.e. have a `project()` declaration and able to be built independently if supported). This clashes if the super-project is sharing the same tree because both will want to use `CMakeLists.txt`.

While somewhat cute, for third-party bundled libraries we solve this by including the super-project setup in a `if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)` and `return()` top-level switch, with the code that follows dedicated to the sub-project. As a dark corner of TheRock that few look at, this is tolerable, but it doesn't make for a great experience at scale in user-projects.

Instead, we will do something like this (precise details TBD, considering any gotchas that come up during implementation):

```
airplane/
  therock.cmake (for therock_add_subdirectory())
  engine/
    therock.cmake (for therock_add_subdirectory())
    housing/
      CMakeLists.txt
      therock.cmake (for defining global options)
      therock_subprojects.cmake
      therock_pre_hook_housing.cmake
      therock_artifact_housing.toml
```

This will necessitate some upgrades to how the existing macros work but shouldn't be too bad. We will need to add a new `therock_add_subdirectory()` to add super-project sub-directories that look for the special `therock_subproject.cmake`, which performs actual subproject setup.

The idea is that `therock.cmake` does the following:

- Issues calls to `therock_add_subdirectory()` as needed to populate the directory tree.
- Declares super-project options relevant to the subtree.
- Calls other `therock_*` configuration time functions (i.e. declare components, query gfx arches, etc).

Then, once the directory tree is traversed, and sub-directories that contain a `therock_subprojects.cmake` will be invoked in new scopes as the super-project `CMakeLists.txt` are today.

This will reduce the add_subdirectory block in TheRock's top-level `CMakeLists.txt` to something like:

```
therock_add_subdirectory(rocm-systems)
therock_add_subdirectory(rocm-libraries)
```

### `rocm-systems` directory layout

Bracketed (`[]`) items are where project source directories can be moved in the future.

```
base/
  [rocm-core] -> projects/rocm-core
  [rocm-smi-lib] -> projects/rocm-smi-lib
  [rocprofiler-register] -> projects/rocprofiler-register.cmake
  therock.cmake
  therock_subprojects.cmake
  therock_artifact_base.toml
runtime/
  [rocr-runtime] -> projects/rocr-runtime
  [rocminfo] -> projects/rocminfo
  [clr] -> projects/clr
  [hip] -> projects/hip
  [hip-tests] -> projects/hip-tests
  [hipother] -> projects/hipother
  therock.cmake
  therock_subprojects.cmake
  therock_artifact_rocr.toml
  therock_artifact_hip.toml
profiler/
  trace-decoder/
    CMakeLists.txt
  [aqlprofile/] -> projects/aqlprofile
  [sdk/] -> projects/rocprofiler-sdk
  [tracer/] -> projects/roctracer
  therock_artifact_rocprofiler-sdk.toml
  therock.cmake
  therock_subprojects.cmake
```

### `rocm-libraries` directory layout

Bracketed (`[]`) items are where project source directories can be moved in the future.

```
shared/
  mxdatagenerator/
  rocroller/
  tensile/
  therock.cmake
  therock_subprojects.cmake

gemmla/
  common/
    [hipblas-common] -> project/hipblas-common
    therock.cmake
    therock_subprojects.cmake
    therock_artifact_gemmla-common.toml
  gemm-dense/
    [rocblas] -> projects/rocblas
    [hipblaslt] -> projects/hipblaslt
    therock.cmake
    therock_subprojects.cmake
    therock_artifact_gemm-dense.toml
  gemm-sparse/
    [rocsparse] -> projects/rocsparse
    [hipsparselt] -> projects/hipsparselt
    [hipsparse] -> projects/hipsparse
    therock.cmake
    therock_subprojects.cmake
    therock_artifact_gemm-sparse.toml
  la/
    [rocsolver] -> projects/rocsolver
    [hipsolver] -> projects/hipsolver
    therock.cmake
    therock_subprojects.cmake
    therock_artifact_la.toml
  gemm-api/
    [hipblas] -> projects/hipblas
    therock.cmake
    therock_subprojects.cmake
    therock_artifact_gemm-api.toml

kernel-libs/
  [rocrand] -> projects/rocrand
  [hiprand] -> projects/hiprand
  [rocfft] -> projects/rocfft
  [hipfft] -> projects/hipfft
  therock.cmake
  therock_subprojects.cmake
  therock_artifact_fft.toml
  therock_artifact_rand.toml

compute-libs/
  [rocprim] -> projects/rocprim
  [hipcub] -> projects/hipcub
  [rocthrust] -> projects/rocthrust
  [composable-kernel] -> projects/composable-kernel (plan TBD, shown for illustration)
  therock.cmake
  therock_subprojects.cmake
  therock_artifact_prim.toml
  therock_artifact_composable-kernel.toml

dnn-api/
  [hipdnn] -> projects/hipdnn
  therock.cmake
  therock_subprojects.cmake
  therock_artifact_dnn.toml

dnn-providers/
  miopen-provider/
    [miopen/] -> projects/miopen
    therock.cmake
    therock_subprojects.cmake
    therock_artifact_miopen.toml
  hipblaslt-provider/
    ...
  foo-provider/
    ...
```

#### Implied Functional Changes

##### Relayering of the GEMM/LA libraries to better reflect current thinking

Implicit in this layering is an expansion with respect to the artifacts in the new `gemmla` grouping. Whereas we did have one `blas` artifact, we now will have:

- `gemmla-common`
- `gemm-dense` (dep: `gemmla-common`)
- `gemm-sparse` (dep: `gemmla-dense`, `gemmla-common`)
- `la` (dep: `gemmla-dense`, `gemmla-common`)
- `gemm-api` (dep: `gemmla-common`, `gemm-dense`, `la`)

These artifacts will be mirrored into native packages by the packaging scripts. Note that in the future, API layers from hipblaslt and hipsparselt may be moved to the `gemm-api` artifact if we desire a more normalized library layering.

##### Categorizing kernel and device libraries

Futher, we create two additional categories of libraries:

- `kernel-libs`: Non gemmla libraries of specialty kernels.
- `device-libs`: (mostly header only) APIs for doing device library development.

##### Setting up the namespace for hipdnn

As mentioned above, miopen is being expanded into a suite. This layout reflects where that is going.

### Implementation Plan

1. Implement needed TheRock CMake sub-project extensions and switch current directory structure to use `therock*` files instead of `CMakeLists.txt` (atomic PR on TheRock).
1. Land changes into `rocm-libraries` and `rocm-systems` to materialize the current sub-project setup from TheRock into the directory structures above (sequence of NFC changes that won't get used until TheRock is redirected/passes).
1. Change top-level `add_subdirectory()` calls into current TheRock build tree to directly refer to the `rocm-libraries` and `rocm-systems` submodules, deleting the existing files/directories (atomic PR to TheRock).
1. Rename the `rocm-libraries` submodule path to `libraries/` and `rocm-systems` to `systems/` (NFC ergonomic upgrade that can be done at any time).
1. Work with project teams to move their source code from the flat `projects/` tree to new taxonomized locations (sequence of atomic, NFC PRs to super-repos).

## Revision History

- 2025-09-08: stellaraccident: Initial version
