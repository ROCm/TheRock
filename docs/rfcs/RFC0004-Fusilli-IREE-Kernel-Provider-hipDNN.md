---
author(s): Sambhav Jain, Aaron St. George, and Mahesh Ravishankar
created: 2025-10-17
modified: 2025-11-13
status: draft
discussion: https://github.com/ROCm/TheRock/discussions/1817
---

# Fusilli+IREE as a kernel provider and JIT engine for hipDNN

This RFC proposes adding IREE as a kernel provider to hipDNN to leverage JIT
compiled and codegenerated kernels in ML training and inference solutions.
This is made possible with the development of Fusilli - a C++ graph API and
JIT engine for IREE. We believe hand-authored kernel libraries are great for
highly tuned performance but they are difficult to 1) scale to newer models
or target architectures and 2) package and release effectively. This RFC is
founded on the overarching goal to complement our software stack with JIT
solutions while being competitive to hand-authored kernel libraries. Apart
from the usual benefits of having a compiler-backed JIT engine that gets
progressively better, a systemic benefit of this is it helps reduce build
times and binary sizes, making it easier to ship software effectively.

## Overview

[IREE](https://github.com/iree-org/iree/) is an open source ML compiler stack
built using MLIR that is intended to support the compilation and execution of
ML models. While IREE supports multiple target backends, over the past couple
of years a lot of effort has gone into improving the codegeneration for AMD
GPUs, specifically Instinct (MI-series) GPUs. Much of the IREE compiler stack
is geared towards optimizing execution of full-scale ML models. However, a key
objective of this work is to have efficient kernel code generation for MI300+
GPUs.

[Fusilli](https://github.com/iree-org/fusilli) is a C++ graph API that leverages
the kernel codegeneration capabilities of IREE and packages it to be usable as
a JIT engine for hipDNN. This allows use of IREE for specific portions of the
program, even for training use cases. The advantages of this approach are:

1. IREE has been built from the ground-up as a fusion compiler. The
   kinds of fusions that libraries like hipDNN are expected to provide
   are supported out-of-the box in IREE.
1. Fusilli allows compiling codegenerated kernels just-in-time (on-demand)
   without having to ship pre-built kernels with hipDNN - saving both build
   times and binary sizes.

## Workplan

From a code organization standpoint, there are three components to reason about:

1. IREE. This includes the compiler and runtime stack. It is a Linux Foundation
   project and lives [here](https://github.com/iree-org/iree).
1. Fusilli. This is a general purpose API and backend-neutral JIT engine for
   IREE that lives [here](https://github.com/iree-org/fusilli).
   It depends minimally on IREE compiler (CLI) and IREE runtime (C-API), and
   does NOT require a direct HIP dependency (abstracted by IREE's HAL design).
1. Fusilli-Plugin. The hipDNN engine plugin for Fusilli that specializes it for
   use within hipDNN - specifically for AMD GPUs. Currently it is being developed
   [here](https://github.com/nod-ai/shark-ai/tree/main/fusilli-plugin).

### Short term plan

The immediate goal is to build the hipDNN engine plugin (i.e., component 3
above) in TheRock. The plugin's dependencies require all three components to
be part of the build. IREE is a large dependency; the desire is to keep it an
optional, experimental, component not built by default. Therefore IREE's
TheRock build and all dependent projects (Fusilli, Fusilli-Plugin) will be
gated behind a CMake option `THEROCK_ENABLE_BUILD_IREE_LIBS`

For the various build scripts, this RFC proposes a new top level directory
`iree-libs`, gated in the top level `CMakeLists.txt`. Note: `iree-libs` name is
subject to change, we need a name, so `iree-libs` is a placeholder for the RFC.

```diff
 add_subdirectory(comm-libs)
 add_subdirectory(math-libs)
 add_subdirectory(ml-libs)
+
+if(THEROCK_ENABLE_BUILD_IREE_LIBS)
+  add_subdirectory(iree-libs)
+endif()
```

```
...
├── iree-libs
│   ├── CMakeLists.txt
│   ├── iree    (submodule)
│   └── fusilli (submodule)
├── math-libs
...
```

The following sections detail where each dependency will live outside of
TheRock, and what its build will look like inside of TheRock.

#### IREE

IREE will remain in its Linux Foundation-governed iree-org repo. TheRock
will fetch IREE as a submodule, and otherwise treat it as a fairly standard
`therock_cmake_subproject_declare`.

The full IREE project includes the IREE compiler and runtime, but initially
TheRock will only build the runtime. The compiler isn't (currently) necessary
as Fusilli uses the standalone `iree-compile` executable. IREE has many
submodules, including its own LLVM, but for the initial integration
`build_tools/fetch_sources.py` will only fetch what's necessary to build the
runtime:
[`flatcc`](https://github.com/iree-org/iree/tree/817512d4b8af1f4c2d2109d8e211aee3d69e6af8/third_party).
Longer term, the story will evolve.

`iree-libs/CMakeLists.txt`

```cmake
therock_cmake_subproject_declare(therock-iree-runtime
  EXCLUDE_FROM_ALL
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/iree-runtime"
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree"
  ...
```

The IREE runtime builds as a series of `.a` archives which are intended to be
linked into a final executable with LTO style optimization.

#### Fusilli

Fusilli will remain in its Linux Foundation-governed iree-org repo. TheRock
will fetch Fusilli as a submodule, and again otherwise treat it as a fairly
standard `therock_cmake_subproject_declare`. Fusilli has no submodules.

`iree-libs/CMakeLists.txt`

```cmake
therock_cmake_subproject_declare(therock-fusilli
  EXCLUDE_FROM_ALL
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fusilli"
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusilli"
  BUILD_DEPS
   therock-iree-runtime
  ...
```

Fusilli builds as a series of `.h` files.

A small note on C++ standards: Fusilli and the hipDNN engine plugin for Fusilli
are built on the C++20 standard. We believe this should not pose any issues from an
integration standpoint but happy to revisit this further if the need arises.

#### Fusilli-Plugin

Fusilli-Plugin is currently in
[`amd-shark-ai`](https://github.com/nod-ai/shark-ai/tree/b33f16e77ef00b4c9378fcd5edd3123d72fdcb68/fusilli-plugin),
and move into a `plugins/` subdirectory of
[`fusilli`](https://github.com/iree-org/fusilli). It will remain a standalone
project emulating the end state (as if it was "in TheRock") - built _with_
the Fusilli API, rather than being a part of the Fusilli API itself - but just
in the Fusilli repo to stage the roll-out to be less disruptive. This allows
for it to be easily movable alongside other hipdnn plugins at a later point if
desired.

Fusilli-Plugin will build as a `therock_cmake_subproject_declare`, reaching into
the `plugins/` directory of the `fusilli` submodule.

```cmake
therock_cmake_subproject_declare(therock-fusilli-plugin
  EXCLUDE_FROM_ALL
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fusilli-plugin"
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusilli/plugins/hipdnn-plugin"
  BUILD_DEPS
   hipDNN
   therock-iree-runtime
   therock-fusilli
  ...
```

The expected build artifact from the plugin integration is a self-contained
`libfusilliplugin.so`, built with Fusilli headers, and linking IREE runtime
libraries with LTO style optimizations.

### Long term plan

As the Fusilli integration evolves, so will TheRock's IREE integration. It will likely follow the steps below:

1. (current state) Fusilli uses a "side loaded" `iree-compile` binary. If users build TheRock with `THEROCK_ENABLE_BUILD_IREE_LIBS=ON` (off by default) they will also need to pip install [`iree-compiler`](https://pypi.org/project/iree-compiler/) on their system.
1. (mid term) Fusilli will move to using IREE compiler's C-API + `libIREECompiler.so` built in TheRock. IREE uses a *very* recent LLVM (daily integration). In this phase, `build_tools/fetch_sources.py` will be updated to fetch IREE's captive LLVM, adding yet another LLVM build to TheRock. The main build will not be affected, but builds with `THEROCK_ENABLE_BUILD_IREE_LIBS=ON` will be slower at this point in the integration.
1. (long term) Unify some set of LLVMs to eliminate IREE's captive LLVM - this will be a _large_ effort. After unification, building Fusilli integration should be cheap.

## Revision History

- 2025-10-17: Sambhav Jain: Initial version

- 2025-11-13: Aaron St George: Added detail to Short term plan
