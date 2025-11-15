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

[Fusilli](https://github.com/nod-ai/shark-ai/tree/main/sharkfuser) is a C++
graph API that leverages the kernel codegeneration capabilities of IREE and
packages it to be useable as a JIT engine for hipDNN. This allows use of IREE
for specific portions of the program, even for training use cases. The
advantages of this approach are:

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
   IREE that currently lives [here](https://github.com/nod-ai/shark-ai/tree/main/sharkfuser).
   It depends minimally on IREE compiler (CLI) and IREE runtime (C-API), and
   does NOT require a direct HIP dependency (abstracted by IREE's HAL design).
1. Fusilli-Plugin. The hipDNN engine plugin for Fusilli. This specializes Fusilli for use within
   hipDNN specifically for AMD GPUs. Currently it is being developed
   [here](https://github.com/nod-ai/shark-ai/tree/main/fusilli-plugin), and has
   begun a migration to
   [rocm-libraries](https://github.com/ROCm/rocm-libraries/tree/develop/projects/fusilli-plugin).
   In addition to Fusilli's dependencies, the plugin also depends on HIP, hipDNN
   frontend/SDK and hipDNN's dependencies transitively.

### Short term plan

The immediate goal is to build the hipDNN engine plugin (i.e., component 3
above) in `TheRock`. The plugin's dependencies require all three components to be part of
the build. IREE is a large dependency; the desire is to keep it an optional component
not built by default. Therefore IREE's `TheRock` build and all dependent projects
(Fusilli, Fusilli-Plugin) will be gated behind a CMake option
`THEROCK_ENABLE_BUILD_IREE_LIBS`

For the various build scripts, this RFC proposes a new top level directory
`iree-libs`, gated in the top level `CMakeLists.txt`. Note: `iree-libs` name is
subject to change, we need a name, so `iree-libs` is a placeholder for the RFC.

```diff
 add_subdirectory(comm-libs)
 add_subdirectory(math-libs)
 add_subdirectory(ml-libs)
+if(THEROCK_ENABLE_BUILD_IREE_LIBS)
+  add_subdirectory(iree-libs)
+endif()
```

```
...
├── iree-libs
│   ├── CMakeLists.txt
│   ├── fusilli
│   │   └── CMakeLists.txt
│   ├── fusilli-plugin
│   │   └── CMakeLists.txt
│   └── iree
│       └── CMakeLists.txt
├── math-libs
...
```

The following sections detail where each dependency will live outside of
`TheRock`, and what its build will look like inside of `TheRock`.

#### `iree`

IREE will remain in its Linux Foundation-governed `iree-org` repo. `TheRock`
will fetch IREE as a git repository through `therock_subproject_fetch` but
otherwise treat it as a fairly standard `therock_cmake_subproject_declare`.

`iree-libs/iree/CMakeLists.txt`

```cmake
  therock_subproject_fetch(therock-iree-sources
    CMAKE_PROJECT
    GIT_REPOSITORY https://github.com/iree-org/iree.git
    GIT_TAG v1.2.3
    GIT_SUBMODULES third_party/flatcc/repo # IREE runtime only requires flatcc
  )
  therock_cmake_subproject_declare(therock-iree-runtime
    ...
```

IREE will only fetch submodules necessary to build the runtime. That's
currently sufficient for Fusilli as it uses the `iree-compile` binary (typically
made available through a pip-install) for compilation. Longer term Fusilli may
move towards using IREE compiler's C-API + `libIREECompiler.so` built in
`TheRock` or it could pick up an `iree-compile` also built in `TheRock` - this
is also addressed in the "Long term requirements" section below.

Question: How can the `iree-compile` dependency be expressed in `TheRock`? Can
the `iree-compile` dependency be pulled in as part of `TheRock` build, or the
`iree-compiler` pip dependency be added to the final `TheRock` pip package?

The IREE runtime builds as a series of `.a` archives which are intended to be
linked into a final executable with LTO style optimization.

#### `fusilli`

Fusilli is moving to `iree-org` as a standalone repo under Linux Foundation governance.

As Fusilli is a header only library with no submodules, it can be vendored similarly to
other third party deps in `TheRock`.

`iree-libs/fusilli/CMakeLists.txt`

```cmake
therock_subproject_fetch(therock-fusilli-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/iree-org/fusilli/archive/refs/tags/v0.0.1.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/fusilli-0.0.1.tar.gz
  URL_HASH SHA256=9102253214dea6ae10c2ac966ea1ed2155d22202390b532d1dea64935c518ada
therock_cmake_subproject_declare(therock-fusilli
    ...
```

Fusilli builds as a series of `.h` files.

A small note on C++ standards: Fusilli and the hipDNN engine plugin for Fusilli
are built on the C++20 standard. We believe this should not pose any issues from an
integration standpoint but happy to revisit this further if the need arises.

#### `fusilli-plugin`

`fusilli-plugin` is currently in a superposition between
[`rocm-libraries`](https://github.com/ROCm/rocm-libraries/tree/1a50a48a748c73d19f75a17d5b99d843fe8d9641/projects/fusilli-plugin)
and
[`shark-ai`](https://github.com/nod-ai/shark-ai/tree/b33f16e77ef00b4c9378fcd5edd3123d72fdcb68/fusilli-plugin). It will complete its move to `rocm-libraries`.

Fusilli-Plugin takes a build time dependency on `therock-fusilli` and
`therock-iree-runtime` (a runtime dependency on IREE will be required when it uses
`libIREECompiler.so`), but otherwise builds as any other project in `TheRock`.

The expected build artifact from the plugin integration is a self-contained
`libfusilliplugin.so`, built with Fusilli headers, and linking IREE runtime
libraries with LTO style optimizations.

### Long term requirements

While the initial integration will just focus on pulling in the hipDNN IREE
plugin into the monorepo, long term the expectation is that Fusilli and IREE
are sourced through official release mechanisms that allow TheRock to
seamlessly pull them in (through lockstep versioning). Some questions that need
to be answered for those are:

1. The expectation is that Fusilli will start using the C-API for the IREE compiler
   (through `libIREECompiler.so`) and reserve the use of `iree-compile` binary
   only for debugging and sharing reproducers. This would require significant
   changes to current IREE workflow. Apart from resolving where the IREE project
   lives, i.e. if it should move into the monorepo as well (unlikely), another
   challenge to solve there is which LLVM version should IREE use. IREE currently
   tracks top-of-main of LLVM pretty closely. This would need to change to use
   either the LLVM version within monorepo or a release version of LLVM/MLIR.

## Revision History

- 2025-10-17: Sambhav Jain: Initial version

- 2025-11-13: Aaron St George: Added detail to Short term plan
