---
author(s): Sambhav Jain, Aaron St. George, and Mahesh Ravishankar
created: 2025-10-16
modified: 2025-10-16
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
1. The hipDNN engine plugin for Fusilli. This specializes Fusilli for use within
   hipDNN specifically for AMD GPUs. Currently it is being developed
   [here](https://github.com/nod-ai/shark-ai/tree/main/fusilli-plugin).
   In addition to Fusilli's dependencies, the plugin also depends on HIP, hipDNN
   frontend/SDK and hipDNN's dependencies transitively.

### Short term plan

The immediate workplan is to move the hipDNN engine plugin (i.e., component 3
above) into `rocm-libraries` (under `dnn-providers` once build tree is normalized
per RFC0003) following guidelines from the MIOpen plugin restructuring effort.
This will be built conditionally (NOT on by default) and will pull in Fusilli
and IREE as external dependencies.

The expected build artifact from the plugin integration is a self-contained
`libfusilliplugin.so` that is linked against Fusilli headers and IREE runtime
sources built and statically linked. The dependency on the IREE compiler is
through the `iree-compile` binary (made available typically through a pip-install),
as Fusilli currently invokes the compiler through its command-line-interface.

A small note on C++ standards: Fusilli and the hipDNN engine plugin for Fusilli
are built on the C++20 standard. We believe this should not pose any issues from an
integration standpoint but happy to revisit this further if the need arises.

### Long term requirements

While the initial integration will just focus on pulling in the hipDNN IREE
plugin into the monorepo, long term the expectation is that Fusilli and IREE
are sourced through official release mechanisms that allow TheRock to
seamlessly pull them in (through lockstep versioning). Some questions that need
to be answered for those are:

1. Where should Fusilli live? Fusilli is a general purpose C++ Graph API around
   IREE and as such is tightly coupled with IREE. A natural home for Fusilli is
   within the same GitHub organization as IREE itself. This will allow Fusilli
   to not only address a gap in the IREE ecosystem for JIT/training use-cases,
   but also participate in the release processes in place for IREE already.
1. The expectation is that Fusilli will start using the C-API for the IREE compiler
   (through `libIREECompiler.so`) and reserve the use of `iree-compile` binary
   only for debugging and sharing reproducers. This would require significant
   changes to current IREE workflow. Apart from resolving where the IREE project
   lives, i.e. if it should move into the monorepo as well (unlikely), another
   challenge to solve there is which LLVM version should IREE use. IREE currently
   tracks top-of-main of LLVM pretty closely. This would need to change to use
   either the LLVM version within monorepo or a release version of LLVM/MLIR.
