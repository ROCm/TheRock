---
author: Alexis Montoison (amontoison)
created: 2026-07-20
modified: 2026-10-27
status: Approved
---

# Distributing Fortran bindings across rocm-systems and rocm-libraries

## Summary

A Fortran *binding* is the set of Fortran declarations that lets a Fortran program call a C library.
This RFC moves the ROCm bindings out of `ROCm/hipfort` and next to the C headers they describe.

- **Where.** HIP and ROCTx go to one `rocm-systems` project, `projects/fortran/`.
  Each math library gets a `fortran/` directory in its own `rocm-libraries` project.

- **How.** A standalone tool, `rocm-fortran-generator`, generates the bindings from the C headers.
  The generated files are committed.
  A regeneration workflow keeps them in sync.
  It runs on a CI runner, apart from the developers' pull requests, and regenerates the bindings from the headers; when something changed, a bot opens a pull request with the result.
  The few facts a header cannot express live in a `metadata/` directory next to each binding.
  For example, `double* alpha` may be one scalar or an array, and may point to host or device memory; a wrong guess still compiles, but passes the wrong data.

- **What ships.** Per library:

  - one module, named after the library (`use rocblas`, `use hip`);
  - a static archive, `lib<lib>_fortran.a`;
  - the compiled module interfaces (`.mod` files, the Fortran counterpart of a precompiled header, specific to one compiler and version) under `include/fortran/<compiler>/`;
  - the generated source under `share/<lib>/fortran/`;
  - a CMake package, `<lib>-fortran`, exporting `roc::rocblas_fortran`, `hip::hip_fortran`, and so on.

  ROCm ships them precompiled for `amdflang`, its Fortran compiler.
  Users of other compilers build them from the installed source.

- **How it builds.** Each project builds its own binding behind one switch, `BUILD_FORTRAN_BINDINGS`.
  It is ON by default on Linux and skipped when no Fortran compiler is found.
  TheRock builds and packages the bindings with the C libraries, using the `amdflang` it builds.

- **When.** The packaged bindings land in ROCm 10.3.
  `ROCm/hipfort` is maintained beside them through 10.x and removed at 11.0.
  The hand-written modules some libraries ship today (`rocblas_module.f90`, `rocrand_m.f90`, ...) are replaced at 10.3.

- **For users.** hipfort users rename their `use` lines and their link targets; their calls do not change.
  The migration guides live in the hipfort documentation.

## What this means for a library maintainer

This section is for the maintainers of HIP, ROCTx, and the ten `rocm-libraries` projects with a binding: rocBLAS, hipBLAS, rocSPARSE, hipSPARSE, rocSOLVER, hipSOLVER, rocFFT, hipFFT (which also carries hipFFTW, the FFTW-compatible interface), rocRAND and hipRAND.

- Your project gains a `fortran/` directory.
  It holds the generated binding, its curated metadata, its CMake build and its tests.
  Your team and the Fortran-bindings team own it together through CODEOWNERS.
  Your team reviews its curated metadata, which records what only the library knows, such as whether a pointer argument is on the host or the device.
- You never write binding code, and you never have to run the generator.
  After your PR merges, the regeneration workflow's next run regenerates the binding, and a bot opens a pull request that the bindings team reviews.
  You may regenerate the binding in your own PR instead, but that is optional.
- Your PR is never blocked by the bindings: in your PR, the project's Fortran build and test jobs run but are not required.
- Your public headers must stay valid C, because the generator parses them with `clang -x c`.
  If a header only compiles as C++, that library's regeneration fails and you get an issue; you fix the header in a normal follow-up PR, and nothing is reverted.
- Your build gains the `BUILD_FORTRAN_BINDINGS` option.
  It is ON by default on Linux: the binding is then built with the Fortran compiler CMake finds (amdflang first, otherwise another one such as gfortran), and skipped when there is none.
  Pass `-DBUILD_FORTRAN_BINDINGS=OFF` to skip it anyway.
- If your library ships a hand-written Fortran module today, it is removed at 10.3: the bindings team opens that PR and writes the CHANGELOG entry, and you review them.

The section [What changes for a library repository](#what-changes-for-a-library-repository) has the details.

## Overview

hipfort provides Fortran interfaces to the HIP runtime and the ROCm math libraries.
They are declared with `bind(C)`, Fortran's standard mechanism for calling C functions.
Today hipfort lives in a separate GitHub repository, and part of its interfaces were written by hand.
Its coverage is only as current as its last manual update: it drifts, and it stalls when no one maintains it.

This RFC puts the generated bindings in the same repository as the C headers they mirror and, for each math library, in the same project:

- the HIP runtime and ROCTx bindings go to `rocm-systems`, in one standalone project, `projects/fortran/`, built against the installed HIP and ROCTx;
- each math library's binding goes to `rocm-libraries`, in a `fortran/` directory of that library's project.

The bindings live in that `fortran/` directory, not in the `include/` tree, so a change to a `.h` can reach its `.F90` in the same repository.
The generator, `rocm-fortran-generator`, is a standalone build-time tool with its own repository.
The bindings stay up to date without any work from the developer who changes a header.
A regeneration workflow regenerates them from the headers, and when they changed, a bot opens a pull request with the result (see [Keeping the bindings in sync](#keeping-the-bindings-in-sync)).

Until ROCm 10.2, the goal is narrow: keep the existing hipfort current with the headers without breaking it.
That is done for 7.14, 10.0 and 10.1.
The real change lands in ROCm 10.3: packaging returns, and the bindings split across `rocm-systems` and `rocm-libraries`.
This RFC calls the two deliverables *tracks*:

- the *packaged track* is the co-located, generated bindings this RFC describes, and is the deliverable from 10.3 on;
- the *compatibility track* is hipfort under its current name and layout, maintained beside them through the 10.x series and removed at ROCm 11.0.

This document fixes the direction and is the source of truth for the design; its [Roadmap and rollout](#roadmap-and-rollout) section holds the schedule.

## Background

hipfort is a thin Fortran-to-C translation layer, with no GPU code of its own.
Its modules declare `interface ... bind(C)` blocks against the ROCm C functions.
They add convenience wrappers on top, which this RFC calls the *ergonomic* forms: for example, a form that takes a Fortran array instead of a raw pointer.

This section describes hipfort as it is today.
The packaged track keeps the shape of its build artifacts: a `.mod` set per compiler and a static archive, separate from the vendor library (the [appendix on hipfort's build artifacts](#appendix-hipforts-build-artifacts) explains why).
It changes the rest: the separate repository, the `hipfort_` prefix, the split modules, and the CUDA backend.
The roadmap says when.

```text
   ┌───────────────────────────────────────────────────────┐
   │  Your Fortran application                             │
   ├───────────────────────────────────────────────────────┤
   │  hipfort: Fortran bind(C) interfaces (*.mod)          │
   │           + generic runtime wrappers (libhipfort*.a)  │
   ├───────────────────────────────────────────────────────┤
   │  ROCm math libraries (.so): librocblas, libhipfft, ...│
   │  HIP runtime         (.so): libamdhip64               │
   ├───────────────────────────────────────────────────────┤
   │  AMD GPU driver / hardware                            │
   └───────────────────────────────────────────────────────┘
```

The sources (`lib/hipfort/*.F90`) are one module per library plus a few base modules:

| File                                                           | Role                                 |
| :------------------------------------------------------------- | :----------------------------------- |
| `hipfort.F90`                                                  | HIP runtime (device, stream, launch) |
| `hipfort_types.F90`, `hipfort_enums.F90`                       | HIP types and enumerations           |
| `hipfort_check.F90`                                            | return-code checking helpers         |
| `hipfort_hipmalloc` / `hipmemcpy` / `hiphostregister`          | generic memory wrappers              |
| `hipfort_{roc,hip}blas` / `fft` / `solver` / `sparse` / `rand` | one module per math library          |
| `hipfort_auxiliary.F90`                                        | device-property helpers              |
| `hipfort_*_types.F90` (hipFFT, hipFFTW, rocRAND)               | library derived types                |
| `hipfort_roctx.F90`, `hipfort_cuda_errors.f90`                 | profiling, CUDA-backend errors       |

Hand-written bindings lag for a simple reason.
A new function, enum, or struct field in a header must be copied by hand, later, by someone other than the header's author.
The result is delay and silent drift, such as a missing `_64` variant or a stale struct layout.
The copying stops entirely when there is no maintainer.

Worse, the hand-written work is already duplicated: several math libraries ship their own Fortran module in their source tree.

- rocBLAS ships `rocblas_module.f90`, and hipBLAS ships `hipblas_module.f90`.
- rocSPARSE ships `rocsparse.f90` (over 7000 lines) plus `rocsparse_enums.f90`.
- hipSPARSE ships `hipsparse.f90` plus `hipsparse_enums.f90`, and hipSOLVER ships `hipsolver_module.f90`.
- rocRAND and hipRAND ship a third variant, gated by `BUILD_FORTRAN_WRAPPER`: the deprecated `rocrand_m` and `hiprand_m` wrappers, with a small `hipfor` HIP module.

Five of these define a module named exactly after the library: `rocblas`, `hipblas`, `rocsparse`, `hipsparse` and `hipsolver`.
rocBLAS, hipBLAS and rocSPARSE install theirs as a public source file.
hipSPARSE compiles its module only into its own samples.
hipSOLVER builds its module into `libhipsolver_fortran.so`, but installs the source only under `BUILD_FORTRAN_MODULE`, an option its CMake never declares, so by default the source is not installed.
The packaged track's modules have the same unprefixed names, so they replace these modules rather than coexist with them (see [What changes for a library repository](#what-changes-for-a-library-repository)).

Each of these modules is maintained independently of hipfort's binding for the same library.
So `rocblas_dgemm` has two separate Fortran interfaces, in two repositories, kept current by two different people, and either copy can drift from the C header and from the other.
Generating the binding next to the header replaces these parallel copies with one source of truth per library.

rocSPARSE goes further.
It also ships fourteen Fortran example programs (`clients/samples/example_fortran_*.f90`) written against its own module.
Yet its documentation never mentions them, or its own module: its one Fortran section points users to hipfort instead.
The same library therefore carries a second Fortran binding and a second example set, maintained apart from hipfort and not documented.
Generating the binding in place, with the examples beside it, brings that work back into the library.

## Design principles

**P1. Co-locate bindings with their headers.**
The binding for `rocblas_dgemm` should live in the same repository as `rocblas.h`.
It is then regenerated from that repository's own headers, with no copy step, and built and tested against the same commit's C library.
Proximity bounds the lag: the binding follows the header on the regeneration workflow's next run, or in the same PR if its author regenerates it.

**P2. Each library builds its own binding; TheRock packages them.**
The deliverable is per library.
Each library ships its own `.F90`, `.mod`, `lib<lib>_fortran.a` and CMake target (`roc::rocblas_fortran`, `hip::hip_fortran`, ...).
They are built by that library's own CMake, from its `fortran/` directory, in the same build as the C library.
HIP and ROCTx are the exception: `rocm-systems`' `projects/fortran/` builds them against the installed runtime.
No separate job gathers the bindings from every repository.
TheRock already builds every project, so it packages the Fortran artifacts with the C ones (see the [Packaging in TheRock](#appendix-packaging-in-therock) appendix).
Fusing the pieces into one package, or building only a subset, stays possible, but the default is per library.

**P3. Keep the generator standalone.**
The generator is build-time tooling, not a runtime dependency.
It lives in its own repository, `ROCm/rocm-fortran-generator`, which is being made public, so that the monorepos' regeneration workflow can run it (see [Rollout](#rollout-and-success-criteria)).
This RFC commits to one thing: the generator's *output* is checked in, as reviewable Fortran, in the library repositories.
Wherever the generator lives, the Fortran-bindings team owns and maintains it, not the compiler developers, and it is not part of the compiler build.
Library repositories run it as a standalone tool and do not vendor it.

**P4. A bot, not hand-copying, keeps the bindings current.**
A C-library developer should not have to learn Fortran, or run the generator, to keep a binding current.
A regeneration workflow keeps a binding current: it reads the headers, regenerates every binding, and when one changed, a bot opens a pull request with the regenerated `.F90`.
Developers may still regenerate a binding in their own PR, but nothing requires it, and no Fortran check ever blocks a header change.
The one manual part is the curated metadata, in each `fortran/metadata/`.
Most changes need none.
When a new function needs a classification, the regeneration reports it to the Fortran-bindings team, who add the one line.

## Non-goals

The following are out of scope for this RFC.

- **Windows.** The target is Linux HPC.
  ROCm's existing Fortran clients are already disabled on Windows: rocBLAS, hipBLAS, hipSOLVER and rocSPARSE all build Fortran only under `NOT WIN32` or `UNIX`.
  A Windows Fortran product is not a goal.
- **Fortran under sanitizers.** The ASan and coverage build variants will ship no Fortran, because ROCm's sanitizer configuration cannot instrument Fortran yet.
  Today the projects still build the bindings under ASan and only skip their tests; TheRock's ASan and coverage builds will turn them off (see the [Packaging in TheRock](#appendix-packaging-in-therock) appendix).
- **A binding for compilers limited to Fortran 77.** Fortran 77 code built with a current compiler can use the bindings; only a compiler that stops below Fortran 90 cannot.
  [Fortran 77 code](#fortran-77-code), under [Language standard](#language-standard), explains what that means and what such a binding would take.
- **The generator as a runtime or shipped product.** `rocm-fortran-generator` is written in Julia and uses Clang.jl to parse the headers.
  It is build-time tooling: none of its code reaches a consumer, the compiler build, or the deliverable.
- **Non-CMake consumer tooling as a first-class path.** CMake's `find_package` is the supported integration.
  A `pkg-config` file or a flags script would be a convenience, not a guarantee.
- **A new build system or test framework.** This RFC defines where the bindings live and how they stay in sync.
  It reuses the monorepos' existing CMake, CODEOWNERS and CI rather than introducing parallel ones.

## Proposal

```text
   BEFORE  (super-repo, always downstream)
     rocm-systems (HIP headers) and rocm-libraries (rocBLAS, ...)
            │  headers copied downstream
            ▼
     hipfort  (separate repo, updated by hand, stalls when unmaintained)
            │
            ▼
     packaging

   AFTER  (co-located sources, built with each library)
     rocm-systems/    projects/hip/*.h, projects/rocprofiler-sdk/.../roctx.h
                      projects/fortran/       hip.F90, roctx.F90, metadata/
     rocm-libraries/  projects/rocblas/*.h  + projects/rocblas/fortran/rocblas.F90, metadata/
                      projects/hipfft/*.h   + projects/hipfft/fortran/hipfft.F90, metadata/
            │  the regeneration workflow regenerates each .F90 (a bot PR per library),
            │  or the header's author does it in the same PR
            ▼
     each project's build (BUILD_FORTRAN_BINDINGS)
            fortran/<lib>.F90  →  <lib>.mod + lib<lib>_fortran.a
                               →  <lib>-fortran package (roc::rocblas_fortran, hip::hip_fortran, ...)
            ▼
     TheRock: builds every project with amdflang, packages the Fortran
              artifacts with the C ones (P2)
```

Where each piece goes:

| Piece                                                                   | Repo                                        | Why                                                  |
| :---------------------------------------------------------------------- | :------------------------------------------ | :--------------------------------------------------- |
| HIP runtime module `hip` (absorbing HIP types, enums, check, auxiliary) | `rocm-systems`, `projects/fortran/`         | HIP lives here (P1)                                  |
| ROCTx module `roctx`                                                    | `rocm-systems`, `projects/fortran/`         | ROCTx lives here (P1)                                |
| Each math library's `<lib>` module (`rocblas`, `hipfft`, ...)           | `rocm-libraries`, `projects/<lib>/fortran/` | next to its C headers (P1)                           |
| Curated metadata (`argkinds`, `native`, `optional`, ...)                | the module's `fortran/metadata/`            | manual, curated next to the header (P1)              |
| Generator configuration (`<lib>.toml`)                                  | the generator                               | project convention, read by the generator's code     |
| CUDA maps (`<lib>.txt`, `<lib>_enums_cuda.txt`, `*.cuda.finc`)          | the generator (compatibility track only)    | CUDA is not in the packaged track (see CUDA backend) |
| The generator (`rocm-fortran-generator`)                                | its own repository                          | build-time Fortran tooling (P3)                      |
| Build (per-library `.mod`/`lib<lib>_fortran.a`, CMake package)          | each project's `fortran/CMakeLists.txt`     | per-library deliverable (P2)                         |
| Packaging                                                               | TheRock                                     | already builds and packages every project (P2)       |

### Keeping the bindings in sync

The bindings are kept in sync by a regeneration workflow, separate from the developers' pull requests:

```text
   Developer opens a PR editing  projects/rocblas/.../rocblas.h
            │   (may regenerate rocblas.F90 in the same PR: optional)
            ▼
   PR reviewed and merged as usual: no Fortran check, never blocked
            │
            ▼
   Regeneration workflow, on a CI runner, fully offline:
     reads the headers of the default branch, regenerates every binding
            │
            ├─ no binding changed        → nothing to do
            │
            ├─ some bindings changed     → a bot opens (or updates) one PR per
            │                                library with its regenerated .F90;
            │                                the Fortran-bindings team reviews it
            │
            └─ a header is not valid C   → that library's regeneration fails,
                                             and an issue goes to the header's
                                             owners (see C-header hygiene)
```

The workflow is *out-of-band*: it runs on its own, reads the default branch, and never runs on, blocks, or comments on a developer's pull request.

**For a developer who changes a header:**

- Nothing is required: the bot regenerates the binding after the PR merges.
- In the PR, the Fortran build and test jobs run but are not required.
  Until the bot's pull request lands, a removed or renamed C function can make a binding test fail to link.
  Since each archive has one section per procedure and the tests link with `-Wl,--gc-sections`, only a test that calls that function fails.
- The developer may also regenerate the binding in the same PR, with the generator's `--overwrite` option (it needs Julia; see the [generator appendix](#appendix-the-generator-rocm-fortran-generator)).
  This touches `fortran/`, so the bindings team then has to review the PR.

**How the bot works:**

- **When.** On each push to the default branch that changes a public header, a `fortran/metadata/` directory or the pin file; also weekly, and by hand.
  A binding lags its header by one run at most.
- **What.** Every binding of the monorepo, so a change to a shared header (such as HIP, or rocBLAS for rocSOLVER) or a new generator release is caught too.
- **Output.** One pull request per library, on the branch `bot/fortran/<lib>`, recreated on each run and closed when it has nothing left to change.
  A blocked library never delays the others.
- **Review.** The Fortran-bindings team, which co-owns `fortran/` through CODEOWNERS, reviews and merges these pull requests within five business days.
  The project's Fortran jobs are required on them, and the bindings team fixes a pull request that fails.
- **Flags.** The pull request lists the changes that need a decision: a public Fortran name removed, an existing signature changed, an argument renamed, an argument not yet classified.
- **Inputs.** The generator release, its Julia environment and, for `rocm-libraries`, the source of the HIP headers are pinned in a file of the monorepo.
  The bot's output is the reference: the generator warns when a manual run uses other inputs, and the bot's next run overwrites its result.

**Special cases:**

- **A new function that needs metadata.** The bindings team adds the line in its own pull request.
  Meanwhile the bot emits only the raw `type(c_ptr)` form for that argument, so classifying it later adds a form and breaks no caller.
- **A header that is not valid C.** Only that library's regeneration fails, and an issue goes to the header's CODEOWNERS.
  The binding stays at its last good state until the header is fixed.
- **Releases.** The bindings team merges a fresh bot pull request before each release branch is cut, and the bot also runs on release branches, so cherry-picks are followed.
  A release is tagged only when regenerating its bindings gives no diff; this check never blocks a developer's PR.

### Maintaining the bindings under the split

A binding is more than its `.F90`, and the split keeps all of it next to the library.

- **Generated `.F90`.** It is derived from the header.
  The regeneration workflow regenerates it and the bot proposes the result, and no one edits it by hand (P4).
- **Curated metadata.** This is the manual part: the intent the C API cannot express (see [Curated metadata](#curated-metadata-the-manual-surface), in the [generator appendix](#appendix-the-generator-rocm-fortran-generator)).
  It lives in `fortran/metadata/`, next to the header it describes, and the generator reads it when it runs on the checkout (P1).
  Most header changes need none.
  A scalar pointer argument that is still unclassified is reported by the regeneration, for the Fortran-bindings team to classify.
  A classification that matches its function but is wrong is caught instead by the runtime tests and by the CODEOWNERS review of the metadata (see [Keeping the classifications correct](#curated-metadata-the-manual-surface), in the [generator appendix](#appendix-the-generator-rocm-fortran-generator)).
- **Tests.** A library's Fortran tests live in its `fortran/test/` and follow one convention (see [Tests and documentation](#tests-and-documentation), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).
  Coverage is uneven.
  HIP, ROCTx, rocSOLVER, hipSOLVER, rocSPARSE, hipSPARSE, rocFFT, hipFFT, hipFFTW, rocRAND and hipRAND have a `fortran/test/`, mostly ported from hipfort.
  rocBLAS and hipBLAS are tested for now only through their existing `clients/`: the gtest Fortran shims and the samples, which now link the generated binding.
  Filling that gap is part of the rollout, not a precondition for co-location.
  The bot's pull request compiles the tests without a GPU; the packaging or validation job runs them on an AMD GPU.
- **Docs and examples.** The libraries' existing Fortran examples (the rocBLAS, hipBLAS, rocSPARSE and hipSPARSE samples) move to the generated module.
  The packaged modules carry no docstrings yet (see [Tests and documentation](#tests-and-documentation), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).
  The current hipfort documentation can likewise be split across the libraries it covers.

### Building and shipping a binding

Each project builds its binding in its own CMake build, from its `fortran/` directory.
For HIP and ROCTx, that directory is `rocm-systems`' `projects/fortran/`.
TheRock builds every project nightly and on release branches.
It compiles the bindings in the C dependency order (rocBLAS before rocSOLVER), with the `amdflang` it builds, and packages them with the C libraries.

For each compiler, a binding installs:

- a `.mod` set under `include/fortran/<compiler>/`;
- a static archive under `lib/fortran/<compiler>/` (`librocblas_fortran.a`, `libhip_fortran.a`, ...);
- its generated `.F90` under `share/<lib>/fortran/`;
- a CMake package, `<lib>-fortran`, exporting a target in the C library's namespace (`roc::rocblas_fortran`, `hip::hip_fortran`, ...).

The deliverable is per library: `find_package(rocblas-fortran)` gives the Fortran binding and, through it, the C library.
Fusing everything into one CMake package, or shipping only a subset, stays possible (P2).
The build appendix has the details: the compiler guard, the CMake package, the install tree, and the test convention.

**Distro packages.**
A library's binding ships in that library's existing development package, not in a separate `-fortran` package: it adds a few MB and no dependency for C users.
The development packages follow the existing `amdrocm-*` stems, one per library family, and are named `-devel` in RPM and `-dev` in DEB:

| Package                 | Bindings                |
| :---------------------- | :---------------------- |
| `amdrocm-runtime-devel` | HIP                     |
| `amdrocm-blas-devel`    | rocBLAS, hipBLAS        |
| `amdrocm-sparse-devel`  | rocSPARSE, hipSPARSE    |
| `amdrocm-solver-devel`  | rocSOLVER, hipSOLVER    |
| `amdrocm-fft-devel`     | rocFFT, hipFFT, hipFFTW |
| `amdrocm-rand-devel`    | rocRAND, hipRAND        |
| `amdrocm-profiler-base` | ROCTx                   |

The HIP and ROCTx bindings are both built by `projects/fortran/`, but each ships with its C library.
ROCTx has no development package of its own: its headers and CMake config are in `amdrocm-profiler-base`, so its binding goes there too.
The packages carry the `.mod` files of `amdflang`, the compiler ROCm ships; users of another compiler build the binding from the installed `.F90` sources.
`amdrocm-core-devel` depends on every package in this table except `amdrocm-profiler-base`, so it installs every binding but ROCTx.
`amdrocm-core-sdk` adds `amdrocm-profiler-base`, through `amdrocm-developer-tools`, and with it the ROCTx binding.

For a Fortran user who wants one command, the `amdrocm-fortran` metapackage already declared in TheRock's packaging (today disabled, and depending only on `amdrocm-llvm`) gains a dependency on the seven packages above.
It then installs the compiler and every binding.
Because those packages are built per GPU architecture, `amdrocm-fortran` becomes per-architecture too, like `amdrocm-core-devel`.
The opt-in HPC SDK development metapackage, `amdrocm-hpc-sdk`, may depend on `amdrocm-fortran`; that is for the HPC SDK owners to decide.
The `hipfort` package stays separate until 11.0 and installs side by side.

Every project with a binding takes the same options.
Their per-library forms use the project's prefix: `ROCSPARSE_`, `HIPFFT_` and `HIPFFTW_`, `HIP_`, and so on.
Every routine has a raw form that takes C pointers (see [Calling styles](#calling-styles)).
The *array tier*, selected by `FORTRAN_ARRAY_INTERFACES`, adds forms that take Fortran arrays: one per array rank (`assumed-shape`, Fortran 2008, the default), a single form for any rank (`assumed-rank`, Fortran 2018), or none.

| Option                         | Default                                             | Effect                                                                                 |
| :----------------------------- | :-------------------------------------------------- | :------------------------------------------------------------------------------------- |
| `BUILD_FORTRAN_BINDINGS`       | `${UNIX}` (`OFF` on a hip\* library's CUDA backend) | build the binding; skipped, with a `STATUS` line, without a Fortran compiler           |
| `<LIB>_BUILD_FORTRAN_BINDINGS` | unset                                               | per-library override of `BUILD_FORTRAN_BINDINGS`                                       |
| `FORTRAN_ARRAY_INTERFACES`     | `assumed-shape`                                     | array overloads: `none`, `assumed-shape` (Fortran 2008), `assumed-rank` (Fortran 2018) |
| `BUILD_FORTRAN_CLIENTS`        | `ON`                                                | build and register `fortran/test/`; in-tree, also needs the project's test switch      |
| `<LIB>_BUILD_FORTRAN_CLIENTS`  | unset                                               | per-library override of `BUILD_FORTRAN_CLIENTS`                                        |
| `<LIB>_FORTRAN_COMPILER_DIR`   | from the compiler ID                                | the `<compiler>` install subdirectory, to keep a second version of one compiler apart  |
| `<LIB>_HAVE_FORTRAN_BINDINGS`  | (result)                                            | true when this library's binding was built                                             |

A few projects add a guard of their own:

- rocSPARSE builds no binding under `BUILD_ROCSPARSE_ILP64`, and installs nothing under `BUILD_CLIENTS_ONLY`;
- hipSPARSE installs nothing under `HIPSPARSE_ENABLE_HOST=OFF`;
- `projects/fortran/` takes its install component from `HIP_FORTRAN_INSTALL_COMPONENT` (default `dev`).

To build a binding with its tests and run them, standalone against an installed ROCm:

```bash
cmake -S projects/rocsparse/fortran -B build/rocsparse-fortran -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build/rocsparse-fortran
ctest --test-dir build/rocsparse-fortran
```

The tests of a `rocm-libraries` binding need the HIP binding installed (`find_package(hip-fortran)`): put its prefix on `CMAKE_PREFIX_PATH`.

## What changes for a library repository

For a `rocm-systems` or `rocm-libraries` project, the change is small and bounded.

- A `fortran/` subdirectory holds the generated binding, its curated metadata (`fortran/metadata/`), its build, and its Fortran tests (`fortran/test/`).
  In `rocm-systems`, HIP and ROCTx share one such directory, `projects/fortran/`.

- A regeneration workflow runs the generator outside the developers' PRs, and a bot opens a pull request per library when a binding changed (see [Keeping the bindings in sync](#keeping-the-bindings-in-sync)).
  Developers may regenerate their binding in their own PR, but nothing requires it.

- A CODEOWNERS entry gives `fortran/` to the library's team and the Fortran-bindings team together.
  The library's team reviews the curated metadata, which records what only the library knows.

- C and C++ developers do not write binding code.
  The bot proposes the regenerated bindings, and the bindings team maintains the manual metadata, which the library's team reviews.

- A Fortran failure never blocks a C or C++ PR, because the Fortran jobs are not required there.
  A failed regeneration, for example on a header that is no longer valid C, opens an issue for the header's owners.

- Any Fortran module the library ships today, such as `rocblas_module.f90`, is replaced by the generated binding.

  - The bindings team opens the removal PR and writes the CHANGELOG entry; the library's maintainers review it.
  - The removal covers the module under `library/` and its `<lib>_enums` companion, with no compatibility shim: `use rocsparse_enums` becomes `use rocsparse`.
  - For rocRAND and hipRAND, it also covers `BUILD_FORTRAN_WRAPPER` and the wrappers it built; their Fortran tests move to `fortran/test/`.
  - The Fortran test shims in `clients/` stay, but link the generated target instead of compiling the hand-written source.

- The generated module follows the C API more closely than a hand-written one, so a few call sites change for its users:

  - output scalars are typed and passed by reference, instead of through `c_loc`;
  - enum arguments use the enum's integer kind;
  - handle creation takes the handle variable itself.
    This last change needs care: the old `rocblas_create_handle(c_loc(handle))` still compiles, but leaves the handle unset.

  Each library's CHANGELOG lists these changes.

## The generated bindings

The generated modules are what a Fortran program `use`s and links.
This section describes them: the module names, how they are called, the Fortran standard they need, and the CUDA option.
The generator that produces them is described in its own [appendix](#appendix-the-generator-rocm-fortran-generator).

### Naming and module structure

The compatibility track keeps today's names and layout: the `hipfort_` prefix, and separate interface, type and enum modules per library.
So `use hipfort_rocblas` keeps working.
The prefix also avoids a clash with a library's own `rocblas` module.

The packaged track changes both.
It drops the prefix, as NVIDIA does (`use cublas`, not `nvfort_cublas`), and puts each library's interfaces, types and enums into a single module.
The unprefixed name does not clash with the libraries' own modules (`rocblas_module.f90`, `rocsparse.f90`, ...), because the split removes those: the generated module is the only `rocblas` in the tree.

- The HIP runtime module becomes `hip`: `use hip`, not `use hipfort`.
- The cross-cutting modules `hipfort_check` and `hipfort_auxiliary` disappear.
  Their contents come from the library module the program already uses: `hipCheck` from `hip`, `rocblasCheck` from `rocblas`.
  Handles are `type(c_ptr)` (see [Ambiguous or unmappable C](#ambiguous-or-unmappable-c), in the [generator appendix](#appendix-the-generator-rocm-fortran-generator)).
- A library that needs another library's types `use`s that library's module: rocSOLVER uses rocBLAS.

The generator emits the packaged, unprefixed modules that are committed next to the headers.
`ROCm/hipfort` keeps its split, prefixed layout until 11.0.
Both are generated, so keeping the two costs little.

### Calling styles

Each routine is a single *generic*: Fortran's overloading mechanism, under the routine's public name.
A generic groups several *specifics*, concrete procedures that differ in their argument types.
The compiler picks the specific from the actual arguments, so a caller passing a Fortran array never calls `c_loc` itself.

- **Raw `type(c_ptr)`.** This is today's hipfort interface, and what OpenMP target data needs.
  Existing code keeps working.
  As in hipfort, the raw `bind(C)` specific of a routine `X` is named `X_` and stays callable by that name.
  Adding an ergonomic form later therefore never changes what `X` or `X_` means.
- **Ergonomic forms, under the same name.** A plain Fortran array, a scalar passed by reference, and an optional argument just work.
  For ROCTx, so does a `character(kind=c_char)` string.

The ergonomic array forms take a Fortran array and pass its address to C with `c_loc`, which sets a contract:

- **The array must be contiguous and have the `TARGET` attribute** (the Fortran attribute that allows taking an object's address).
  A non-contiguous section, such as `a(1:n:2)`, is passed as it is, not copied: no array form declares its arguments `contiguous`, in either array tier.
  `c_loc` then returns the address of its first element, and the library silently reads the wrong elements.
- **The array must outlive the operation.**
  It must stay allocated, and the host must not modify an array the device reads, until the operation completes on its stream: after a `hipStreamSynchronize` or an event, exactly as with the C API.
- **Stream-ordered calls need no copy, and get none.**
  Most math-library routines and the HIP `*Async` calls are *stream-ordered*: they enqueue the work and return before the device is done.
  An ergonomic form passes the caller's own array, never a copy, so it is safe for such calls under the two rules above.
  The one trap is an actual argument that is an expression rather than a variable: the compiler evaluates it into a temporary, which is freed while the device may still read it.
- **No wrapper makes its own copy.**
  A wrapper's own local data, such as a computed shape or stride array or a NUL-terminated copy of a string, is freed when the wrapper returns.
  The generator therefore never passes a pointer to such data for an argument the device uses after the call returns.

That is why no argument is declared `contiguous`.
With that attribute, the compiler would copy a non-contiguous section into a host temporary.
For an array in device memory (`hipMalloc` + `c_f_pointer`), that copy reads device memory from the host, which faults on a GPU without XNACK (GPU page-fault retry) enabled.
For a stream-ordered call, the temporary would also be freed before the device uses it.
Contiguity is therefore a correctness requirement on the caller, not a performance one.

### Language standard

The minimum is Fortran 2003, and two features raise it in well-defined steps:

- **Fortran 2003 (`iso_c_binding`)** is the minimum: `bind(C)`, `type(c_ptr)`, `c_loc`, `VALUE`, interoperable derived types, and `enum, bind(c)`.
  The raw `type(c_ptr)` interfaces need nothing more, so a site restricted to Fortran 2003 can use them as they are, built with `FORTRAN_ARRAY_INTERFACES=none`.
- **Fortran 2008** is needed for the ergonomic array forms: *assumed-shape* array arguments, which take their shape from the caller's array (`dimension(:)`, `dimension(:,:)`, ...), and passing a Fortran array through `c_loc`.
- **Fortran 2018** is needed only for the opt-in *assumed-rank* forms (`-DFORTRAN_ARRAY_INTERFACES=assumed-rank`), where one `dimension(..)` interface accepts an array of any rank.

The default tier, `assumed-shape`, is therefore Fortran 2008, and so is the precompiled amdflang set.
CI will hold each tier to its level by compiling every module with gfortran at `-std=f2003` for `none`, `-std=f2008` for `assumed-shape` and `-std=f2018` for `assumed-rank`.
A construct from a later standard then fails the build instead of reaching a site that relies on the lower one.
The tiers meet these levels today when checked by hand, but the generator's compile gate builds all three at `-std=f2018`, so nothing enforces them yet.

The supported compilers are gfortran, amdflang and Cray CCE; their minimum versions are an open question.
Each `fortran/` directory also carries toolchain files for Intel (`ifx`, `ifort`) and NVIDIA HPC (`nvfortran`).
The appendix on `.mod` incompatibility explains why a `.mod` cannot be shared across compilers, or even across versions of one compiler.

#### Fortran 77 code

The question comes up often, so here is the answer in one place.

- **Fortran 77 code can use the bindings.**
  What matters is the compiler, not the age or the form of the source.
  A Fortran 77 code base built with a current compiler (gfortran, amdflang, ...) calls the bindings like any other code.
  The file that calls them adds one `use rocblas` line; fixed-form source, implicit typing and the rest of the code are untouched.
- **Only a compiler limited to Fortran 77 cannot.**
  The bindings are Fortran modules built on `iso_c_binding`, from Fortran 2003.
  A compiler that stops below Fortran 90 has no modules and no `use` statement, so it cannot read them.
  The same holds for a current compiler held to Fortran 95 by a flag (`gfortran -std=f95`), which rejects `iso_c_binding`.
  The file that `use`s a binding is then compiled in the compiler's default mode; fixed-form source needs no extra flag.
- **Nothing is taken away, and a path stays open.**
  hipfort never offered a path for such a compiler either.
  One would be a separate artifact: a C shim library with Fortran 77 linkage, plus an include file of `PARAMETER` constants, emitted by the same generator from the same declarations and metadata.
  It would still be a project of its own, with a new emitter, tests, packaging and symbol-name conventions, for callers that cannot use the modules.
  An application that needs it would get its own RFC, scoped to its entry points; nothing decided here would change.

### CUDA backend

The generated modules are `.F90` files, with an uppercase extension, so the compiler runs the C preprocessor on them.
For hipfort, the generator also emits `#ifdef USE_CUDA_NAMES` branches: compiled with `-DUSE_CUDA_NAMES`, hipfort's sources bind to CUDA instead of HIP.
The generator cannot infer CUDA names from the AMD headers, so data files in the generator feed those branches:

- a curated `hipName cuName` map per library, checked against dumps of the CUDA symbols;
- enum values and struct layouts, which tools under `src/tools/` derive from the CUDA headers.

The packaged modules carry no CUDA branches; their preprocessing only selects the array tier.

The NVIDIA backend lives only in the compatibility track (`ROCm/hipfort`), kept there until ROCm 11.0 to compare HIP and CUDA.
The packaged track does not carry it: it is not co-located, not packaged, and not shipped in ROCm.
A `hip*` library configured for its CUDA backend (hipBLAS, hipSPARSE, hipSOLVER, hipFFT, hipRAND) defaults `BUILD_FORTRAN_BINDINGS` to `OFF`.
Forcing it on builds the ROCm-generated module, with a warning that it is not tested against the CUDA-backed library.

## Migration and compatibility

Existing hipfort users make two mechanical edits, and lose only one capability, the CUDA backend (see the end of this section).
Routine names, argument lists and calling styles do not change.
The `use` lines and the link targets change; the calls do not, so the migration is find-and-replace, not a rewrite.

**1. Module names.** The `hipfort_` prefix is dropped, and each library becomes a single module.

| Old                                                | New           |
| :------------------------------------------------- | :------------ |
| `use hipfort` (+ `hipfort_types`, `hipfort_enums`) | `use hip`     |
| `use hipfort_rocblas` (+ `_enums`)                 | `use rocblas` |
| `use hipfort_<lib>` (+ `_enums`, `_types`)¹        | `use <lib>`   |

¹ `_types` exists only for rocRAND, hipFFT and hipFFTW.

- The libraries' own enum modules fold in the same way: `use rocsparse_enums`, `use hipsparse_enums` and `use rocblas_enums` become `use <lib>`.
- The HIP memory helpers (`hipfort_hipmalloc`, `hipfort_hipmemcpy`, `hipfort_hiphostregister`) fold into `hip`.
- The cross-cutting modules (`hipfort_check`, `hipfort_auxiliary`) fold into the library module the program already uses: `hipCheck` comes from `hip`, and `rocblasCheck` from `rocblas`.
  Handles stay `type(c_ptr)`.

**2. Link target.** The single `libhipfort-*.a` becomes one archive and one CMake target per library, in the C library's own namespace.
`find_package(<lib>-fortran)` finds it and also brings in the C library.
Once rocm-cmake lets a library's generated `<lib>-config.cmake` optionally include the Fortran config, `find_package(<lib>)` alone will do (see [Building the bindings](#building-the-bindings), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).

```cmake
# before
find_package(hipfort REQUIRED)
target_link_libraries(app PRIVATE hipfort::rocblas)

# after
find_package(rocblas-fortran REQUIRED)
target_link_libraries(app PRIVATE roc::rocblas roc::rocblas_fortran)
```

The namespace follows each C library, not a hip-versus-roc rule:

- `hip::` for the HIP runtime, hipFFT, hipFFTW and hipRAND;
- `roc::` for ROCTx, rocBLAS, hipBLAS, rocSPARSE, hipSPARSE, rocFFT, rocSOLVER, hipSOLVER and rocRAND, matching the `roc::hipsolver_fortran` that standalone hipSOLVER already exports.

Without CMake, a build passes the per-compiler directories itself, for example `-I/opt/rocm/include/fortran/amdflang -L/opt/rocm/lib/fortran/amdflang -L/opt/rocm/lib -lrocblas_fortran -lrocblas -lamdhip64`.

**Both bindings can be installed at once**, so a code base can migrate one file at a time.
Their module names differ, their `.mod` files land in different directories, their archives have different file names, and their object symbols are decorated with the module name (`__hipfort_rocblas_MOD_...` against `__rocblas_MOD_...`).
The one rule is not to `use` both bindings of the same library in one scope: they export the same public names, so a reference would be ambiguous.
The clash is limited to that scope at compile time, which is exactly why a file-by-file migration works.

One capability does not carry over: the CUDA (`nvptx`) backend is not built for the packaged track (see [CUDA backend](#cuda-backend)), and keeps working on the compatibility track until 11.0.
`hipfort_cuda_errors`, which only serves that backend, has no packaged counterpart.
`roctx` does carry over, as `use roctx` and `roc::roctx_fortran`, from `rocm-systems`' `projects/fortran/`.

The user-facing migration guides live in the hipfort documentation, not in this RFC, so that they follow the packaging as it lands.
They give the full per-library table, the compile and link lines, when users have to build a binding themselves, and a scripted rename.
There are two guides, for two populations:

- hipfort users, who rename, and can overlap the two bindings until 11.0;
- users of a library's own module, who change some call sites at 10.3.

## Roadmap and rollout

### Version targeting and roadmap

The bindings target the ROCm version of the headers the generator reads: the `develop` branch of the `rocm-systems` and `rocm-libraries` monorepos, or a ROCm install.

hipfort's committed bindings are already regenerated from the ROCm 7.14, 10.0 and 10.1 headers (hipfort 0.8.0, 0.9.0 and 0.10.0); the next step is 10.2.
Packaging targets 10.3.
10.3 is the target, but the design does not depend on it: this RFC fixes the destination and the order of the steps.

The transition runs on the two tracks, both generated from the same headers.

- The compatibility track is `ROCm/hipfort` with its current layout: the `hipfort_` prefix, the CUDA backend, and separate interface, type and enum modules.
  It is kept as source, not packaged.
  Users who build it themselves keep its layout, module names and [CUDA backend](#cuda-backend) through ROCm 7.14, 10.0, 10.1 and 10.2.
- It stays **maintained beside the packaged track through the whole 10.x series**, so that from 10.3 to 11.0 users can migrate on their own schedule, one file at a time.
  Maintained means regenerated against each 10.x release's headers and fixed when it breaks, by the Fortran-bindings team.
  New features land only in the packaged track.

A packaged binding is versioned with its library, and stays compatible across a major ROCm series.

- The interfaces of a binding built for 10.3 stay source- and link-compatible with the 10.3 library and any later 10.x one.
- Its CMake package accepts any version with the same major number (`SameMajorVersion`).
- A precompiled `.mod` still needs the compiler that wrote it (see the [appendix on `.mod` incompatibility](#appendix-fortran-mod-and-a-incompatibility-across-compilers-and-versions)).

Users therefore revisit their build at a major release, not at every point release.
At link time, this promise rests on each archive being compiled with one section per procedure and linked with `-Wl,--gc-sections`.
A wrapper whose C function is missing from the installed library then breaks only the programs that call it (see [Why not fuse with the vendor library](#appendix-hipforts-build-artifacts), in the [appendix on hipfort's build artifacts](#appendix-hipforts-build-artifacts)).

Within a major series, a binding only grows: new routines, new specifics in a generic, new enumerators and types.
An existing Fortran signature changes only when its C function does, or at a major release, even when the change comes from the generator or a metadata entry rather than the header.
The exception is a fix.
An interface that does not match its C function (a wrong type or kind, a scalar bound as an array) is corrected in the release that finds it.
Code compiled against it was already calling the library incorrectly.

The compatibility track covers only one of the four ways users get hipfort today; the table lists all four.

| How you get hipfort today                                                           | What the transition does to you                                                                                                     |
| :---------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| You build it from source                                                            | Nothing. The compatibility track keeps its layout, module names, and CUDA backend.                                                  |
| You install a package                                                               | Nothing: ROCm 10.0 to 10.2 ship no hipfort package, `.mod`, or archive, so there is nothing to lose. Packaging *returns* at 10.3.   |
| You use a library's own Fortran module (`rocblas_module.f90`, `rocsparse.f90`, ...) | A real break at 10.3, when the generated module supersedes it. See What changes for a library repository.                           |
| You build one code on several ROCm versions, some before 10.3 (for example 6.x)     | Nothing until your oldest system reaches 10.3: use the hipfort release tagged for each ROCm (`rocm-<version>`). See the note below. |

A hipfort package from an earlier ROCm keeps working against that ROCm.
Once such a system moves to 10.3 or later, it is in the last row of the table.
For that row, a thin wrapper module can also select `use hipfort_<lib>` or `use <lib>` per system at compile time, and the packaged bindings can be generated from an older release's headers on request.

Only the third row is a break, and it is planned and coordinated with each library, not a side effect.
If a distribution still publishes its own hipfort package, that package continues through the transition, and any removal is announced together with that distributor.
Keeping both tracks until 11.0 costs little, because both are generated rather than written by hand.

### Rollout and success criteria

The rollout has three stages, each with a clear exit.

1. **Bring current (7.14 / 10.0 / 10.1 / 10.2).** The committed bindings match the current headers; 7.14, 10.0 and 10.1 are done (hipfort 0.8.0 to 0.10.0).
   A binding is refreshed by regenerating it and committing the result by hand, which is how the co-location PRs were produced; they stay in draft.
   Meanwhile the regeneration workflow is prepared.
   It needs a public generator, so that the monorepos' CI can run it, and a workflow and a bot identity in each monorepo.
   The generator repository, `ROCm/rocm-fortran-generator`, is being made public.

1. **Co-locate and package (10.3).** Each in-scope library has a `fortran/` directory, each project builds its `.mod` set and `lib<lib>_fortran.a`, and TheRock packages them.
   The regeneration workflow and its bot run on those directories, once their inputs are trustworthy:

   - the generator, its toolchain and the HIP header source are pinned in each monorepo's pin file;
   - each library's header input is defined;
   - the metadata check fails on unmatched entries;
   - a failed regeneration, such as a header that is valid C++ but not valid C, opens an issue for the header's owners;
   - each archive is compiled with one section per procedure, and the tests link with `-Wl,--gc-sections` (done in the twelve pull requests).

   The backlog appendix details each of these.
   Before the first release that packages the bindings, every `bind(C, name=)` label must resolve against the shipped libraries.

1. **Retire the compatibility track (11.0).** `ROCm/hipfort`, under its current name, is removed, and the packaged track is the only deliverable.

The rollout is done when:

- `projects/fortran/` (`hip`, `roctx`) and the `fortran/` directories of the ten in-scope `rocm-libraries` projects (eleven modules) carry generated source, curated metadata where needed, and tests;
- a header change that changes a binding is followed by the bot's pull request within one run, and that pull request is merged within five business days;
- every release branch is cut with no binding drift, and every release tag passes the regeneration check;
- `find_package(<lib>-fortran)` exposes `<ns>::<lib>_fortran` when built, where `<ns>` is the C library's namespace, and so does `find_package(<lib>)` once rocm-cmake can include the Fortran config;
- TheRock builds and packages the bindings (see the [Packaging in TheRock](#appendix-packaging-in-therock) appendix);
- each array tier compiles at its declared standard (see [Language standard](#language-standard));
- an external program compiles and links against the installed package with at least amdflang and gfortran.

The co-location step is open as one pull request per library: the HIP binding and ROCTx in `rocm-systems`, and one per `rocm-libraries` project.
Every `rocm-libraries` binding test uses `hip`, except rocRAND's and hipRAND's version checks, so these tests need the HIP binding installed.
Until it is, they are skipped at configure time, and the `rocm-libraries` pull requests stay in draft.
All these pull requests share one shape:

- the options of Building the bindings;
- a static `<lib>_fortran` target in the C library's namespace;
- the `<lib>-fortran` package beside the C one, with a forwarder config so that `find_package(<lib>-fortran)` resolves;
- the install tree of Layout after the split;
- the test convention of Tests and documentation.

What remains before they leave draft:

- each `rocm-libraries` `fortran/` gains its `metadata/`; only the HIP binding has one so far, and ROCTx needs none;
- hipBLAS drops the temporary `hip` stub module its clients build until the HIP binding is installed;
- the projects wire `fortran/` the same way: today rocSOLVER adds it inside `if(BUILD_LIBRARY)` while the others add it unconditionally, and some inline the amdflang preference while others keep it in `fortran/cmake/ROCmFortranCompiler.cmake`.

rocBLAS's and hipBLAS's `fortran/test/` come after, as part of the rollout.

## Open questions and risks

This is a draft, and several points are still open.

- **Cross-repository header dependencies: a risk, settled.**
  Some bindings depend on another library's headers (rocSOLVER on rocBLAS) or on the HIP runtime, so the modules have a compile order.
  This mirrors the existing C dependency graph, and is resolved the same way: build order in TheRock, and `find_package` for a standalone build of one repository (see [Building the bindings](#building-the-bindings), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).
  The regeneration workflow, which regenerates every binding on each run, catches drift across headers.
- **Assumed-rank interfaces (Fortran 2018): decided, with a residual risk.**
  The default stays the `assumed-shape` tier, whose per-rank forms work on older compilers.
  A build option, `-DFORTRAN_ARRAY_INTERFACES=assumed-rank`, selects assumed-rank instead.
  It is one option with three values rather than two booleans, because the tiers exclude each other and two booleans could express an illegal combination.
  Enabling assumed-rank *replaces* the per-rank forms rather than adding to them: the compiler cannot tell an assumed-rank argument from the per-rank ones by rank, so they cannot legally coexist in one generic.
  Fortran 2018 assumed-rank is not equally mature in gfortran, flang, ifx and Cray CCE, and HPC sites adopt new standards late.
  It compiles with amdflang today, but corner cases are a risk.
  It can become the default later, once the compilers are proven.
  It would also shrink the generated sources and their build time; for example, `hipfort_hipmalloc` is about 23k lines in today's hipfort.
- **Minimum compiler versions.**
  The minimum standard is settled at Fortran 2003 (see [Language standard](#language-standard), and [Fortran 77 code](#fortran-77-code)); the minimum compiler versions are not.
  The question worth asking the large HPC sites is which compiler versions they pin.
- **Size of the generated files.**
  A library's binding is one generated file, and for the large APIs it is very large: about 46k lines for `rocsparse.F90`, 43k for `hipsparse.F90`, and 58k for `rocblas.F90`.
  That is too large for GitHub to show in a diff, a concern raised in review.
  Two options:
  - split the output per C header: one `.F90` per header, still assembled into one module (with Fortran `include` or submodules), which keeps diffs reviewable at the cost of more files;
  - keep one file, treated like any generated artifact: marked `linguist-generated`, and reviewed through the generator and the metadata rather than line by line.
    The regeneration workflow makes this option safer, because the regenerated diff is then produced by the tool rather than by hand.
- **Docstrings and the documentation site (Sphinx and Doxygen).**
  The layout is decided (see [Tests and documentation](#tests-and-documentation), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).
  `fortran/` has no `docs/` subdirectory, and each library documents its Fortran API from its existing top-level `docs/`.
  For HIP and ROCTx, the documentation is built from `projects/fortran/`, or from HIP's and rocprofiler-sdk's `docs/` with their owners' agreement, since the bindings otherwise stay out of those projects.
- **Separate Fortran distro packages.**
  The bindings ship in their library's development package (see [Building and shipping a binding](#building-and-shipping-a-binding)).
  The alternative is one `amdrocm-<stem>-fortran` package per library family, so that a C-only install carries no Fortran files.
  The install tree does not depend on this choice, so the packages can be split later without moving a file.
  It is not taken now, for three reasons:
  - the development packages are built per GPU architecture, and an architecture-independent Fortran package has no simple way to depend on the right one;
  - every library's artifact descriptor would need a new `fortran` component, and the Fortran config would have to be excluded from `dev` by file name, since it sits beside the C config;
  - the saving for a C-only install is a few MB.

## Alternatives considered

**A single bindings super-repository.**
All bindings could stay in one dedicated repository: today's hipfort, with generated rather than hand-written content.
That keeps a clean boundary and simple ownership, and it remains a reasonable option.
With a single bindings super-repository:

- the headers must be copied or fetched from two monorepos, at pinned revisions, before every regeneration;
- the metadata is separated from the header it describes;
- a developer cannot regenerate a binding in the PR that changes its header;
- the binding is not built and tested against the same commit's C library, in the library's CI and in TheRock, and it needs its own packaging.

Co-location gives up the clean boundary for bindings that are built, tested and packaged with their library (P1, P2).

## Appendix: The generator (rocm-fortran-generator)

`rocm-fortran-generator` is a Julia program.
It regenerates the whole binding set from the current C headers.
It parses them with [Clang.jl](https://github.com/JuliaInterop/Clang.jl), the Julia bindings to libclang.
It keeps hipfort's routine, enum, and check-routine names, so existing hipfort calls compile unchanged.
Using the modules needs no Julia.
Only regenerating does.

```text
   Header sources (one of)
     a rocm-systems checkout         (HIP, ROCTx)
     a rocm-libraries checkout       (the math libraries, HIP from an installed ROCm)
     an installed ROCm               (/opt/rocm/include)
            │  parse as C  (clang -x c, via Clang.jl)
            ▼
   Generator (Julia)
     generate_all.jl  →  common/fortran_gen.jl, common/hip_generics.jl,
                         <lib>/generator.jl + <lib>.toml
                         + the library's fortran/metadata/ from the checkout
            │  emit
            ▼
   fortran/  (committed, packaged track: one module per library, no prefix)
     rocm-systems projects/fortran/:  hip.F90, roctx.F90
     rocm-libraries projects/<lib>/fortran/:  rocblas.F90, hipblas.F90, rocsparse.F90,
       hipsparse.F90, rocfft.F90, hipfft.F90 + hipfftw.F90 (both in projects/hipfft/), rocsolver.F90,
       hipsolver.F90, rocrand.F90, hiprand.F90
            →  hip runtime + roctx + 11 libraries, ~5800 functions
```

The engine parses each header as C.
For each library, it emits:

- the interface blocks;
- the enum and `#define` constants;
- the derived types;
- the check routine.

The generator handles each library in a separate process.
Each monorepo's CI runs the generator on that monorepo's checkout only:

- `rocm-systems` with no ROCm installed;
- `rocm-libraries` with the HIP headers of an installed ROCm.

Some headers are generated by a build: the version, config and export headers.
The generator synthesizes them from the checkout's templates.
Any Clang error fails that library's generation; the other libraries still generate.

The profiling binding (`roctx`) sits beside HIP in `rocm-systems`' `projects/fortran/`.
It moves there from `projects/rocprofiler-sdk/fortran/`.
It is built against the installed `rocprofiler-sdk-roctx` and exports `roc::roctx_fortran` from a `roctx-fortran` package.
It is a dozen `bind(C)` interfaces with no array overloads.
So the array option `FORTRAN_ARRAY_INTERFACES` (see [Building the bindings](#building-the-bindings)) changes nothing for it.

Regeneration is cheap.
That is what lets the regeneration workflow rebuild every binding of a monorepo on each run.
It also means a release's final headers can be picked up at the last moment, not weeks ahead.
The generator is also deterministic: the same headers and metadata produce byte-identical output.

The generator is about 5.8k lines of Julia, or 3.7k without comments and blank lines.
On top of that come 1.4k lines of CUDA tooling and 1.7k of tests.
Only the emit stage is Fortran-specific.
The front end parses headers with Clang.jl into a language-agnostic description of functions, enums, and structs.
It could be retargeted to another language by swapping the emitter.

### Free side effect: C-header hygiene

The generator parses every public header as C (`clang -x c`), because a `bind(C)` interface mirrors the C ABI.
So "the headers are valid C" is a hard prerequisite, checked for free on every run.
ROCm headers are meant to be C-includable, but in practice they are only pulled from C++.
C-only breakage therefore hides for releases.
Typical cases are:

- a C++ keyword used as an identifier;
- a missing `extern "C"`;
- a C++-only include.

The generator fails on such a header, and the regeneration workflow reports it.
The bindings team can then follow up with the header's owners.
This also benefits the C libraries themselves.
Several such issues have already been caught this way.

### Ambiguous or unmappable C

Not every valid C construct maps cleanly to a `bind(C)` interface.
Some constructs are ambiguous or have no Fortran equivalent.
For those, the generator falls back to a defined, ABI-correct form rather than dropping them or guessing.

- **Bit-fields, unions, anonymous members.** A struct that is not field-mappable is emitted as an opaque byte blob of the exact C size and alignment.
  A union is narrower: it becomes one field of its enclosing struct, so the struct's other fields stay accessible.
  That field is a blob of the union's size when the union's members differ in type.
  Structs that embed it keep their field offsets, so nothing downstream breaks.
  The cost is that the opaque struct's own members cannot be accessed individually.
  This is how `hipDeviceProp_t` survives, even though it embeds the bit-field struct `hipDeviceArch_t`.
  A struct with an incomplete or zero-size layout is genuinely skipped.

- **Opaque handles.** A pointer to an incomplete struct, such as `rocblas_handle`, is a `type(c_ptr)`, as it was in hipfort.

- **Opaque workspace pointers.** A typed pointer that is really scratch memory is always bound as a `type(c_ptr)` passed by value, never as a Fortran array.

- **C strings.** `char*` is bound as the C API spells it, a `type(c_ptr)`, in every role:

  - an input string;
  - a `const char*` return value, such as the status-to-string helpers and `hipGetErrorString`;
  - a caller-supplied output buffer.

  This is hipfort's surface, so code written against hipfort compiles unchanged.
  A library can opt in to a typed form for its input strings.
  ROCTx does: its routines also accept a `character(kind=c_char)` string, passed without a copy.
  The caller therefore appends the terminating NUL (`"zone"//c_null_char`).

- **Null pointers as a mode selector.** Across rocSPARSE, rocBLAS, and rocSOLVER, `NULL` is sometimes not an error but a mode selector.
  This is the buffer-size query/execute idiom: call once with `nullptr` to get the size, again with the buffer.
  A Fortran array argument cannot be "absent".
  `OPTIONAL` is Fortran's marker for an argument the caller may omit, but it is illegal on the raw surface.
  Fortran 2018 constraint C865 forbids `OPTIONAL` with `VALUE`, and every raw pointer is passed by value.
  So the `<lib>.optional` metadata records which by-reference arguments accept `NULL`.
  For those, the generator adds a plain (non-`bind(C)`) procedure to the same generic, where `OPTIONAL` is legal.
  That procedure passes `c_null_ptr` for an omitted argument, so omitting it from a call passes `NULL` to C.
  The raw specific `X_` stays callable by name.

- **Complex numbers.** A rocBLAS or rocSPARSE complex is a two-float C struct.
  The C type system does not distinguish it from any other two-float struct.
  So mapping it to a Fortran intrinsic `complex` is a curated fact, not something inferred.
  It rests on an ABI assumption: the struct is passed the same way as `complex(c_float_complex)` on the supported platforms.
  That assumption is to be checked by the size-and-alignment gate (see the [backlog appendix](#appendix-implementation-backlog-and-open-risks)).
  The alternative, a `bind(C)` derived type, would break every existing user and the shipped examples that pass a plain `complex`.

- **Macros and vector types.** Function-like and internal (`__`-prefixed) macros are skipped, while `#define` constants are extracted as parameters.
  The low-level HIP vector types (`float4`, `int4`, and the rest) are skipped.
  Functions taking a pointer to one keep `type(c_ptr)`.
  A library that passes one by value lists it to be emitted (rocRAND's `uint4`).

Two ambiguities the C type system cannot resolve on its own are left to the curated metadata (see [Curated metadata](#curated-metadata-the-manual-surface)):

- a scalar host value versus a data buffer (`const double* alpha` versus `double* x`, identical in C);
- device-side output scalars.

Invalid C, as opposed to merely ambiguous C, is a separate case: the generator fails on it (see [C-header hygiene](#free-side-effect-c-header-hygiene)).

### Curated metadata (the manual surface)

The generator emits everything the C ABI carries: signatures, enums, and structs.
A few Fortran-side facts are not present in the C headers and cannot be inferred.
They live in small per-library files that a maintainer curates.
These files sit in a `metadata/` directory next to the library's module:

- `projects/fortran/metadata/` in `rocm-systems`;
- `projects/<lib>/fortran/metadata/` in `rocm-libraries`.

When the generator runs on a checkout, it reads that directory.
So the metadata is curated next to the header it describes, in the same repository (P1).
This is the one exception to "the generator removes the hand-copying".
The bindings are generated, but this thin layer of per-argument intent stays manual, because the C API does not carry it.
Several files concern the *pointer mode*.
The pointer mode is a per-handle setting that says whether scalar arguments such as `alpha` and `beta` are read from host or device memory.

| File               | What it encodes                                                                     | Why the headers cannot give it                                                           |
| ------------------ | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `<lib>.argkinds`   | which pointer args are scalars (`alpha`, `beta`, an output) vs data buffers         | `const double* alpha` and `double* x` are identical in C                                 |
| `<lib>.ptrmode`    | the subset of those scalars the library reads per its pointer mode (host or device) | the pointer mode is runtime handle state                                                 |
| `<lib>.devonly`    | scalars the device writes (hipSOLVER's `devInfo`), which get no host form           | the C type does not say which side writes                                                |
| `<lib>.cptrargs`   | pointers kept as pass-through `type(c_ptr)` in the array overloads (a dot `result`) | pointer-mode outputs live on the device, so binding them as host Fortran arrays is wrong |
| `<lib>.native`     | which functions get the array (`_rank_*`) overloads, and the max rank               | rank is a Fortran-side choice, absent from the ABI                                       |
| `<lib>.optional`   | by-reference arguments the C API accepts as `NULL`                                  | a nullable pointer and a required one are identical in C                                 |
| `<lib>.byvalue`    | arguments passed by value although their C type is a pointer (HIP's `extra`)        | the C type does not say how the API reads it                                             |
| `<lib>.funptrargs` | arguments that are function pointers declared `void*`                               | the C type erased the function type                                                      |
| `<lib>.arrayargs`  | pointers to a derived type that are arrays (hipFFTW's `fftw_iodim`)                 | a pointer to one element and to an array are identical in C                              |

A library carries only the files it needs:

- none for ROCTx;
- one to three for the FFT and RAND libraries;
- four to six for HIP and the BLAS, SOLVER and SPARSE libraries.

The generator configuration is `<lib>.toml`: name prefix, opaque handles, own headers, check overrides.
It is project convention rather than header fact.
So it stays with the generator, which owns the code that reads it.
The CUDA map files are `<lib>.txt`, `<lib>_enums_cuda.txt`, and the per-type `.cuda.finc`.
They serve only `ROCm/hipfort`'s CUDA backend, kept until 11.0.
They stay in the generator's `src/<lib>/metadata/` and are read only for the hipfort layout.
They are never copied into a library's `fortran/metadata/`.
The HIP binding's pull request carries its `metadata/` (ROCTx needs none).
The `rocm-libraries` pull requests add theirs before they leave draft; until then, the generator keeps a copy of each.

The maintenance cost is bounded, because these files were bootstrapped and are tool-validated, not written from scratch.
Every `.argkinds` was derived from the original hipfort sources in one run.
A script derives the `.ptrmode` and `.devonly` files from `.argkinds`, minus a short per-library list of host-only arguments.
The generator's `apply_argkinds` step reports scalar-looking arguments that are still unclassified.

The residual manual work is small, and it is maintenance rather than authoring.
A new ROCm function may have a scalar pointer argument that the bootstrap did not cover.
That argument must be classified by hand, and the report points at it.

**Keeping the classifications correct.**
The `apply_argkinds` report catches an *unclassified* argument, but not a *stale* one.
A stale entry stays well-formed after a function is renamed or its signature changes.
So on each run the generator checks the entries against the parsed C declarations and reports any that match nothing.
Today that check covers every metadata file, function names and parameter names alike, but it only warns.
Before the regeneration workflow is enabled (see [Rollout](#rollout-and-success-criteria)), an unmatched entry **fails generation** instead of being reported.
The bot then never proposes a binding built from a stale entry, on top of the report of *unclassified* arguments.
That leaves one case no such check can catch: an entry that matches but is wrong.
It is covered two ways:

- a one-time audit of the classifications bootstrapped from hipfort;
- runtime tests that exercise **both host-pointer and device-pointer modes** for a representative set of entry points.

Those tests prioritize the pointer-mode-ambiguous families (`alpha`/`beta`/`result`).
A wrong classification there turns into a fault or a silently wrong answer.

### Testing the generator

The generator is tested at three levels:

- its own logic;
- the shape of its output;
- the output at runtime.

Its logic has a unit-test suite in the generator repository.
It exercises the pure transforms, with no ROCm headers:

- C-type to Fortran mapping;
- name and spec helpers;
- the metadata loaders;
- the bind, wrapper, enum, and struct emitters;
- 132-column line wrapping;
- the selection of header sources from checkouts.

Small fixture headers are parsed with the libclang Clang.jl ships.
The suite also includes:

- a byte-for-byte golden test, which asserts the emitters still produce the same output as before;
- `compile_golden.jl`, which runs `gfortran -fsyntax-only` on regenerated fixtures to prove the output is valid Fortran;
- `array_ifaces.jl`, which checks that each `FORTRAN_ARRAY_INTERFACES` setting yields the same API as the guarded module compiled under its macro.

The suite runs in the generator repository's CI on every supported Julia version.

The output is also checked in CI, in two places, and those checks double as regression tests.
The generator repository's CI compiles its own output for its test fixtures (`compile_golden.jl`), and checks the array tiers against each other.
In each monorepo, the CI of the bot's pull request builds the regenerated modules with the project's compilers, in each array configuration.
It also compiles each library's Fortran tests (`fortran/test/`, where present) against them.
This catches interface drift without a GPU.

These gates are compile-only, by design, so they need no GPU.
The bindings themselves are exercised at runtime by those tests on an AMD GPU, as GPU-labeled CTest tests of each library.
That runtime suite must cover both pointer modes, host and device, for the `argkinds`-classified entry points.
A misclassified scalar pointer compiles and links cleanly.
So it is the one correctness property no compile-only gate can see (see [Keeping the classifications correct](#curated-metadata-the-manual-surface)).

Because the transforms are unit-tested, the trust the regeneration workflow needs is largely in place.
The bindings team, reviewing the bot's pull request, relies on a tool whose logic is checked, not only on its output.

### Dependencies

The generator needs Julia, Clang.jl, and the C headers.
It is a standalone, build-time tool that regenerates the committed `.F90`.
It is never part of a consumer build or of the compiler build.
No Julia dependency reaches the compiler, ROCm, or any consumer: using the committed `.F90` needs only a Fortran compiler.
Consumers are unchanged: they still link `libamdhip64` and the relevant math-library `.so`.

The coupling to Clang is deliberately shallow.
The generator parses public headers at the C API surface (`clang -x c`), not Clang's internal AST.
It needs Julia 1.11 or later and Clang.jl 0.19.
It uses the libclang bundled with Clang.jl; no system Clang is involved.

Across repositories, `rocm-systems` and `rocm-libraries` gain a `fortran/` subdirectory and a regeneration workflow that invokes the generator.
TheRock builds both but never runs the generator.

## Appendix: Building, installing and testing the bindings

### Building the bindings

The Fortran sources live in each library's `fortran/` directory.
A dedicated, self-contained `fortran/CMakeLists.txt` builds them.
It does `enable_language(Fortran)`, compiles the modules, and produces the `.mod` and `.a`.
It requires CMake 3.18 for `Fortran_PREPROCESS`, and 3.20 to run its tests with `ctest --test-dir`.
The versions in use already suffice:

- The in-scope `rocm-libraries` C projects declare a minimum CMake version between 3.13 (rocSOLVER) and 3.25.2 (hipSPARSE, and the monorepo root).
  Those are their C floors: a `fortran/` that declares less still needs 3.18.
- TheRock requires 3.25, and its builders run 3.31.
- All of these handle Fortran module dependency ordering and `Fortran_MODULE_DIRECTORY`.
- A build using the Ninja generator wants CMake >= 3.20 for reliable Fortran module dyndep (dynamic dependency scanning).

Every project that carries a binding exposes one uniform switch, `BUILD_FORTRAN_BINDINGS`.
They are the in-scope `rocm-libraries` projects and `rocm-systems`' `projects/fortran/`.
The switch is **ON by default on Linux but guarded**:

- its default is `${UNIX}`, because Windows ships no Fortran compiler (see [Non-goals](#non-goals));
- it builds the bindings when a Fortran compiler is present;
- it silently skips them when one is not, so a C-only site with no Fortran compiler still configures and builds.

The guard reuses a probe the same configure already ran:

- In `rocm-libraries`, it is `ROCM_LIBS_HAVE_FORTRAN` from the root `CMakeLists.txt`.
- In `rocm-systems`, which has no root probe, it is `ROCM_HAVE_FORTRAN` from `shared/cmake/ROCmFortran.cmake`.
  That module was added with the HIP binding, and `projects/fortran/` includes it.
- In a standalone per-project build, it is the project's own `check_language(Fortran)`.
  When neither `CMAKE_Fortran_COMPILER` nor `FC` is set, it prefers the `amdflang` found beside the C/HIP compiler or under `ROCM_PATH`.

In `rocm-libraries`, the project root adds `fortran/`.
In `rocm-systems`, `projects/fortran/` is a project of its own.
`fortran/CMakeLists.txt` declares the options and runs the guard.
It calls `enable_language(Fortran)` only when the switch is ON and a compiler exists.
Otherwise it prints a `STATUS` line and moves on, never a fatal error.
That is how ON-by-default keeps the promise that C developers need no Fortran.
No C build is ever *forced* to acquire a Fortran compiler; it just builds the bindings wherever it already can.
A C developer who has a Fortran compiler but does not want the bindings passes `-DBUILD_FORTRAN_BINDINGS=OFF`.
Each switch also has a per-library form, `<LIB>_BUILD_FORTRAN_BINDINGS` (for example `ROCSPARSE_BUILD_FORTRAN_BINDINGS`).
It overrides the global one, so a single project can differ from the rest of a monorepo build.

Two more options complete the set, identical in every project:

- `FORTRAN_ARRAY_INTERFACES` selects the array tier: which ergonomic array specifics each generic carries besides the raw form.
  Its values are `none`, `assumed-shape` and `assumed-rank`, and its default is `assumed-shape`.
  It replaces hipfort's two booleans `HIPFORT_USE_FPOINTER_INTERFACES` and `HIPFORT_ASSUMED_RANK`.
  See [Open questions](#open-questions-and-risks) for why assumed-rank replaces rather than adds.
  ROCTx has no array overloads, so the option has no effect on it.
- `BUILD_FORTRAN_CLIENTS` builds the binding's own tests in `fortran/test/`.
  Its default is `ON`, and its per-library form is `<LIB>_BUILD_FORTRAN_CLIENTS`.
  rocBLAS, hipBLAS and rocSPARSE already used that name for the Fortran part of `clients/`, and it keeps that meaning.
  Asking for clients with the bindings off skips them, never an error.
  In-tree, the binding tests also follow the project's own test switch: the library's `BUILD_CLIENTS` / `BUILD_CLIENTS_*`, `BUILD_TEST` for rocRAND and hipRAND, or `BUILD_TESTING`.
  hipSPARSE's Fortran samples stay under its existing `HIPSPARSE_ENABLE_FORTRAN`.
  `projects/fortran/` in `rocm-systems` has no other test switch: `BUILD_FORTRAN_CLIENTS` alone gates its tests.

Each `fortran/` is also a self-contained CMake project.
When top-level, it declares `project(<lib>-fortran LANGUAGES NONE)`.
So a binding builds against an installed ROCm without rebuilding the C library.
`rocm-systems`' `projects/fortran/` is only ever built that way.
It finds the installed HIP (`find_package(hip)`) and ROCTx, and no other project adds it.

```bash
cmake -S projects/rocblas/fortran -B build/rocblas-fortran \
  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build/rocblas-fortran
# rocm-systems: HIP (and ROCTx), against the installed runtime
cmake -S projects/fortran -B build/fortran -DCMAKE_PREFIX_PATH=/opt/rocm
```

A build-tree result variable `<LIB>_HAVE_FORTRAN_BINDINGS` (for example `ROCSPARSE_HAVE_FORTRAN_BINDINGS`) is true only when this library's bindings were actually built.
It is deliberately distinct from the root's `ROCM_LIBS_HAVE_FORTRAN`, which only reports that a Fortran compiler exists.
A compiler can be present while a given library has `BUILD_FORTRAN_BINDINGS=OFF`.
TheRock, which always has amdflang, builds every library's binding.

The guard is small, but each `rocm-libraries` project's `fortran/CMakeLists.txt` currently repeats it rather than sharing it.
`rocm-systems` gets `shared/cmake/ROCmFortran.cmake` for the compiler probe with the HIP binding.
In `rocm-libraries`, hipSPARSE, rocSOLVER and hipSOLVER carry the amdflang preference in a `fortran/cmake/ROCmFortranCompiler.cmake`.
The other projects inline it.
Folding the probe and the guard into one helper, in rocm-cmake or a shared module, is implementation backlog.
The logic each `fortran/CMakeLists.txt` runs today is:

```cmake
# default: ${UNIX}, but OFF for a hip* library configured for its CUDA backend
option(BUILD_FORTRAN_BINDINGS "Build the generated Fortran bindings" "${UNIX}")
if(DEFINED <LIB>_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${<LIB>_BUILD_FORTRAN_BINDINGS})
endif()
# reuse the probe the root already ran, otherwise probe locally (amdflang first)
if(DEFINED ROCM_LIBS_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_LIBS_HAVE_FORTRAN})
elseif(DEFINED ROCM_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_HAVE_FORTRAN})
else()
  include(CheckLanguage)
  check_language(Fortran)
  set(_have_fortran ${CMAKE_Fortran_COMPILER})
endif()
if(NOT BUILD_FORTRAN_BINDINGS)
  return()
elseif(NOT _have_fortran)
  message(STATUS "BUILD_FORTRAN_BINDINGS=ON but no Fortran compiler; skipping")
  return()
endif()
enable_language(Fortran)
# ... add_library(<lib>_fortran ...), install, tests ...
# set only once the target exists, e.g. ROCSPARSE_HAVE_FORTRAN_BINDINGS
set(<LIB>_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
```

For discovery, the Fortran target ships as its own per-library config package, `<lib>-fortran`.
A consumer writes `find_package(rocblas-fortran)`, which `find_dependency()`s the C library.
The `<lib>-fortran-config.cmake` filename is reused from rocRAND and hipRAND, which already install it beside their C config.
Today their C config `include()`s it; it is not found on its own.
The config file sits beside the C one (`<libdir>/cmake/rocblas/rocblas-fortran-config.cmake`).
A forwarder in `<libdir>/cmake/rocblas-fortran/` lets `find_package` resolve the hyphenated name.
HIP and ROCTx install theirs directly in `<libdir>/cmake/<lib>-fortran/`.

Keeping it beside the C config is what allows the next step.
Once rocm-cmake grows an `include(<lib>-fortran-config.cmake OPTIONAL)` hook in the generated C config, `find_package(rocblas)` alone will expose `roc::rocblas_fortran`.
It will do so when the binding is installed, and ignore it otherwise.
Until then the `-fortran` package is the entry point, and consumer code written against it keeps working afterwards.
The same is possible for HIP and ROCTx, whose C configs come from clr and rocprofiler-sdk rather than from rocm-cmake.
One optional line in each, run only when the consumer has enabled Fortran, would let `find_package(hip)` expose `hip::hip_fortran`.
That line is `include("${CMAKE_CURRENT_LIST_DIR}/../hip-fortran/hip-fortran-config.cmake" OPTIONAL)`.
It touches those two projects, which the bindings otherwise stay out of.
So it is a possibility to propose to their owners later, not part of this RFC.
Until then, `find_package(hip-fortran)` and `find_package(roctx-fortran)` are the entry points.

Each library's Fortran target sits in its own C library's existing CMake namespace.
The targets include `roc::rocblas_fortran` and `roc::hipsolver_fortran`, but also `hip::hip_fortran`, `hip::hipfft_fortran`, and `hip::hiprand_fortran`.
This is per-library, not a hip-versus-roc rule: hipBLAS, hipSPARSE, and hipSOLVER are `roc::`.
The full list is in Migration and compatibility.
There is no per-library `<lib>::` namespace.
There is also deliberately no umbrella `rocm-fortran::` one, since the bindings have no shared deliverable beyond the per-library ones.

The consumer-facing presence flag keeps the variable name already in the tree, `<pkg>_FORTRAN_FOUND`.
rocRAND and hipRAND set `rocrand_FORTRAN_FOUND` / `hiprand_FORTRAN_FOUND` today, from `find_package(<lib>)`.
The Fortran config package now defines it.
It is `FALSE` when the package is installed but holds nothing for the consumer's compiler.
The config requires Fortran to be enabled before `find_package`.
When no `.mod` matches the consumer's compiler, it sets `<pkg>_FORTRAN_FOUND` to `FALSE` with a `NOT_FOUND_MESSAGE`.
That message names the compilers that are installed and the `<LIB>_FORTRAN_COMPILER_DIR` override.
With it, a user can tell a missing rebuild from a wrong path.
The message is the presence check: it tells a user whether a binding is installed for their compiler, with no separate tool.
Without CMake, the installed compilers are the subdirectories of `<libdir>/fortran/`.
The bindings for one compiler are the archives in `<libdir>/fortran/<compiler>/`.
Each per-library Fortran package is versioned with its C library (`SameMajorVersion`).
So a binding resolves to the same major version as the C library it was generated from.
A static archive carries no soname, so this package version is the version handle.

The `${UNIX}` default generalizes what hipSOLVER already does.
hipSOLVER defaults the switch to `${UNIX}` today, so the default is unchanged there.
The design adds the compiler guard so that default is safe everywhere.
It also redefines the target as the generated binding.
The hand-written `hipsolver_module.f90` is removed under the split.
Its shared `libhipsolver_fortran.so` becomes the static `libhipsolver_fortran.a` under the same `roc::hipsolver_fortran` name.
The design retires hipSOLVER's `EXPORT_FORTRAN_BINDINGS` and `BUILD_FORTRAN_MODULE` too, with the module they gated.
This has one consequence for hipSOLVER users.
`find_package(hipsolver)` exported `roc::hipsolver_fortran` through `EXPORT_FORTRAN_BINDINGS` (ON by default), and no longer does.
A consumer moves to `find_package(hipsolver-fortran)` until the rocm-cmake hook lands.
The reference build of the shipped bindings is TheRock's.
It builds every project, and so every `fortran/`, in dependency order.

Cross-module dependencies are handled the same way the C libraries already handle them.
`rocsolver` uses `rocblas`, because `rocsolver-functions.h` includes `rocblas.h` and the rocSOLVER API takes `rocblas_handle` and rocBLAS enums.
That is the existing C dependency graph, now in Fortran, not a new problem.
TheRock compiles rocBLAS before rocSOLVER.
A standalone per-repo build reuses the `roc::rocblas_fortran` target when both are configured in one tree.
Otherwise it resolves it with `find_package(rocblas-fortran)`, exactly as the C build depends on `librocblas`.
The rocBLAS `.mod` must come from the same compiler, so a non-amdflang site builds the two bindings in that order.
Because each library is a single module under the packaged track, `rocsolver` simply `use`s `rocblas`.
The dependency is kept as-is, matching the C graph.

### Layout after the split

**Source tree (committed).** Each math library owns a `fortran/` directory in its `rocm-libraries` project.
`rocm-systems` has one `projects/fortran/` for HIP and ROCTx.
Each holds the generated modules, their curated metadata, their build files, and their tests.
The generator *code* is never vendored here (P3).
Only the metadata it reads is co-located, in `metadata/` (see [Curated metadata](#curated-metadata-the-manual-surface)).

```text
rocm-systems/  projects/fortran/        a standalone project, built against the installed HIP and ROCTx
   hip.F90                       generated: module hip (runtime + types/enums/check/auxiliary)
   roctx.F90                     generated: module roctx
   metadata/                     curated: hip.argkinds, hip.optional, hip.byvalue, hip.funptrargs
   CMakeLists.txt                self-contained build → libhip_fortran.a + hip.mod, libroctx_fortran.a + roctx.mod
   hip-fortran-config.cmake.in   find_package export → hip::hip_fortran
   roctx-fortran-config.cmake.in find_package export → roc::roctx_fortran
   cmake/toolchains/             example toolchain files (amdflang, gnu, cray, intel, intel-classic, nvhpc)
   test/                         Fortran tests

rocm-libraries/  projects/rocblas/fortran/
   rocblas.F90                   generated: module rocblas (one module)
   metadata/                     curated: rocblas.argkinds, .cptrargs, .native, .ptrmode
   CMakeLists.txt                → librocblas_fortran.a + rocblas.mod
   rocblas-fortran-config.cmake.in
   cmake/toolchains/             example toolchain files
   test/                         Fortran tests (use hip: need the HIP binding)
```

HIP and ROCTx share `projects/fortran/` rather than living under `projects/hip/` and `projects/rocprofiler-sdk/`.
That keeps the bindings out of those projects' builds, and gives the directory the same shape as a library's `fortran/` in `rocm-libraries`.
HIP's binding is in review.
ROCTx's moves there from `projects/rocprofiler-sdk/fortran/`.
hipFFT's `fortran/` holds two bindings and installs two packages from one build:

- `hipfft-fortran`: `hip::hipfft_fortran`, `libhipfft_fortran.a`;
- `hipfftw-fortran`: `hip::hipfftw_fortran`, `libhipfftw_fortran.a`.

Each has its own `HIPFFT_*` / `HIPFFTW_*` overrides of `BUILD_FORTRAN_BINDINGS` and `BUILD_FORTRAN_CLIENTS`, and they share one compiler directory.
Both configs sit in `<libdir>/cmake/hipfft/`, with forwarders in `<libdir>/cmake/hipfft-fortran/` and `<libdir>/cmake/hipfftw-fortran/`.

The CUDA metadata (`<lib>.txt`, `<lib>_enums_cuda.txt`, `*.cuda.finc`) does **not** live here.
The packaged track has no CUDA backend, so those files stay in the generator.
The generator reads them only to generate `ROCm/hipfort`, whose CUDA backend is kept until 11.0.
The CODEOWNERS entry is not a file in `fortran/` either.
It is a line in the monorepo's root `.github/CODEOWNERS`, which both monorepos edit by hand.
In `rocm-libraries`, `/projects/<lib>/fortran/` is owned by the library's review team (for example `@ROCm/fft-reviewers`) and the Fortran-bindings team, `@ROCm/rocm-fortran-maintainers`.
The line goes after the project's other entries, because the last matching pattern wins, and a broader one such as `/projects/rocblas/**/CMakeLists.txt` would otherwise take `fortran/CMakeLists.txt`.
In `rocm-systems`, `/projects/fortran/` is owned by `@ROCm/hip-reviewers`, which reviews HIP's metadata, and the Fortran-bindings team.
A team must have write access to a repository for its CODEOWNERS entry to take effect, so the Fortran-bindings team needs it in both monorepos.

**Install tree.** A `.mod` is compiler- and version-specific (see the [appendix on `.mod` incompatibility](#appendix-fortran-mod-and-a-incompatibility-across-compilers-and-versions)).
So artifacts install under a per-compiler subdirectory, one archive and one module set per library:

```text
<prefix>/
   <libdir>/fortran/<compiler>/        librocblas_fortran.a, libhip_fortran.a, libroctx_fortran.a, ...
   <includedir>/fortran/<compiler>/    rocblas.mod, hip.mod, roctx.mod, ...
   <datadir>/rocblas/fortran/          rocblas.F90          ← the generated source
   <datadir>/hip/fortran/              hip.F90
   <datadir>/roctx/fortran/            roctx.F90
   <libdir>/cmake/rocblas/             rocblas-fortran-config.cmake  → roc::rocblas_fortran
   <libdir>/cmake/rocblas-fortran/     forwarder, so find_package(rocblas-fortran) resolves
   <libdir>/cmake/hip-fortran/         hip-fortran-config.cmake      → hip::hip_fortran
   <libdir>/cmake/roctx-fortran/       roctx-fortran-config.cmake    → roc::roctx_fortran
```

`<libdir>`, `<includedir>` and `<datadir>` are the GNUInstallDirs values (`lib64` on Fedora, RHEL and SUSE).
A stock ROCm install gives `/opt/rocm/include/fortran/amdflang/` and `/opt/rocm/lib/fortran/amdflang/`.
The distro packages install under a versioned prefix, `/opt/rocm/core-<X.Y>`, so ROCm 10.3 installed from packages gives `/opt/rocm/core-10.3/include/fortran/amdflang/` and `/opt/rocm/core-10.3/lib/fortran/amdflang/`.
There is no `hipfort/<backend>/` nesting.
The packaged track has a single backend, so the `.mod` files sit directly under `include/fortran/<compiler>/`.
The `rocm-libraries` configs sit beside their C library's, so that the C config can later pick them up (see [Building the bindings](#building-the-bindings)).
HIP's and ROCTx's come from `projects/fortran/`, a project of their own.
They stay in their own directories rather than inside another project's.

Every file installs in the project's development component, the one that holds the C library's headers and CMake config.
That component is `devel` in `rocm-libraries`, and `dev` for `projects/fortran/`, HIP's name for it.
So the bindings land in the same development package as the library they bind.
`projects/fortran/` installs both HIP and ROCTx in that one component.
So both land in its own development package rather than in HIP's or `rocprofiler-sdk-roctx`'s.
That gives one project, one TheRock artifact and one package.
The artifact can still be split by path if ROCTx ever needs its own package.

`<compiler>` is the name a user types, derived from the compiler ID, so several compilers can share one prefix.
The name is `amdflang`, `gfortran`, `ftn`, `ifx`, `ifort`, `nvfortran`, or else the lowercased basename.
The name ignores the compiler version.
`<LIB>_FORTRAN_COMPILER_DIR` (for example `gfortran-13`) gives a second version of one compiler its own directory.
The per-compiler subdirectory is always used: there is no counterpart to hipfort's `HIPFORT_MULTITOOLCHAIN_LAYOUT=OFF`.
The array tiers produce `.mod` files of the same name that are not interchangeable.
The precompiled amdflang set is the default tier, `assumed-shape`.
A site that wants `none` or `assumed-rank` builds its own set.
It installs that set under its own prefix or its own `<LIB>_FORTRAN_COMPILER_DIR` (for example `amdflang-assumed-rank`).
`find_package(rocblas-fortran)` resolves the subdirectory matching the consuming project's compiler, so CMake users never write these paths.
A non-CMake build passes them as `-I` and `-L`.

**The generated `.F90` installs too, under `share/<lib>/fortran/`** (`share/hipfft/fortran/` holds both `hipfft.F90` and `hipfftw.F90`).
A user on a compiler other than the one ROCm ships bindings for has to build the binding themselves.
Shipping the source means they do not have to clone a monorepo to get it.
They compile it with `-DUSE_ASSUMED_SHAPE=1` (or `-DUSE_ASSUMED_RANK=1`) to get that tier.
Without a macro, the source yields the `none` tier.
Shipping the source also removes a failure mode.
Source obtained separately can be generated from a different ROCm than the installed `.so`.
That surfaces as an undefined symbol at link time or, worse, as an interface that differs silently from the one the caller meant.
Installed source matches the installed library by construction.

The source goes to `share/<lib>/fortran/` rather than `include/`, for three reasons.

- **It is not in the compiler-partitioned tree.**
  Every entry under `include/fortran/` answers "which compiler is this for?".
  The `.F90` is the one artifact that is compiler-*independent*.
  Placing it there is a category error, with a concrete symptom.
  A config that enumerates `include/fortran/*` to report which compilers are installed would list `rocblas.F90` as if it were a compiler.
  The user would be invited to select it.
- **Room to grow.**
  If the Fortran side ever ships more than the module (a second module, an example), `share/<lib>/fortran/` holds them together.
  `include/<lib>/` would scatter them among C headers.
- **Packaging.**
  `include/` is the development-headers namespace that tooling walks.
  A Fortran source in it is noise for anything scanning the include path.
  `share/` is defined as architecture-independent read-only data, which is what a source file is.

This is a deliberate break from where the hand-written modules used to live (`include/rocsparse/rocsparse.f90`, `include/hipblas/hipblas_module.f90`).
Those were headers-by-convention that a user compiled into their own build.
The generated module is a source artifact of the library, and the filenames change in the move anyway.
So there is no drop-in path to preserve.

### Tests and documentation

**Tests.** hipfort's test suite splits across the libraries it covers.
Each library's Fortran tests move to its `fortran/test/`, next to the binding they exercise.
rocBLAS and hipBLAS are not there yet (see [Maintaining the bindings under the split](#maintaining-the-bindings-under-the-split)).
Every math library's `fortran/test/` follows one convention:

- Each test is one Fortran program and its own CTest test.
  A `<name>.f03` test calls the raw and typed forms and is registered as `<lib>_fortran_<name>`.
  A `<name>.f08` test calls the array overloads and is registered as `<lib>_fortran_<name>_f2008`; it is built unless `FORTRAN_ARRAY_INTERFACES` is `none`.
  It is labeled `gpu`, since it needs an AMD GPU.
- Tests are built under `BUILD_FORTRAN_CLIENTS` and, in-tree, under the project's own test switch.
  They are skipped under ASan.
- They link against the binding, the C library and the HIP binding.
  The tests `use hip`, found with `find_package(hip-fortran)`.
  So until the HIP binding is installed, they configure as skipped with a `STATUS` message.
- A test that needs what the host lacks, such as a second GPU or a feature of the device, exits with code 77.
  CTest reports it as skipped once the test sets `SKIP_RETURN_CODE 77`.
  HIP's tests all set it; elsewhere it is set per test.

rocRAND and hipRAND are no exception.
Their former Fortran tests live in `test/fortran/`, built with the FRUIT framework against the deprecated `rocrand_m` / `hiprand_m` wrappers.
They move into `fortran/test/` like everyone else's, ported to the generated module.
The result is hipfort's programs, one per random-number generator and distribution, and the FRUIT suite, kept as one more CTest test of the same kind.
Their one test that touches no device, the library version check, is their only test without the `gpu` label.
It does not `use hip`.

The `rocm-systems` tests in `projects/fortran/test/` differ only where they must.
HIP's link the in-tree `hip_fortran` rather than finding it, and ROCTx's single test needs no GPU.

The bot's pull request compiles them without a GPU; this is what catches interface drift.
The packaging or validation job runs them on an AMD GPU.
None of the projects installs its Fortran test executables yet.
They reach a library's `test` component when it does (see [Packaging in TheRock](#appendix-packaging-in-therock)).
A test that spans libraries, such as a rocSOLVER test that `use`s rocBLAS, follows the same dependency order as the bindings.
It is resolved the same way: build order in TheRock, `find_package` for a standalone build.

**Documentation.** The packaged modules carry no docstrings today.
A Fortran routine has the same name, arguments and semantics as the C function it binds, so the library's C reference documents it.
Each library's documentation should state how the Fortran API maps onto it: module name, generic forms, link target.
The rocSPARSE and hipSPARSE pull requests start doing this in their usage pages.
The headers' Doxygen comments can be emitted as `!>` docstrings, as hipfort's modules carry them.
Turning that on for the packaged modules, and publishing a Fortran reference from them, is not decided yet.

**Do the docs get their own subfolder in `fortran/`?**
No.
Any docstrings live in the `.F90`.
The *doc build* is a whole-library concern, because the C API and the Fortran API belong in one site.
So it stays at the library's existing top-level `docs/`, which would add `fortran/*.F90` to its Doxygen input.
We deliberately do not stand up a second doc-build system inside `fortran/`: co-locate the content (P1), not a parallel toolchain.
So `fortran/` holds `test/`, but no `docs/`.

**Dependency impact on the doc build.** If the packaged modules carry docstrings, cross-references are where library dependencies reach the docs.
A rocSOLVER docstring that mentions `rocblas_handle` resolves only if rocBLAS's Doxygen tag file or Sphinx inventory is available when rocSOLVER's docs are built.
A Doxygen tag file is the index a Doxygen build exports so other projects can link to its symbols.
So the doc build carries the same dependency graph as the code, expressed as tag-file and inventory dependencies rather than `.mod` dependencies.
That is the argument for assembling the site in one aggregating job rather than building each library's Fortran doc in isolation.
That job is the doc job that already has every project's output.
An isolated build cannot resolve a cross-project reference without the other project's tag file.
The remaining mechanics, portal integration and the exact tag-file wiring, need the ROCm documentation team.

## Appendix: Packaging in TheRock

TheRock already builds every project above as a subproject, so it builds their `fortran/` with them.
Nothing about the bindings needs a separate pipeline.
Building them needs a short list of changes in TheRock itself, none of which changes the design.
Part of it is already open as a TheRock pull request that brings flang and OpenMP throughout ROCm, which:

- adds `FORTRAN_OPTIONAL` / `FORTRAN_REQUIRED` to `therock_cmake_subproject_declare`;
- emits `CMAKE_Fortran_COMPILER` for the subprojects that opt in;
- marks rocBLAS, hipBLAS, rocSPARSE, rocSOLVER, hipSOLVER, rocRAND and hipRAND `FORTRAN_OPTIONAL`;
- builds flang as its own `amd-llvm-flang` subproject, with `flang-rt` from the `amd-llvm-offload` runtime layer;
- drops the hipSOLVER `.so` entries.

The bindings build on that pull request.
The remaining in-scope `rocm-libraries` subprojects (hipSPARSE, rocFFT, hipFFT) opt in the same way.
`rocm-systems`' `projects/fortran/` becomes one more subproject, Linux-only and `FORTRAN_REQUIRED`.
It comes after `hip-clr` and `rocprofiler-sdk`, whose installed HIP and ROCTx it builds against.
The items below are written against that mechanism rather than a second one.

1. **One Fortran compiler for every subproject: the `amdflang` TheRock builds.**
   TheRock builds flang on Linux, and `base/aux-overlay` exposes it as `bin/amdflang`.
   Today flang is built in `compiler/amd-llvm`.
   The flang pull request moves it to `compiler/amd-llvm-flang`, with `flang-rt` from `amd-llvm-offload`.
   TheRock generates a toolchain file for each subproject (`cmake/therock_subproject.cmake`).
   It sets `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` and, for `amd-hip`, `CMAKE_HIP_COMPILER`.
   But it sets no `CMAKE_Fortran_COMPILER`.
   Each project's probe then picks a compiler on its own.
   The `fortran/` probe looks for `amdflang` beside the C compiler first.
   But a project root that probes earlier with a plain `check_language(Fortran)` takes whatever is on `PATH`.
   On a TheRock builder, with the CMake 3.31 it pins, that is `gfortran`, which the environment check requires.
   Newer CMake already prefers the flang beside the C compiler.
   The fix is to emit `CMAKE_Fortran_COMPILER` in that generated file for every subproject that opts in.
   Its value is the toolchain's flang, which `bin/amdflang` points to.
   The flang pull request makes this fix.
   Every subproject then compiles its binding with the compiler that ships.
   So the shipped `.mod` files match the shipped `amdflang` exactly, version included.
   It also keeps the dependent bindings consistent.
   On `develop`, rocSOLVER and hipSOLVER default `CMAKE_Fortran_COMPILER` to `gfortran` before `project()`.
   hipBLAS does so in its clients project.
   The co-location pull requests narrow that default to builds of the test, benchmark and sample clients, and prefer `amdflang` otherwise.
   But TheRock builds those clients when testing is on.
   Without this change, it would build `rocsolver.mod` with gfortran against an `amdflang`-built `rocblas.mod` that gfortran cannot read.
   The rocSOLVER binding would then be skipped.
   A compiler set by the toolchain overrides that default.
   The projects keep gfortran for their clients on purpose.
   The clients link a host LAPACK, and a gfortran-built LAPACK needs the gfortran runtime.
   This does not apply in TheRock.
   Its host LAPACK is OpenBLAS built with `C_LAPACK=ON` (`third-party/host-blas`), which needs no Fortran runtime.
   So forcing `amdflang` is safe for the clients too.
1. **Artifact descriptors.**
   The default `dev` component already collects `**/*.a`, `**/include/**` and `**/cmake/**`.
   So the `.mod` files, the archives and the config packages land in `dev` with no change.
   From there they reach the library's existing development package (see [Building and shipping a binding](#building-and-shipping-a-binding)).
   Six edits remain:
   - Declare the `projects/fortran/` subproject and give it an artifact descriptor.
     Its `dev` component holds the HIP binding's `.mod` files, archive, config package and `share/hip/fortran/`.
     It excludes the ROCTx files, which go to the rocprofiler-sdk artifact (next item).
     Its `test` component holds the subproject's tests, once they are installed.
   - In `profiler/artifact-rocprofiler-sdk.toml`, add a `dev` component over the `projects/fortran/` stage directory that takes only the ROCTx files: `roctx.mod`, `libroctx_fortran.a`, `cmake/roctx-fortran/**` and `share/roctx/fortran/**`.
     The descriptor already gathers several subprojects (rocprofiler-sdk, roctracer), and its `dev` component already reaches `amdrocm-profiler-base`, so the ROCTx binding ships with the ROCTx library with no `package.json` change.
     The rocprofiler-sdk artifact adds the `projects/fortran/` subproject to its `SUBPROJECT_DEPS` (in `profiler/CMakeLists.txt`), and the two descriptors' patterns stay complementary, so no file lands in both artifacts.
   - Add `share/<lib>/fortran/**` to the `dev` component of each library.
     For HIP and ROCTx, the two items above already cover it.
     This is needed because `share/` is not a default `dev` path, and the installed `.F90` is what users of other compilers build from.
     HIP's existing `share/hip/**` rule sits in the `lib` component of `core/artifact-core-hip.toml`.
     The Fortran files come from their own subproject, so they do not fall under it.
   - Fix the existing Fortran entries in `math-libs/BLAS`.
     hipSOLVER's `lib/libhipsolver_fortran.so*` (in `artifact-solver.toml`) goes away once the hipSOLVER binding lands.
     The binding becomes a static archive.
     The `-DEXPORT_FORTRAN_BINDINGS` passed to hipSOLVER retires with that option (next item).
     hipBLAS's `lib/libhipblas_fortran.so` entries (in `artifact-blas.toml`) already match nothing, so they are removed.
     They match nothing because the clients' Fortran helper is a static library that is not installed.
     `libhipblas_fortran.a` is now the binding, under `lib/fortran/<compiler>/`.
   - Revert what the flang pull request adds for the modules the split removes:
     - the `-DBUILD_FORTRAN_WRAPPER` passed to rocRAND and hipRAND, with the `rocrand/src/fortran/**` and `hiprand/src/fortran/**` entries of `artifact-rand.toml`.
       The rocRAND and hipRAND bindings remove that option.
     - the `-DEXPORT_FORTRAN_BINDINGS` passed to hipSOLVER.
   - In `build_tools/packaging/linux/package.json`, add the `dev` component of the `projects/fortran/` artifact to `amdrocm-runtime-devel`.
     Enable `amdrocm-fortran`, make it per-architecture (`Gfxarch`), and add to its dependencies the seven packages that carry a binding (`amdrocm-runtime-devel`, the five math `-devel` packages, and `amdrocm-profiler-base`).
     The math libraries' bindings need no packaging change, since they are already in their library's `dev` component.
1. **Dependency order.**
   This order is already expressed: rocSOLVER depends on rocBLAS, and every library depends on `hip-clr`.
   So the Fortran build order follows the C one.
   The `rocm-libraries` binding tests `use hip`, so they need the HIP binding installed first.
   `projects/fortran/` therefore builds after `hip-clr` and before the math libraries that test against it.
1. **Tests.**
   Some libraries have binding tests, in `fortran/test/` under `BUILD_FORTRAN_CLIENTS`.
   These are CTest tests labeled `gpu`.
   They join each library's existing `test` component and test runner once the project installs them.
   None of the co-location pull requests installs them yet.
   rocRAND and hipRAND used to install their Fortran wrapper test.
   So the tests reach TheRock's test artifacts only once each project installs them.
1. **Fortran runtime.**
   An `amdflang`-built test executable links flang's runtime (`flang-rt`) rather than `libgfortran`.
   The test artifacts depend on the `amd-llvm-offload` runtime libraries that provide it.
   A user's application gets the runtime from the compiler that links it.

Windows builds keep `BUILD_FORTRAN_BINDINGS` off (see [Non-goals](#non-goals)).
The ASan and coverage variants will pass `-DBUILD_FORTRAN_BINDINGS=OFF` and skip the `projects/fortran/` subproject.
The projects themselves build the bindings under ASan and only skip their tests.
So it is TheRock that keeps Fortran out of those variants.
Today TheRock compiles Fortran uninstrumented under ASan (`cmake/therock_sanitizers.cmake`).
Instrumenting it is a draft pull request.
Once these land, TheRock's nightly builds are the first place a user gets the bindings precompiled, ahead of a ROCm release.
Until then, a user tries a binding by building its `fortran/` from the pull request against a nightly install.

## Appendix: hipfort's build artifacts

This appendix describes how hipfort installs today.
It also explains why its artifacts take the shape the packaged track keeps.

After `make install`, under `$CMAKE_INSTALL_PREFIX`:

```text
├── lib/fortran/<compiler> (HIPFORT_MULTITOOLCHAIN_LAYOUT=ON)
│   ├── libhipfort-amdgcn.a              ← AMD backend  (default)
│   └── libhipfort-nvptx.a               ← NVIDIA backend (deprecated; compat track only)
├── lib/cmake/hipfort/                   ← find_package(hipfort): hipfort::rocblas ...
├── include/fortran/<compiler>/hipfort/{amdgcn,nvptx}/*.mod ← read on `use hipfort...`
```

The install layout has a toolchain dimension.
A `.mod` is the compiled interface file that a Fortran `use` statement reads.
It is specific to the compiler that produced it, and even to that compiler's version.
Two Fortran compilers cannot share one `.mod`.
`HIPFORT_MULTITOOLCHAIN_LAYOUT` selects the layout.
When ON (the default), each compiler gets its own `<compiler>` subdirectory, for example `lib/fortran/<compiler>/`.
Several toolchains then coexist under one prefix, as in the diagram above.
When OFF, the `.a` goes directly under `lib/` and the `.mod` tree under `include/hipfort/`.
A second compiler's install would then conflict.
The backend (`amdgcn` for AMD, `nvptx` for NVIDIA) is a second, independent axis.
This is because compiling for CUDA also produces a different `.mod` (see [One source set, two backends](#appendix-hipforts-build-artifacts)).
`find_package(hipfort)` resolves to the subdirectory matching the consuming project's compiler and the AMD backend.
A presence check reports which pieces are installed for the user's compiler.
A user can thus confirm the Fortran extension is available before building.
Under the split, this becomes a per-library check with per-library targets (`find_package(rocblas-fortran)`, `roc::rocblas_fortran`); see [Building the bindings](#building-the-bindings).

The root cause is that Fortran has no portable ABI: `.mod` files and compiled `.a` archives are specific to the compiler and its version.
The only stable ABI is C's, exposed by the vendor `.so`, and the `bind(C)` layer calls through it.
That constraint drives the whole packaging design: ship source, plus a precompiled set for `amdflang`.
The appendix on `.mod`/`.a` incompatibility sets it out in full.

Both `.mod` and `.a` are needed, for different phases.

```text
   lib/hipfort/*.F90
         │  compile
         ▼
   ┌────────────────────────────┬────────────────────────────────┐
   │  *.mod  (declarations)     │   *.o  (only for real bodies)  │
   │  interfaces, enums, types  │   generic wrappers             │
   └─────────────┬──────────────┴───────────────┬────────────────┘
   COMPILE phase │                              │ ar → archive
   your_app.F90 ─┘ reads *.mod          libhipfort-amdgcn.a
         │                                      │
         ▼ compile                              ▼ LINK phase
   your_app.o ── unresolved symbols ──► libhipfort-amdgcn.a  (rocblas_dgemm_rank_0, rocblas_dgemm_full_rank, ...)
                                       ► librocblas.so       (rocblas_dgemm)
                                       ► libamdhip64.so      (HIP runtime)
```

The `.mod` is a compile-time artifact.
It carries declarations.
The `.a` is a link-time artifact; it carries compiled bodies.
A math-library call has two forms.
The raw `bind(C)` interface (`rocblas_dgemm_`) is a pure declaration.
It sits in the `.mod`, adds nothing to the `.a`, and its symbol is resolved from the vendor `.so`.
The ergonomic variants (`rocblas_dgemm_rank_1`, `rocblas_dgemm_full_rank`, ...) take Fortran arrays and are real wrappers.
They call `c_loc`, which returns the C address of a Fortran variable, and forward to the raw form.
So their bodies are compiled into the `.a`.
These wrappers are numerous: about 2,500 for rocBLAS alone in the default tier.
They sit alongside the HIP runtime wrappers (`hipMalloc`, `hipMemcpy`, `hipCheck`).
So the `.a` is not just the memory wrappers: every math library with ergonomic variants contributes bodies to it.
Both the `.mod` and the `.a` must ship.

**Why a static archive, not a shared library.**
The `.a` holds thin forwarding shims: each wrapper does `c_loc` and calls the raw `bind(C)` form.
So a shared library buys nothing:

- there is nothing substantial to share in memory;
- the caller is bound to the module at compile time through the `.mod`, so changing a wrapper means recompiling anyway;
- a prebuilt `.so` is compiler- and version-locked exactly like a `.mod`.

A `.so` would only add a runtime dependency, and a PLT indirection that blocks inlining a one-line shim.

**Why not fuse with the vendor library.**
We keep the Fortran `.a` separate from the vendor `.so` rather than fusing them.
Fusing would tie the binding's version to the vendor release and make that `.so` carry Fortran symbols.
It would gain nothing: with amdflang, the wrapper object is under 2% of `librocblas.so`.
A separate static `.a` also keeps unused wrappers out of the binary, but only at the granularity of its members.
The linker extracts whole object files from an archive, not single symbols.
Each library's module compiles to one object, so a program that calls one rocBLAS wrapper links all of them.
A shared library that links the archive also re-exports them all.
Compiling the archive with `-ffunction-sections -fdata-sections` puts each wrapper in its own section.
A program linked with `-Wl,--gc-sections` then keeps only the wrappers it calls.
A shared library that links the archive exports every wrapper, and the linker never removes an exported symbol.
Such a library therefore also links with `-Wl,--exclude-libs,librocblas_fortran.a:libhip_fortran.a`.
The archives are separated by colons, because `-Wl,` splits on commas.
The wrappers still work but are no longer exported (see the [backlog appendix](#appendix-implementation-backlog-and-open-risks)).
This also decides whether a program links at all when the binding and the vendor library do not come from the same headers.
That can happen in either direction:

- a binding newer than the library (generated from development headers, or from a later 10.x) has wrappers for routines the installed `.so` does not export yet;
- a binding older than the library has wrappers for routines the library has since removed or renamed, such as entry points that were marked deprecated.

Either way the archive member holds an undefined symbol.
Without per-procedure sections, every program that uses the module then fails to link, whether or not it calls that routine.
With them and `-Wl,--gc-sections`, only a program that calls that routine fails, which is the failure it should get.
A binding and its library built together, as in a ROCm release, never hit this.
Mixing releases, which the `SameMajorVersion` promise invites, does.
There is no C-side clash either, since the C and Fortran objects compile in different directories.

**One source set, two backends.**
The same sources compile twice under the CUDA switch (`-DUSE_CUDA_NAMES`).
This produces a per-backend archive and `.mod`: `libhipfort-amdgcn.a` for AMD, `libhipfort-nvptx.a` for NVIDIA.
A compiled `.mod`/`.o` is backend-specific, so the two cannot share one archive.
The packaged track keeps only the AMD backend.

## Appendix: How NVIDIA does it (baseline)

The closest analog is NVIDIA's Fortran story for its GPU math libraries, and the contrast is instructive.

NVIDIA ships precompiled Fortran modules for cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuTENSOR, and more.
They ship inside the HPC SDK, next to `nvfortran`.
You write `use cublas` and link with `-cudalib=cublas`.
The decisive difference is that NVIDIA owns the only compiler its users use.
With a single supported compiler the "`.mod` incompatible across compilers" problem simply does not exist for them, so shipping prebuilt `.mod` is fine.
We cannot copy that: our users are on gfortran, amdflang, Cray CCE, ifx and others.
So we ship the source for every compiler, plus a precompiled set for `amdflang`.

A few more contrasts, point by point:

- **Location.** NVIDIA's interfaces live in the compiler/SDK, not co-located in each library's source repo.
  Ours live next to the C headers in `rocm-systems` and `rocm-libraries`, and are regenerated from them by each monorepo's own workflow.
- **Naming.** No prefix: `use cublas`, not `nvfort_cublas`.
  The module name is the library name.
- **Backward compatibility.** They keep the legacy and the v2 calling conventions side by side (`cublas` and `cublas_v2`).
- **Array handling.** They do not use assumed-rank.
  They use the CUDA Fortran `device` attribute.
  `nvfortran` distinguishes host from device arrays, and the interfaces declare device arrays, so host/device dispatch is resolved by the language.
  gfortran and flang do not know `device`.
  So our `bind(C)` plus `c_loc` plus generated variants is the portable substitute for what `nvfortran` does natively.
- **Scalar host-vs-device (alpha/beta).** Because `nvfortran` knows host from device, its non-v2 modules set and restore the library pointer mode implicitly, at a small overhead.
  We cannot rely on the compiler for this, so `.argkinds` records which scalar pointers are host values.
  One caveat static metadata cannot remove: the host-versus-device choice for `alpha`/`beta` is also *runtime* handle state (`rocblas_set_pointer_mode`).
  So the ergonomic host-scalar form is only correct when the handle is in host-pointer mode.
  The wrappers therefore must not silently set or restore that mode.
  It is shared, mutable handle state, so touching it would race other threads.
  The raw `type(c_ptr)` surface stays the escape hatch for device-pointer mode.
- **Tooling.** Their tooling is the compiler driver: `-cudalib` plus an automatic module search path.
  There is no separate installation checker because the bindings are bundled with the compiler.
  Ours are not bundled with a single compiler, which is why explicit packaging and a presence check are worth having.

The through-line: most of the hard questions here exist for us precisely because we do not own the one compiler our users run.
These questions are cross-compiler `.mod`, ABI across compilers, where the bindings live, and generation.
NVIDIA answers them by monoculture; we answer them by generation, source shipping, and co-location.

## Appendix: Fortran `.mod` and `.a` incompatibility across compilers and versions

This is the constraint that shapes the whole packaging design, so it is set out here in full.

Unlike C, Fortran has no standard ABI and no standard module format.
Two artifacts are affected.

- **The `.mod` file.** A `.mod` is a compiler-private serialization of a module's interface (names, kinds, generic-resolution tables).
  Its format is undocumented and specific to the compiler.
  It is also **not stable across versions of the same compiler**.
  gfortran, for instance, changes its module format between some releases (format 15 since GCC 8) and promises no compatibility.
  So a `.mod` written by one gfortran can be rejected by another.
  There is no interoperability across compilers either: a gfortran `.mod` is meaningless to ifx, amdflang, or Cray, and vice versa.
- **The compiled `.a`.** The wrapper bodies compile to objects whose symbols are **name-mangled** by a compiler-specific scheme.
  Examples are trailing underscores and module-name decoration such as `__rocblas_MOD_`.
  So the archive is tied to the same compiler, and version, as the `.mod` it accompanies.

So there is exactly one portable thing to ship: **the Fortran source**.
A prebuilt `.mod`/`.a` set is usable only by the exact compiler, at essentially the exact version, that produced it.

One relaxation exists on the compiler we ship.
amdflang has an opt-in `-fmodule-mismatch-check=warn` (since ROCm 10.1) that downgrades the module-version check to a warning.
A newer amdflang can then read a `.mod` written by an older one.
That softens the *version* axis for the precompiled amdflang set.
That set is the one we ship, and this is the skew our users are most likely to meet: a ROCm-shipped `.mod` against a slightly different amdflang.
It does nothing for the *compiler* axis, which is the one that forces us to ship source: no flag makes a gfortran `.mod` meaningful to amdflang.

This is why the design:

- ships **source** so any site can compile with its own compiler, and additionally ships a **precompiled set for the default compiler (`amdflang`)** for the common case;
- keys the install tree by **compiler** under `lib/fortran/<compiler>/` and `include/fortran/<compiler>/` (a site keeping two versions of one compiler sets `<LIB>_FORTRAN_COMPILER_DIR`, for example `gfortran-13`);
- treats the only stable ABI as the **C one**, the vendor `.so`, which the `bind(C)` layer rides on and which every compiler agrees about.

NVIDIA escapes all of this by owning the single compiler (`nvfortran`) its users run, which is why it can ship prebuilt `.mod` and we cannot (see [How NVIDIA does it](#appendix-how-nvidia-does-it-baseline)).

## Appendix: Implementation backlog and open risks

The direction is set.
The items below are implementation and CI work that does not change the design.
They are recorded here so they are not lost.
None blocks the co-location pull requests.
The few that are prerequisites of a later stage are named in Rollout: the regeneration workflow and the first packaged release.

**CI and reproducibility.**

- Pin what the generator resolves:

  - its revision;
  - the Julia and Clang.jl versions;
  - the include roots;
  - the macro profile;
  - the C standard level it parses at.

  A regeneration is then reproducible.
  A toolchain change reaches `hip`, `roctx` and the library modules only when someone moves the pin.

- Some public headers do not parse from a source checkout alone.
  The generator already synthesizes the version, config and export headers from their templates.
  But feature-gate headers (`*.h.in`) can still change the visible API without touching a tracked `.h`.
  Define the generator's input as a named, configured artifact per library.
  Or let the regeneration workflow configure or partly build the tree for the libraries whose headers need it.

- The regeneration workflow reads only the default branch and executes no pull-request code.
  But it needs a token that can push a branch and open a pull request.
  Give it a dedicated bot identity: a GitHub App rather than the workflow's `GITHUB_TOKEN`.
  A pull request opened with `GITHUB_TOKEN` does not trigger the CI that must check it.
  Allow the App to push only `bot/fortran/*` branches and to open pull requests.
  Run the workflow on a trusted runner.

- Keep the workflow cheap.
  Cache the Julia depot (`~/.julia`, about 2 GB with libclang) between runs.
  The bot's pull requests touch only `fortran/`, so let them run the project's Fortran jobs rather than its full GPU matrix.

- Report a failed regeneration where someone will act on it.
  A header that is valid C++ but not valid C fails that library's generation.
  The workflow then opens or updates an issue assigned to the header's CODEOWNERS, with the bindings team copied.
  The other libraries still regenerate.

**Release integrity.**

- Resolve every `bind(C, name=)` label in the generated sources against the symbol tables of the libraries actually shipped in a release.
  Fail on any unresolved name, so a feature-gated API cannot ship as a Fortran interface with no code behind it.
  Linking the built archive with `-Wl,--whole-archive -Wl,--no-undefined` does the same for the wrapper bodies in the `.a`.
  This matters more because each archive is one object.
  A wrapper that references a missing C symbol breaks the link of every program that uses the archive, not only of those that call it.
- Add the size-and-alignment gate.
  It is a CI check that compares each emitted struct's size and alignment against the C compiler's `sizeof`/`_Alignof` on the target.
  The gate is needed because those ABI facts are frozen into a committed artifact.
  It also covers the complex-number mapping, which maps a two-float C struct to a Fortran `complex` on an ABI assumption.

**Type mapping.**

- Integer scalars map to named `iso_c_binding` kinds, not host-measured ones.
  Document two assumptions:
  - A site that builds with `-i8` (8-byte default `INTEGER`) must give integer literals an explicit kind (`1_c_int`).
  - The `iso_c_binding` kinds describe the C compiler paired with the Fortran compiler.
    That need not be the compiler that built the `.so`.
- Check the complex mapping under `ROCM_MATHLIBS_API_USE_HIP_COMPLEX`.
  It makes the math libraries' complex types `hipFloatComplex` / `hipDoubleComplex`.
  The generator parses the default spelling, so both must have the layout of `complex(c_float_complex)` / `complex(c_double_complex)`.
- Fix the rule for adding a specific to an existing generic.
  The new specific must be distinguishable from every existing one.
  Adding a form then never makes a call ambiguous or changes which specific it resolves to.
  The generator checks it.
- Function-pointer and callback types map to `c_funptr`, with an emitted `abstract interface` where the signature is mappable.
  A callback that must be a device function pointer is not emitted as if usable.
  A generated comment marks it unsupported-from-Fortran instead.
- This item applies to the compatibility track only.
  The packaged track has no NVIDIA backend.
  A few handle types differ across platforms.
  For example, `hipfftHandle` is a 32-bit int on NVIDIA but a pointer on AMD.
  Either record a per-backend type substitution, or drop that library from the nvptx build.

**Cross-library and metadata consistency.**

- The opaque-handle list, the "own headers" marker, and the name prefix are global decisions, not per-library ones.
  For example, rocSOLVER's whole API takes `rocblas_handle`.
  Promote them to one repository-level config.
  Gate on two modules claiming the same declaration, or on a module referencing a type no module claims.
- Make the metadata check fail generation instead of warning.
  It already covers every metadata file and parameter names.
  The bot then never proposes a binding built from an entry left behind by a renamed parameter (see [Keeping the classifications correct](#curated-metadata-the-manual-surface)).
- Assert the 63-character Fortran identifier limit for generic and binding names.
  The longest bound C symbol, `hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`, has 59 characters.
  Specific names past 63 characters are already cut, with a hash tag when two cuts collide.
  No generic or binding name may need a cut.

**Packaging and build variants.**

- The ASan build variant ships no Fortran, since Fortran is not yet instrumented under ROCm's ASan configuration.
  The presence check reports that distinctly rather than as "not installed for your compiler".
  The same holds for the coverage variant.
- Each `lib<lib>_fortran.a` is built with `-ffunction-sections -fdata-sections`, as in hipfort `develop`, for the reasons given under [Why not fuse with the vendor library](#appendix-hipforts-build-artifacts).
  The twelve pull requests pass these flags only to the compilers that split sections (GNU, LLVMFlang, Flang and NVHPC), and link their tests with `-Wl,--gc-sections`.
  They spell it `-Wl,--gc-sections` rather than `LINKER:--gc-sections`, which CMake 3.27 passes to `amdflang` as a bare `--gc-sections`, failing the link.
  The migration guide documents the link flags; once a release packages these archives, it no longer needs to say that the archives ROCm ships are not split.

**Consumer experience and policy.**

- Offer a non-CMake consumer path beyond the documented command lines, since not every HPC site builds with CMake.
  It could be a `pkg-config` file or a small flags script.
  The migration guide already gives one worked command line per compiler.
  The installed `.F90` can also be compiled by hand with `-DUSE_ASSUMED_SHAPE=1`, the preprocessor macro behind `FORTRAN_ARRAY_INTERFACES=assumed-shape`.

- Decide whether to ship a supported migration helper that rewrites `use hipfort_*` lines to the packaged module names.
  The migration guide documents the `sed` recipe such a tool would automate:

  - delete the folded modules;
  - rename the rest;
  - leave `cuda_errors` alone.

  So the open question is not effort.
  It is the support surface, and the residue the recipe leaves to the user:

  - fixed-form sources;
  - continuation lines;
  - `use hipfort_check` in a file that never used `hipfort`.

- Share the Fortran probe and guard across projects, instead of the copy each `rocm-libraries` `fortran/CMakeLists.txt` carries today.
  This could be one helper in rocm-cmake, or a `shared/cmake` module in each monorepo.
  `rocm-systems` already has `shared/cmake/ROCmFortran.cmake`.
  Also settle one name for the root's "a Fortran compiler exists" flag.
  It is `ROCM_HAVE_FORTRAN` in `rocm-systems` and `ROCM_LIBS_HAVE_FORTRAN` in `rocm-libraries`.

- Record the compiler and its version in the installed config package.
  A consumer whose compiler version differs from the one that built the `.mod` then gets a clear message from `find_package`, rather than the compiler's module-format error.

- Add the `include(<lib>-fortran-config.cmake OPTIONAL)` hook to rocm-cmake's generated config, so that `find_package(<lib>)` alone exposes the Fortran target.

- State a default-inclusion policy.
  Also state who answers support tickets for a public Fortran API shipped under a library's name.
  Bug reports arrive against rocBLAS, while the bindings team owns `fortran/`.
