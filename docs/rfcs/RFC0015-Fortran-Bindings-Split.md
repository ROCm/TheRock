---
author: Alexis Montoison (amontoison)
created: 2026-07-20
modified: 2026-11-12
status: draft
discussion: https://github.com/ROCm/TheRock/pull/8122
---

# Distributing Fortran bindings across rocm-systems and rocm-libraries

## Summary

A Fortran *binding* is the set of Fortran declarations that lets a Fortran program call a C library, through `bind(C)`, Fortran's standard way to call a C function.
ROCm's bindings live today in `ROCm/hipfort`, a separate repository downstream of the C headers they describe.
Until ROCm 7.14, part of hipfort was written by hand, so it drifted from the headers and stalled whenever no one maintained it.
This RFC moves the bindings into the repositories of those headers: HIP and ROCTx to `rocm-systems`' `bindings/fortran/hip/` and `bindings/fortran/roctx/`, and each math library to `rocm-libraries`' `bindings/fortran/<lib>/`.
A binding is a consumer of its library's public API, so it is a standalone project, built against the installed library and never wired into the library's build.

A standalone tool, `rocm-fortran-generator`, generates the bindings from the C headers, and the generated files are committed.
Regeneration workflows regenerate them daily from the default branch, and a bot opens a pull request when a binding has changed (see [Keeping the bindings in sync](#keeping-the-bindings-in-sync)).
What a header cannot say, such as whether `double* alpha` is one scalar or an array, on the host or on the device, lives in a `metadata/` directory in each binding.

Each library ships:

- one module, named after the library (`use rocblas`, `use hip`); hipFFT adds `hipfftxt` for its Xt and Mp API, and hipFFTW, a separate C library built from the same project, has its own `hipfftw` module
- a static archive, `lib<lib>_fortran.a`
- its `.mod` files under `include/fortran/<compiler>/`
- the generated source under `share/<lib>/fortran/`
- a CMake package, `<lib>-fortran`, exporting `roc::rocblas_fortran`, `hip::hip_fortran`, and so on

A `.mod` file can only be read by the compiler version that wrote it, so ROCm ships the bindings precompiled for its own compiler, `amdflang`, and users of other compilers build them from the installed source.
TheRock takes the bindings in stages: build support, off by default, then a CI stage that builds them after the libraries they wrap and runs their tests on a GPU, then release builds and packaging.
Once TheRock's CI builds them, the bindings' test jobs are to be required, like every component's tests, and they pass as long as the C API only grows.
The proposed packaging is one Fortran development package per library family, `amdrocm-<family>-fortran-devel`, for the product and release owners to approve.

The change comes in two steps.
Through ROCm 10.2, hipfort is only kept current with the headers: that is done for 7.14, 10.0 and 10.1 (hipfort 0.8.0 to 0.10.0), and the 10.2 regeneration is merged and awaits its release.
From 10.3, the bindings split across `rocm-systems` and `rocm-libraries`: a tech preview is targeted for 10.3, and production, with the OS packages and the full documentation, for 10.4.
This RFC calls these generated bindings the *packaged track*, and hipfort the *compatibility track*, which keeps its current name and layout through the 10.x series and is removed no earlier than ROCm 11.0.
The hand-written modules some libraries ship (`rocblas_module.f90`, `rocsparse.f90`, ...) are deprecated at the 10.3 tech preview and removed no earlier than 11.0.
For hipfort users, moving over means renaming their `use` lines and link targets, and their calls do not change.
The ROCm documentation gets one Fortran entry point, under *ROCm HPC SDK*, and the migration guides move there from the hipfort documentation before hipfort is removed (see [Roadmap and rollout](#roadmap-and-rollout)).

## Decisions requested

Each row below is a proposal of this RFC until its owners agree, and the linked sections give the details.

| Decision                                                  | Proposal                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Who decides                                                                  |
| :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| D1. Deprecating the hand-written modules                  | Deprecated at the 10.3 tech preview, removed no earlier than ROCm 11.0. ROCm's packages ship only these modules' sources (rocBLAS, hipBLAS, rocSPARSE) and rocRAND's and hipRAND's legacy CMake configs, so removal is a source-level change. hipSOLVER's `EXPORT_FORTRAN_BINDINGS` default turns off at deprecation. See [What changes for a library repository](#what-changes-for-a-library-repository).                                                  | Each library's CODEOWNERS                                                    |
| D2. Release gate                                          | A release-branch regeneration reports any diff to the release owners, who decide whether it blocks the release. See [Keeping the bindings in sync](#keeping-the-bindings-in-sync).                                                                                                                                                                                                                                                                          | Release owners                                                               |
| D3. Packaging                                             | One `amdrocm-<family>-fortran-devel` package per library family, pulled in by RFC0009's `amdrocm-fortran` metapackage; nothing is packaged before approval. See the [Packaging in TheRock](#appendix-packaging-in-therock) appendix.                                                                                                                                                                                                                        | Product and release owners                                                   |
| D4. Review of bot pull requests                           | The Fortran-bindings team aims to review and merge each bot pull request within five business days; as a fallback, the library's CODEOWNERS can merge a bot pull request that has waited ten business days. See [Keeping the bindings in sync](#keeping-the-bindings-in-sync).                                                                                                                                                                              | Bindings team, library leads                                                 |
| D5. Write access and approvals                            | The bindings team gets write access to both monorepos, without which its CODEOWNERS entries have no effect. With both teams on one CODEOWNERS line for the whole binding, `metadata/` included, one approval from either team suffices. See [Open questions and risks](#open-questions-and-risks).                                                                                                                                                          | Monorepo administrators, library leads                                       |
| D6. Compatibility promise                                 | From the 10.4 production release, code that compiles and links against a 10.4 binding keeps compiling and linking against any later 10.x binding, unless the C library itself changes the bound function. The promise relies on one section per procedure and `-Wl,--gc-sections`, which an OpenMP offload program linked with `amdflang` pairs with `-Wl,-z,nostart-stop-gc` for now. See [Version targeting and roadmap](#version-targeting-and-roadmap). | Product and release owners, library leads                                    |
| D7. Required binding test jobs in TheRock                 | Required, like every component's tests; they pass as long as the C API only grows, and regenerating a binding is never a condition for merging. See [What this means for a library maintainer](#what-this-means-for-a-library-maintainer).                                                                                                                                                                                                                  | Proposed by the author, for TheRock's and the library maintainers' agreement |
| D8. Regeneration workflow before the tech preview         | Pin the HIP header source used by `rocm-libraries`, open an issue when a header fails, list the changes that need a decision (removed names, changed signatures, renamed or unclassified arguments) in the bot's pull request, and run on release branches. See [Rollout and success criteria](#rollout-and-success-criteria).                                                                                                                              | Bindings team, monorepo maintainers                                          |
| D9. A generic TheRock stage on per-architecture artifacts | The bindings' stage declares the math libraries as pre-built subprojects with no AMDGPU target, the first generic stage to depend on artifacts built per architecture. See [Open questions and risks](#open-questions-and-risks).                                                                                                                                                                                                                           | TheRock maintainers                                                          |

## What this means for a library maintainer

This section is for the maintainers of HIP, ROCTx and the ten `rocm-libraries` projects with a binding: rocBLAS, hipBLAS, rocSPARSE, hipSPARSE, rocSOLVER, hipSOLVER, rocFFT, hipFFT (which also carries hipFFTW, its FFTW-compatible interface), rocRAND and hipRAND.

Your library gains a binding in its monorepo's `bindings/fortran/` tree (`bindings/fortran/<lib>/` in `rocm-libraries`), with the generated source, its metadata, its CMake build and its tests.
It sits outside your project, and your project's build does not change.
Your team and the bindings team own it together through CODEOWNERS.
Your part is mostly reviewing the metadata, which records what only the library knows, such as whether a pointer argument lives on the host or the device.

You never write binding code, and you never have to run the generator.
The regeneration workflow already runs in your monorepo, and once your library's binding is merged, its daily run regenerates the binding after your PR merges and a bot opens a pull request, which the bindings team reviews.
Until the binding is merged, the bindings team regenerates it by hand in the binding's pull request.
You can regenerate the binding in your own PR if you want to, but nothing requires it.
On a PR that changes your public headers or the binding's metadata, the regeneration workflow checks that the headers still give a binding that compiles, and that check is informative, not required.

Once TheRock's CI builds the bindings, their test jobs run on your PR when it changes your library.
The proposal (D7) is that these jobs are then required, like every component's tests in TheRock.
They test the committed binding against the library the pull request builds, so they pass as long as the C API only grows, which is the rule within a major release.
A new function is simply not bound until the bot regenerates the binding, and that regeneration is never a condition for merging.
A binding test fails only when a change breaks the C API or ABI: within a major release that is a bug the test catches, and at a major release, the pull request that breaks the API regenerates the binding or asks the bindings team to do so.

One thing does change for you.
Your public headers should stay valid C, because the generator parses them as C (libclang with `-x c`).
If a header only compiles as C++, that library's regeneration fails, the header check on your PR, which is informative, reports it, and you fix it in a normal follow-up PR.
Nothing is reverted, and no C pull request is blocked.

If your library ships a hand-written Fortran module today, it is deprecated at the 10.3 tech preview and removed no earlier than ROCm 11.0.
The bindings team opens the deprecation and removal PRs and writes their CHANGELOG entries, and you review them.
[What changes for a library repository](#what-changes-for-a-library-repository) has the details.

## Background

hipfort is a thin Fortran-to-C translation layer, with no GPU code of its own.
Its modules declare `interface ... bind(C)` blocks against the ROCm C functions, and add convenience wrappers on top, which this RFC calls the *ergonomic* forms: for example, a form that takes a Fortran array instead of a raw pointer.

The packaged track keeps the shape of hipfort's build artifacts: a `.mod` set per compiler and a static archive, separate from the vendor library: a `.mod` file is tied to the compiler that wrote it, and a static archive adds no shared library to install or version.
It changes the rest: the separate repository, the `hipfort_` prefix, the split modules and the CUDA backend.
[Roadmap and rollout](#roadmap-and-rollout) says when.

```text
   ┌───────────────────────────────────────────────────────┐
   │  Your Fortran application                             │
   ├───────────────────────────────────────────────────────┤
   │  hipfort: Fortran bind(C) interfaces (*.mod)          │
   │           + generic runtime wrappers (libhipfort*.a)  │
   ├───────────────────────────────────────────────────────┤
   │  ROCm math libraries (.so): librocblas, libhipfft, …  │
   │  HIP runtime         (.so): libamdhip64               │
   ├───────────────────────────────────────────────────────┤
   │  AMD GPU driver / hardware                            │
   └───────────────────────────────────────────────────────┘
```

The sources (`lib/hipfort/*.F90`) are one interface module and one enum module per library, plus a few base modules:

| File                                                                            | Role                                 |
| :------------------------------------------------------------------------------ | :----------------------------------- |
| `hipfort.F90`                                                                   | HIP runtime (device, stream, launch) |
| `hipfort_types.F90`, `hipfort_enums.F90`                                        | HIP types and enumerations           |
| `hipfort_check.F90`                                                             | return-code checking helpers         |
| `hipfort_hipmalloc.F90`, `hipfort_hipmemcpy.F90`, `hipfort_hiphostregister.F90` | generic memory wrappers              |
| `hipfort_{roc,hip}{blas,fft,solver,sparse,rand}.F90`                            | one module per math library          |
| `hipfort_<lib>_enums.F90`                                                       | enumerations of each math library    |
| `hipfort_auxiliary.F90`                                                         | device-property helpers              |
| `hipfort_*_types.F90` (rocRAND, hipFFTW, hipFFT Xt)                             | library derived types                |
| `hipfort_roctx.F90`, `hipfort_cuda_errors.f90`                                  | profiling, CUDA-backend errors       |

Hand-written bindings lag because a new function, enum or struct field in a header has to be copied by hand, later, by someone other than the header's author.
The result is delay and silent drift, such as a missing `_64` variant or a stale struct layout, and when there is no maintainer the copying stops entirely.

The hand-written work is also duplicated, because several math libraries ship their own Fortran module in their source tree:

- rocBLAS ships `rocblas_module.f90`, and hipBLAS ships `hipblas_module.f90`.
- rocSPARSE ships `rocsparse.f90` plus `rocsparse_enums.f90`.
- hipSPARSE ships `hipsparse.f90` plus `hipsparse_enums.f90`, and hipSOLVER ships `hipsolver_module.f90`.
- rocRAND and hipRAND ship a third variant, gated by `BUILD_FORTRAN_WRAPPER`: the deprecated `rocrand_m` and `hiprand_m` wrappers, with a small `hipfor` HIP module.

Five of these define a module named after the library: `rocblas`, `hipblas`, `rocsparse`, `hipsparse` and `hipsolver`.
rocBLAS, hipBLAS and rocSPARSE install theirs as a public source file, while hipSPARSE compiles its module only into its own samples.
hipSOLVER builds its module into `libhipsolver_fortran.so`, but installs the source only under `BUILD_FORTRAN_MODULE`, an option its CMake never declares, so by default the source is not installed.
ROCm's packages therefore ship four hand-written sources (`include/rocblas/rocblas_module.f90`, `include/hipblas/hipblas_module.f90`, `include/rocsparse/rocsparse.f90` and `include/rocsparse/rocsparse_enums.f90`) and rocRAND's and hipRAND's legacy `<lib>-fortran-config.cmake`.
They ship no compiled hand-written module (no `.mod`, archive or shared library), since hipSOLVER, the only library that compiles its module, is built by TheRock with `EXPORT_FORTRAN_BINDINGS=OFF`.
The packaged track's modules have the same unprefixed names, so a program uses one or the other, and the generated ones supersede them (see [What changes for a library repository](#what-changes-for-a-library-repository)).

Each of these modules is maintained independently of hipfort's binding for the same library.
`rocblas_dgemm` thus has two Fortran interfaces, in two repositories, kept current by two different people, and either copy can drift from the C header and from the other.
Generating the binding from the header, in the header's repository, leaves one source of truth per library.

rocSPARSE goes further.
It also ships fourteen Fortran example programs (`clients/samples/example_fortran_*.f90`) written against its own module, yet its documentation mentions neither them nor the module: its one Fortran section points users to hipfort.
So the same library carries a second Fortran binding and a second example set, maintained apart from hipfort and undocumented.
Generating the binding from the library's headers, with the examples moving beside it when the hand-written module is removed, brings that work back to one place.

## Design principles

**P1. Keep each binding in the repository of its headers.**
The binding for `rocblas_dgemm` should live in the same repository as `rocblas.h`, in that repository's `bindings/fortran/` tree.
It is then regenerated from that repository's own headers, with no copy step, and built and tested against the same commit's C library.
Proximity shortens the lag: when regeneration succeeds, the bot's pull request follows the header on the workflow's next run, or the header's author regenerates the binding in the same PR.

**P2. A binding is an API consumer, built apart from its library.**
A binding depends only on its library's public API, which a reorganization of the library's implementation does not change.
So each binding is its own CMake project, outside the library's project, and builds against the installed library: nothing about it is wired into the library's build, and the library's implementation can be restructured without touching it.
Each binding ships its own `.F90`, `.mod`, `lib<lib>_fortran.a` and CMake target (`roc::rocblas_fortran`, `hip::hip_fortran`, ...).
TheRock builds the bindings, declared in its own `bindings/fortran/`, in a build stage after the libraries they wrap, and the proposed packaging groups them by library family (see the [Packaging in TheRock](#appendix-packaging-in-therock) appendix).

**P3. Keep the generator standalone.**
The generator is build-time tooling, not a runtime dependency, and it is not part of the compiler build.
It lives in its own public repository, [`ROCm/rocm-fortran-generator`](https://github.com/ROCm/rocm-fortran-generator), so that the monorepos' regeneration workflows can run it (see [Rollout and success criteria](#rollout-and-success-criteria)).
The bindings team owns and maintains it, not the compiler developers.
Library repositories run it as a standalone tool and do not vendor it.
This RFC commits to one thing: the generator's *output* is checked in, as reviewable Fortran, in the library repositories.

**P4. A bot, not hand-copying, keeps the bindings current.**
A C-library developer should not have to learn Fortran, or run the generator, to keep a binding current.
Each binding's regeneration workflow reads its headers and regenerates it, and when the binding changed, a bot opens a pull request with the regenerated `.F90`.
Developers may still regenerate a binding in their own PR, but nothing requires it: the header check on a header PR is informative, and the bindings' test jobs, required once TheRock's CI builds them, pass as long as the C API only grows.
The one manual part is the curated metadata, in each binding's `metadata/`, and most changes need none.
When a new function needs a classification, the bindings team adds the one line, in the bot's pull request or in its own.

## Non-goals

The following are out of scope for this RFC.

- **Windows.** The target is Linux HPC, and a Windows Fortran product is not a goal.
  ROCm's existing Fortran clients are already disabled on Windows: rocBLAS, hipBLAS, hipSOLVER and rocSPARSE all build Fortran only under `NOT WIN32` or `UNIX`.
- **Fortran under sanitizers.** The ASan build variants ship no Fortran, because ROCm's sanitizer configuration cannot instrument Fortran yet.
  A `rocm-libraries` binding built standalone under ASan still builds, but skips its tests.
  TheRock's sanitizer builds declare no binding, so the bindings' artifacts are empty there (see the [Packaging in TheRock](#appendix-packaging-in-therock) appendix).
- **A binding for compilers limited to Fortran 95 or older.** Fortran 77 code built with a current compiler can use the bindings.
  Only a compiler, or a compiler mode, limited to Fortran 95 or older cannot.
  [Fortran 77 code](#fortran-77-code), under [Language standard](#language-standard), explains what that means and what such a binding would take.
- **The generator as a runtime or shipped product.** `rocm-fortran-generator` is written in Julia and uses Clang.jl to parse the headers.
  None of its code reaches a consumer, the compiler build or the deliverable.
- **Non-CMake consumer tooling as a first-class path.** CMake's `find_package` is the supported integration.
  A `pkg-config` file or a flags script would be a convenience, not a guarantee.
- **A new build system or test framework.** This RFC defines where the bindings live and how they stay in sync, and reuses the monorepos' existing CMake, CODEOWNERS and CI, with no parallel ones.

## Proposal

```text
   BEFORE  (separate repository, always downstream)
     rocm-systems (HIP headers) and rocm-libraries (rocBLAS, ...)
            │  headers copied downstream
            ▼
     hipfort  (separate repo, updated by hand, stalls when unmaintained)
            │
            ▼
     packaging

   AFTER  (in the headers' repositories, apart from the libraries)
     rocm-systems/    projects/hip/*.h, projects/rocprofiler-sdk/.../roctx.h
                      bindings/fortran/          hip.F90, roctx.F90, metadata/
     rocm-libraries/  projects/rocblas/*.h   +   bindings/fortran/rocblas/rocblas.F90, metadata/
                      projects/hipfft/*.h    +   bindings/fortran/hipfft/hipfft.F90, metadata/
            │  the regeneration workflow regenerates each .F90 (a bot PR per library),
            │  or the header's author does it in the same PR
            ▼
     each binding's own build, against the installed library
            <lib>.F90  →  <lib>.mod + lib<lib>_fortran.a
                       →  <lib>-fortran package (roc::rocblas_fortran, hip::hip_fortran, ...)
            ▼
     TheRock: builds every binding from bindings/fortran/ after the libraries,
              with amdflang, off by default and enabled in stages (P2)
```

Where each piece goes:

| Piece                                                                   | Repo                                        | Why                                              |
| :---------------------------------------------------------------------- | :------------------------------------------ | :----------------------------------------------- |
| HIP runtime module `hip` (absorbing HIP types, enums, check, auxiliary) | `rocm-systems`, `bindings/fortran/`         | HIP lives here (P1)                              |
| ROCTx module `roctx`                                                    | `rocm-systems`, `bindings/fortran/`         | ROCTx lives here (P1)                            |
| Each math library's `<lib>` module (`rocblas`, `hipfft`, ...)           | `rocm-libraries`, `bindings/fortran/<lib>/` | same repository as its C headers (P1)            |
| Curated metadata                                                        | the binding's `metadata/`                   | manual, curated beside the binding (P1)          |
| Generator configuration                                                 | the generator                               | project convention, read by the generator's code |
| The generator (`rocm-fortran-generator`)                                | its own repository                          | build-time Fortran tooling (P3)                  |
| Build (per-library `.mod`/`lib<lib>_fortran.a`, CMake package)          | each binding's `CMakeLists.txt`             | standalone API consumer (P2)                     |
| Build and packaging in ROCm                                             | TheRock, `bindings/fortran/`                | own build stage, enabled in stages (P2)          |

### Keeping the bindings in sync

Each binding has a regeneration workflow in its monorepo, which keeps it in sync with its headers, apart from the developers' pull requests.
The workflows are merged in both `rocm-systems` and `rocm-libraries`, and until a binding's directory exists, its workflow runs only the header check.
Each workflow works as follows:

```text
   Developer opens a PR editing  projects/rocblas/.../rocblas.h
            │   (may regenerate rocblas.F90 in the same PR: optional)
            ▼
   Header check on the PR, informative (not required):
     regenerates the binding with the pinned generator and compiles
     its modules with gfortran, on a CPU-only runner with no ROCm install
            │
            ├─ binding unchanged         → passes
            ├─ binding changed           → passes, with a notice
            └─ binding no longer builds  → fails, without blocking the PR
            ▼
   PR reviewed and merged as usual
            │
            ▼
   Daily (or manual) run, on the default branch:
     regenerates the binding from the current headers
            │
            ├─ no change                 → nothing to do
            │
            ├─ binding changed           → the monorepo's GitHub App opens (or updates)
            │                                one PR for the library with its regenerated
            │                                .F90, which the bindings team reviews
            │
            └─ regeneration fails        → the run fails, and the binding stays
               (e.g. a header not valid C)   at its last good state
```

The workflow is *out-of-band* for the bindings: it never pushes to or comments on a developer's pull request.
On a pull request that changes a library's public headers or its metadata, it runs the header check above.
The check fails when the headers no longer give a binding that compiles, and reports a changed binding as a notice.
It is informative, and no branch protection requires it.

Once TheRock's CI builds the bindings, their test jobs are to be required, and they pass as long as the C API only grows (see [What this means for a library maintainer](#what-this-means-for-a-library-maintainer)).
The tests link with `-Wl,--gc-sections`, so with a compiler that puts each procedure in its own section, only a test that calls a removed function fails.

A developer who changes a header has nothing else to do, since the bot regenerates the binding after the PR merges.
The developer may also regenerate the binding in the same PR, with the generator's `--overwrite` option (it needs Julia, as the [generator appendix](#appendix-the-generator-rocm-fortran-generator) explains).
That PR then touches `bindings/fortran/`, so the bindings team has to review it.

The bot runs daily on the default branch, and can also be run by hand.
Each binding's workflow regenerates it from the current headers, including the headers it depends on (rocBLAS's for rocSOLVER, rocRAND's for hipRAND, hipblas-common's for hipBLAS), so it also catches a change to a shared header or a new generator pin.
The result is one pull request per library project (hipFFT's covers hipFFT, its Xt API and hipFFTW), opened by the monorepo's existing GitHub App on the branch `fortran-bindings/regenerate-<lib>`, updated on each run and closed when it has nothing left to change.
When the regeneration succeeds, the bot's pull request reflects a header change within a day, and the binding lags its header until that pull request merges.
Since each library has its own pull request, a blocked library never delays the others.
The bindings team, which co-owns each binding through CODEOWNERS, aims to review and merge these pull requests within five business days.
As a fallback, the library's CODEOWNERS can merge a bot pull request that has waited ten business days (see [Decisions requested](#decisions-requested)).
Once TheRock's CI builds the bindings, its binding jobs run on these pull requests as on any other, and the bindings team merges one only when they pass, and fixes one that fails.
Each pull request is to list the changes that need a decision: a public Fortran name removed, an existing signature changed, an argument renamed, an argument not yet classified.
Today its description only names the generator commit and links the run, so that list is one of the requirements before the tech preview (see [Rollout and success criteria](#rollout-and-success-criteria)).

The generator commit (`GENERATOR_REF`) and the Julia version are pinned in each regeneration workflow, and the runner installs Julia and the generator's packages from the network.
A pull request moves the generator pin.
For `rocm-libraries`, the HIP headers come from the default branch of `rocm-systems`, and pinning that source is also a requirement before the tech preview.
The bot's output is the reference.
The generator records its inputs in a provenance file, and the bot's next run overwrites what a manual run produced.

A new function may need metadata.
Until it has some, an argument binds as its C type says (a typed pointer binds as a data buffer), and the generator warns about the arguments that look like scalars but are not classified.
The bindings team classifies them in the bot's pull request, or in its own before the release, so a scalar bound as an array never ships.
The workflows run the generator in strict mode (`ROCM_FORTRAN_STRICT=1`), so a metadata entry that no longer applies fails the header check and the daily run until the metadata is fixed.
When a header is not valid C, only that library's regeneration fails: the header check of the pull request that broke it fails, without blocking it, and so does the daily run.
The binding stays at its last good state until the header is fixed.
Opening an issue for the header's CODEOWNERS on such a failure is not automated yet, and is a requirement before the tech preview.

Before each release branch is cut, the bindings team merges a fresh bot pull request.
The bot does not run on release branches yet; running it there, so that a header change cherry-picked to a release branch is regenerated too, is a requirement before the tech preview.
A release-branch regeneration reports any diff to the release owners, who decide whether it blocks the release (see [Decisions requested](#decisions-requested)).

### Maintaining the bindings under the split

A binding is more than its `.F90`, and the split keeps all of it in the library's repository.

The generated `.F90` is derived from the header.
The regeneration workflow regenerates it, the bot proposes the result, and no one edits it by hand (P4).

The curated metadata is the manual part: the intent the C API cannot express (see [Curated metadata](#curated-metadata-the-manual-surface), in the [generator appendix](#appendix-the-generator-rocm-fortran-generator)).
It lives in the binding's `metadata/`, in the repository of the header it describes, and the generator reads it when it runs on the checkout (P1).
Most header changes need none.
The regeneration reports a scalar pointer argument that is still unclassified, for the bindings team to classify.
It cannot prove that a classification naming a real function and parameter is right.
The cross-check against the headers' Doxygen comments flags some wrong ones, and the runtime tests and the CODEOWNERS review of the metadata catch the rest (see the *Keeping the classifications correct* paragraph of [Curated metadata](#curated-metadata-the-manual-surface), in the [generator appendix](#appendix-the-generator-rocm-fortran-generator)).

A binding's Fortran tests live in its `test/` and follow one convention (see [Tests and documentation](#tests-and-documentation), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).
Every binding has a `test/`, rocBLAS's and hipBLAS's included, mostly ported from hipfort.
Those two matter most, because, until the hand-written modules are removed, their libraries' `clients/` exercise those modules and not the bindings.
The header check compiles only the modules, with gfortran and without a GPU.
TheRock's `fortran-bindings` stage builds each binding and its tests with `amdflang`, and its test jobs run them on an AMD GPU, once TheRock's CI builds the bindings.
The math-library binding tests `use hip`, so they run only once the `rocm-systems` commit TheRock pins contains the HIP binding.

The libraries' existing Fortran examples (the rocBLAS, hipBLAS, rocSPARSE and hipSPARSE samples) move to the generated module when the hand-written ones are removed.
The packaged modules carry no docstrings yet (see [Tests and documentation](#tests-and-documentation), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).
The current hipfort documentation is split up.
Each library's documentation gets its hipfort tutorial and states how its Fortran API maps onto the C reference.
hipfort's install and how-to pages become the shared Fortran page under *ROCm HPC SDK*.
hipfort's API reference is not carried over.

### Building and shipping a binding

Each binding is a standalone CMake project, `bindings/fortran/<lib>/` in `rocm-libraries` and `bindings/fortran/hip/` and `bindings/fortran/roctx/` in `rocm-systems`, built against the installed library.
No library build adds it.
TheRock builds the bindings, declared in its own `bindings/fortran/`, in a build stage after the libraries they wrap, with the `amdflang` it builds, in the C dependency order (rocBLAS's binding before rocSOLVER's).
It enables them in stages: build support, off by default, then CI, then releases and packaging (see the [Packaging in TheRock](#appendix-packaging-in-therock) appendix).
Until the releases step, the nightly and release tarballs do not carry the bindings, and a user builds a binding from its monorepo against a nightly install.

For each compiler, a binding installs:

- a `.mod` set under `include/fortran/<compiler>/`
- a static archive under `lib/fortran/<compiler>/` (`librocblas_fortran.a`, `libhip_fortran.a`, ...)
- its generated `.F90` under `share/<lib>/fortran/` (hipFFTW's under `share/hipfft/fortran/`)
- a CMake package, `<lib>-fortran`, exporting a target in the C library's namespace (`roc::rocblas_fortran`, `hip::hip_fortran`, ...), and `roc::roctx_fortran` for ROCTx, whose C target is in `rocprofiler-sdk-roctx::`

The deliverable is per library: `find_package(rocblas-fortran)` gives the Fortran binding and, through it, the C library.
The build appendix has the details: the compiler guard, the CMake package, the install tree and the test convention.

With the compilers that put each procedure in its own section, a program linked with `-Wl,--gc-sections` (which the CMake package does not add) needs only the C functions it calls (see [Building the bindings](#building-the-bindings)).
With `amdflang` and `ld.lld`, an OpenMP offload program linked that way also needs `-Wl,-z,nostart-stop-gc` until the toolchain is fixed (see [Open questions and risks](#open-questions-and-risks)).

The packaging below is a proposal, for the product and release owners to approve (see [Decisions requested](#decisions-requested)), and nothing is packaged before they do.
It follows the package naming of [RFC0009](RFC0009-OS-Packaging-Requirements.md), which also defines `amdrocm-fortran` as the package of the Fortran compiler and related components.
In the distro packages, each library family's bindings would ship in a Fortran development package of their own, which depends on that family's C development package.
The packages follow the existing `amdrocm-*` stems and are named with the `-devel` suffix in RPM and `-dev` in DEB, which TheRock's packaging keys on:

| Package                         | Bindings                | Depends on                                           |
| :------------------------------ | :---------------------- | :--------------------------------------------------- |
| `amdrocm-runtime-fortran-devel` | HIP, ROCTx              | `amdrocm-runtime-devel`, `amdrocm-profiler-base`     |
| `amdrocm-blas-fortran-devel`    | rocBLAS, hipBLAS        | `amdrocm-blas-devel`                                 |
| `amdrocm-solver-fortran-devel`  | rocSOLVER, hipSOLVER    | `amdrocm-solver-devel`, `amdrocm-blas-fortran-devel` |
| `amdrocm-sparse-fortran-devel`  | rocSPARSE, hipSPARSE    | `amdrocm-sparse-devel`                               |
| `amdrocm-fft-fortran-devel`     | rocFFT, hipFFT, hipFFTW | `amdrocm-fft-devel`                                  |
| `amdrocm-rand-fortran-devel`    | rocRAND, hipRAND        | `amdrocm-rand-devel`                                 |

Each package is one TheRock artifact, so no file has to be split by path between packages, and a C-only install carries no Fortran files.
The packages carry the `.mod` files of `amdflang`, so they can depend on the compiler's package once the Fortran compiler is packaged on its own, a dependency the C development packages never need.
Users of another compiler build the binding from the installed `.F90` sources.
No package needs a Fortran runtime: a program links the static archive and its compiler's Fortran runtime, which comes with that compiler (`amdflang` links the static `flang-rt` into the program, and a gfortran-built program loads `libgfortran` at run time).

For a Fortran user who wants one command, the `amdrocm-fortran` metapackage already declared in TheRock's packaging (today disabled, and depending only on `amdrocm-llvm`) would gain a dependency on the six packages above.
It would then install the compiler and every binding.
`amdrocm-fortran` would be declared per-architecture (`Gfxarch`), like `amdrocm-core-devel`, because in single-architecture packaging the packages it depends on are per-architecture.
The opt-in HPC SDK development metapackage, `amdrocm-hpc-sdk`, may depend on `amdrocm-fortran`, which is for the HPC SDK owners to decide.
hipfort is not packaged.
Built from source, it can be installed side by side with the packaged bindings, because their module names differ, until its removal, no earlier than 11.0.

Every binding takes the same options.
`<LIB>` in the table below is the project's prefix: `ROCSPARSE_`, `HIP_`, `ROCTX_`, and so on.
hipFFT's build takes `HIPFFT_`, and a consumer of `hipfftw-fortran` can override its directory with `HIPFFTW_FORTRAN_COMPILER_DIR`.
Every routine has a raw form that takes C pointers (see [Calling styles](#calling-styles)).
The *array tier*, selected by `FORTRAN_ARRAY_INTERFACES`, adds forms that take Fortran arrays: one per array rank (`assumed-shape`, Fortran 2008, the default), a single form for any rank (`assumed-rank`, Fortran 2018), or none.

| Option                       | Default              | Effect                                                                                 |
| :--------------------------- | :------------------- | :------------------------------------------------------------------------------------- |
| `BUILD_FORTRAN_BINDINGS`     | `${UNIX}`            | build the binding; without a Fortran compiler, it is skipped with a `STATUS` message   |
| `FORTRAN_ARRAY_INTERFACES`   | `assumed-shape`      | array overloads: `none`, `assumed-shape` (Fortran 2008), `assumed-rank` (Fortran 2018) |
| `BUILD_FORTRAN_CLIENTS`      | `ON`                 | build and register the binding's `test/`                                               |
| `<LIB>_FORTRAN_COMPILER_DIR` | from the compiler ID | the `<compiler>` install subdirectory, to keep a second version of one compiler apart  |

A few bindings add an option of their own:

- rocSPARSE's takes `ROCSPARSE_ILP64` (default `OFF`), which declares that the installed rocSPARSE is ILP64 (64-bit `rocsparse_int`), since an installed rocSPARSE does not record it; when it is `ON`, the binding builds nothing
- `rocm-systems`' `bindings/fortran/` takes its install component from `HIP_FORTRAN_INSTALL_COMPONENT` (default `dev`), while the `rocm-libraries` bindings install in `devel`, and every binding installs its tests in `tests`
- `rocm-systems`' `bindings/fortran/` also takes `HIP_FORTRAN_GENCO_FLAGS`, for the GPU code objects some HIP tests build

To build a binding with its tests and run them, standalone against an installed ROCm:

```bash
cmake -S bindings/fortran/rocsparse -B build/rocsparse-fortran -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build/rocsparse-fortran
ctest --test-dir build/rocsparse-fortran
```

The tests of a `rocm-libraries` binding need the HIP binding installed (`find_package(hip-fortran)`): put its prefix on `CMAKE_PREFIX_PATH`.
Without it, a binding's whole test suite configures as skipped with a `STATUS` message, except rocRAND's and hipRAND's version tests.
Likewise, without the rocBLAS binding, the rocSOLVER binding is skipped with a `STATUS` message.

## What changes for a library repository

[What this means for a library maintainer](#what-this-means-for-a-library-maintainer) already covers the binding in `bindings/fortran/`, its CODEOWNERS entry, the regeneration workflow (see [Keeping the bindings in sync](#keeping-the-bindings-in-sync)) and the Fortran jobs on a PR.
The bindings team maintains the curated metadata and the library's team reviews it.
The library's own project and build are untouched.
What remains are the hand-written modules and what their retirement means for their users.

Any Fortran module the library ships today, such as `rocblas_module.f90`, is superseded by the generated binding, in two steps.
Once the binding ships, at the tech preview targeted for 10.3, the module is deprecated: its library's CHANGELOG says so and points to the binding, and building or installing it reports the deprecation.
hipSOLVER's, rocRAND's and hipRAND's CHANGELOGs already deprecate their modules in favor of hipfort, and their new entries point to the binding instead.
The module is removed no earlier than ROCm 11.0, in a release announced in that CHANGELOG (see [Decisions requested](#decisions-requested)).
No ABI is at stake: ROCm's packages ship only the modules' sources and rocRAND's and hipRAND's legacy CMake configs, no compiled module (see [Background](#background)), so the removal is a source-level change.
The bindings team opens both pull requests and writes their CHANGELOG entries, and the library's maintainers review them.
The removal covers:

- the module under `library/` and, for rocSPARSE and hipSPARSE, its `<lib>_enums` companion, with no compatibility shim: `use rocsparse_enums` becomes `use rocsparse`
- for rocRAND and hipRAND, `BUILD_FORTRAN_WRAPPER`, the wrappers it built and their legacy `<lib>-fortran-config.cmake`
- for hipSOLVER, the `BUILD_FORTRAN_BINDINGS` and `EXPORT_FORTRAN_BINDINGS` options and the `BUILD_FORTRAN_MODULE` install guard

The libraries' Fortran `clients/` (the samples and the gtest Fortran shims) keep using the hand-written modules while they are deprecated, and move with their removal (see [Open questions and risks](#open-questions-and-risks)).

hipSOLVER needs care during the deprecation.
Its build exports the hand-written module's shared library as `roc::hipsolver_fortran` by default (`EXPORT_FORTRAN_BINDINGS=ON`), the same name as the generated binding's target, so a project that finds both packages gets two definitions of one target.
TheRock already builds hipSOLVER with `EXPORT_FORTRAN_BINDINGS=OFF`, so ROCm's packages never export that target, and turning the default off at the deprecation affects only standalone hipSOLVER builds.
Against an installed hipSOLVER that still exports the target, the generated binding fails its configure with an error that says to build hipSOLVER with `EXPORT_FORTRAN_BINDINGS=OFF`.

The generated module follows the C API more closely than a hand-written one, so a few call sites change for its users:

- output scalars are typed and passed by reference, instead of through `c_loc`
- enum arguments use the enum's integer kind
- handle creation takes the handle variable itself

The last change needs care: the old `rocblas_create_handle(c_loc(handle))` still compiles, but leaves the handle unset.
Each binding's documentation and the migration guides (see [Migration and compatibility](#migration-and-compatibility)) list these changes.

## The generated bindings

The generated modules are what a Fortran program `use`s and links.
The generator that produces them has its own [appendix](#appendix-the-generator-rocm-fortran-generator).

### Naming and module structure

The compatibility track keeps today's names and layout, the `hipfort_` prefix and separate interface and enum modules per library (plus a type module for rocRAND, hipFFTW and hipFFT's Xt API), so `use hipfort_rocblas` keeps working.
The prefix also avoids a clash with a library's own `rocblas` module.

The packaged track changes both.
Like NVIDIA's modules (`use cublas`, not `nvfort_cublas`), it drops the prefix, and it puts each library's interfaces, types and enums into a single module.
The unprefixed name is also the name of some libraries' own modules (`rocblas_module.f90`, `rocsparse.f90`, ...): while those are deprecated, a program uses one or the other, and once they are removed the generated module is the only `rocblas`.
The HIP runtime module becomes `hip`, so a program writes `use hip` where it wrote `use hipfort`.
The cross-cutting modules `hipfort_check` and `hipfort_auxiliary` disappear, and their contents come from the library module the program already uses: `hipCheck` from `hip`, `rocblasCheck` from `rocblas`.
hipFFT is the one library split into two modules, `hipfft` and `hipfftxt`, both in `libhipfft_fortran.a` and the `hipfft-fortran` package.
hipfort already has the same split (`hipfort_hipfftxt`).
Every `hipfftXt*` and `hipfftMp*` routine is in `hipfftxt`, including `hipfftXtSetWorkArea`, even though `hipfft.h` declares it, and every other `hipfft*` routine is in `hipfft`.
`hipfftxt` uses and re-exports `hipfft`, so `use hipfftxt` alone gives the whole API.
hipFFTW, a separate C library, keeps its own `hipfftw` module and archive.
Handles are `type(c_ptr)` (see [Ambiguous or unmappable C](#ambiguous-or-unmappable-c), in the [generator appendix](#appendix-the-generator-rocm-fortran-generator)).
A library that needs another library's types `use`s that library's module: rocSOLVER, for instance, uses rocBLAS.

The `hip*` module names follow NVIDIA's [CUDA Fortran interface modules](https://docs.nvidia.com/hpc-sdk/compilers/fortran-cuda-interfaces/), with `hip` for `cu`: `cublas` becomes `hipblas`, `cufft` and `cufftXt` become `hipfft` and `hipfftxt` (Fortran names are case-insensitive, so `use hipfftXt` works), `cusparse` becomes `hipsparse`, `cusolverDn` becomes `hipsolver`, `curand` becomes `hiprand`, and `cudafor` becomes `hip`.
The `hip*` APIs follow cuBLAS, cuFFT, cuSPARSE, cuSOLVER and cuRAND, and routine and constant names map the same way (`cublasSgemm` to `hipblasSgemm`, `CUBLAS_OP_N` to `HIPBLAS_OP_N`).
A CUDA Fortran library call therefore maps to its `hip*` counterpart by renaming, apart from handle declarations and CUDA Fortran's language extensions, which have no HIP counterpart.

The generator emits both: the packaged, unprefixed modules committed in the monorepos' `bindings/fortran/`, and the split, prefixed layout `ROCm/hipfort` keeps until 11.0.

### Calling styles

Each routine is exposed under its public name as one *generic*, Fortran's overloading mechanism.
A generic groups several *specifics*, concrete procedures that differ in their argument types.
The compiler picks the specific from the actual arguments, so a caller passing a Fortran array never calls `c_loc` itself.

One specific takes raw `type(c_ptr)` arguments.
It is today's hipfort interface and what OpenMP target data needs, so existing code keeps working.
As in hipfort, the raw `bind(C)` specific of a routine `X` is named `X_` and stays callable by that name, so adding an ergonomic form later never changes what `X` or `X_` means.
The ergonomic forms sit under the same name: a plain Fortran array, a scalar passed by reference and an optional argument just work, and for ROCTx so does a `character(kind=c_char)` string.

The ergonomic array forms take a Fortran array and pass its address to C with `c_loc`, which sets a contract:

- The array must be contiguous, have nonzero size and have the `TARGET` attribute, which allows taking its address.
  A non-contiguous section such as `a(1:n:2)` is passed as it is, not copied, because no array form in either array tier declares its arguments `contiguous`.
  Passing it to `c_loc` is not standard-conforming.
  In practice the compilers return the address of its first element, and the library silently reads the wrong elements.
  A zero-size array, as in a call with `n = 0`, is not conforming either, and the raw `type(c_ptr)` form covers that case.
- The array must outlive the operation.
  It must stay allocated, and the host must not modify an array the device reads, until the operation completes on its stream (for example after `hipStreamSynchronize` or an event wait), as with the C API.
- Stream-ordered calls need no copy, and get none.
  Most math-library routines and the HIP `*Async` calls are *stream-ordered*: they enqueue the work and return before the device is done.
  An ergonomic form passes the caller's own array, never a copy, so under the two rules above it is safe for such calls.
  The one trap is an actual argument that is an expression instead of a variable: the compiler evaluates it into a temporary, which is freed while the device may still read it.
- No wrapper hands the device its own copy.
  A wrapper's local data, such as a computed shape or stride array or a NUL-terminated copy of a string, is freed when the wrapper returns, so the generator never passes a pointer to such data for an argument the device uses after the call returns.

That is why no argument is declared `contiguous`.
With that attribute, the compiler would copy a non-contiguous section into a host temporary.
For an array in device memory (`hipMalloc` + `c_f_pointer`), that copy reads device memory from the host, which faults on a discrete GPU, and the device would then read a host temporary, which needs XNACK (GPU page-fault retry).
For a stream-ordered call, the temporary would also be freed before the device uses it.
Contiguity is therefore a correctness requirement on the caller, not a performance one.

Scalars such as `alpha` and `beta` follow the library's *pointer mode*, a per-handle runtime setting that says whether they are read from host or device memory.
A form that takes such a scalar as a Fortran variable is only correct when the handle is in host-pointer mode.
The wrappers never set or restore the pointer mode themselves: it is shared, mutable handle state, and touching it would race with other threads.
In device-pointer mode, the caller passes those scalars as `type(c_ptr)`.

### Language standard

The bindings need Fortran 2003, and two features raise that minimum in well-defined steps.
Fortran 2003 brings `iso_c_binding`: `bind(C)`, `type(c_ptr)`, `c_loc`, `VALUE`, interoperable derived types and `enum, bind(C)`.
The raw `type(c_ptr)` interfaces need nothing more, so a site restricted to Fortran 2003 can use them as they are, built with `FORTRAN_ARRAY_INTERFACES=none`.
The ergonomic array forms need Fortran 2008.
They take *assumed-shape* array arguments (Fortran 90), which receive their shape from the caller's array (`dimension(:)`, `dimension(:,:)`, ...), and pass them to `c_loc`, which Fortran 2003 allows only for explicit-shape or assumed-size arrays.
Fortran 2018 is needed only for the opt-in *assumed-rank* forms (`-DFORTRAN_ARRAY_INTERFACES=assumed-rank`), where one `dimension(..)` interface accepts an array of any rank.
The default tier, `assumed-shape`, is therefore Fortran 2008, and so is the precompiled `amdflang` set.

CI will hold each tier to its level by compiling every module with gfortran at `-std=f2003` for `none`, `-std=f2008` for `assumed-shape` and `-std=f2018` for `assumed-rank`.
A construct from a later standard then fails the build instead of reaching a site that relies on the lower one.
The tiers meet these levels today when checked by hand, but nothing enforces them yet: the generator's compile gate builds all three at `-std=f2018`.

In CI, the regeneration workflow's header check compiles the modules with gfortran, and TheRock, once its CI builds the bindings, builds them with `amdflang`.
They also build with Cray CCE, which is tested locally but not in CI.
The minimum versions of these three compilers are an open question.
Each binding also carries toolchain files for Intel (`ifx`, `ifort`) and NVIDIA HPC (`nvfortran`) as a convenience, but these compilers are not supported.
A `.mod` file has no standard format, so it cannot be shared across compilers, or even across versions of one compiler.

#### Fortran 77 code

Fortran 77 code can use the bindings, because what matters is the compiler, not the age or the form of the source.
A Fortran 77 code base built with a current compiler (gfortran, amdflang, ...) calls the bindings like any other code.
The file that calls them adds one `use rocblas` line, and fixed-form source, implicit typing and the rest of the code are untouched.

Only a compiler, or a compiler mode, limited to Fortran 95 or older cannot use them.
The bindings are Fortran modules built on `iso_c_binding`, from Fortran 2003.
A compiler that stops below Fortran 90 has no modules and no `use` statement, so it cannot read them, and a current compiler held to Fortran 95 by a flag (`gfortran -std=f95`) rejects `iso_c_binding`.
Such a file must instead be compiled in the compiler's default mode, and fixed-form source needs no extra flag.

This takes nothing away, since hipfort never offered a path for such a compiler either.
A path also stays open.
It would be a separate artifact: a C shim library with Fortran 77 linkage, plus an include file of `PARAMETER` constants, emitted by the same generator from the same declarations and metadata.
That would still be a project of its own, with a new emitter, tests, packaging and symbol-name conventions, for callers that cannot use the modules.
If an application needs it, that path gets its own RFC, scoped to its entry points, and nothing decided here would change.

### CUDA backend

The generated modules are `.F90` files, and the uppercase extension makes the compiler run the C preprocessor on them.
hipfort's sources also carry `#ifdef USE_CUDA_NAMES` branches, which bind to CUDA instead of HIP when compiled with `-DUSE_CUDA_NAMES`.
The packaged modules carry no CUDA branches, and their preprocessing only selects the array tier.

The CUDA backend lives only in the compatibility track (`ROCm/hipfort`), which keeps it unchanged in its source until ROCm 11.0, on a best-effort basis and with no NVIDIA CI.
The packaged track does not carry it: it is not in the monorepos, not packaged and not shipped in ROCm.
The `hip*` bindings (hipBLAS, hipSPARSE, hipSOLVER, hipFFT, hipRAND) build against the installed ROCm libraries only, and none is built or tested against a CUDA-backed library.

## Migration and compatibility

Existing hipfort users make two mechanical edits and lose only one capability, the CUDA backend.
Routine names, argument lists and calling styles stay the same.
Only the `use` lines and the link targets change, so the migration is a find-and-replace.

The first edit is to the module names: the `hipfort_` prefix is dropped, and each library becomes a single module.

| Old                                                | New           |
| :------------------------------------------------- | :------------ |
| `use hipfort` (+ `hipfort_types`, `hipfort_enums`) | `use hip`     |
| `use hipfort_rocblas` (+ `_enums`)                 | `use rocblas` |
| `use hipfort_<lib>` (+ `_enums`, `_types`)¹        | `use <lib>`   |

¹ `_types` exists only for rocRAND, hipFFTW and hipFFT's Xt module (`hipfort_hipfftxt_types`).

hipFFT is the exception to one module per library: hipfort already has the same split, and `hipfort_hipfftxt` (+ `_enums`, `_types`) becomes `use hipfftxt`.

Other modules fold in the same way.
For users of a library's own module, `use rocsparse_enums`, `use hipsparse_enums`, `use rocblas_enums`, `use hipblas_enums` and `use hipsolver_enums` become `use <lib>`.
The HIP memory helpers (`hipfort_hipmalloc`, `hipfort_hipmemcpy`, `hipfort_hiphostregister`) fold into `hip`.
`hipfort_check` and `hipfort_auxiliary` fold into the library module the program already uses, as described in [Naming and module structure](#naming-and-module-structure), and handles stay `type(c_ptr)`.

The second edit is to the link target.
hipfort's one archive per backend (`libhipfort-amdgcn.a`) becomes one archive and one CMake target per library, in the C library's own namespace.
`find_package(<lib>-fortran)` finds it and also brings in the C library.
If rocm-cmake's maintainers accept a hook that lets a library's generated `<lib>-config.cmake` optionally include the Fortran config, `find_package(<lib>)` alone will do (see [Building the bindings](#building-the-bindings), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).
That hook is not emitted for rocRAND and hipRAND while they install their legacy `<lib>-fortran-config.cmake`.
Until the hook lands, `find_package(<lib>-fortran)` is the entry point.

```cmake
# before
find_package(hipfort REQUIRED)
target_link_libraries(app PRIVATE hipfort::rocblas)

# after
find_package(rocblas-fortran REQUIRED)
target_link_libraries(app PRIVATE roc::rocblas roc::rocblas_fortran)
```

The namespace is the C library's, which does not always match the library's `hip` or `roc` prefix:

- `hip::` for the HIP runtime, hipFFT, hipFFTW and hipRAND
- `roc::` for rocBLAS, hipBLAS, rocSPARSE, hipSPARSE, rocFFT, rocSOLVER, hipSOLVER and rocRAND, matching the `roc::hipsolver_fortran` that standalone hipSOLVER already exports

ROCTx is the exception: its C target is `rocprofiler-sdk-roctx::rocprofiler-sdk-roctx`, and its Fortran target is `roc::roctx_fortran`.

Without CMake, a build passes the per-compiler directories itself, for example `-I/opt/rocm/include/fortran/amdflang -L/opt/rocm/lib/fortran/amdflang -L/opt/rocm/lib -lrocblas_fortran -lrocblas -lamdhip64`.

hipfort and the packaged bindings can be installed at once, so a code base can migrate one file at a time.
Their module names differ, their `.mod` files land in different directories, their archives have different file names, and their object symbols are decorated with the module name (with gfortran, `__hipfort_rocblas_MOD_...` versus `__rocblas_MOD_...`).
The one rule is not to `use` both bindings of the same library in one scope, because they export the same public names and a reference would be ambiguous.
That clash is a compile-time error confined to one scope, so a file-by-file migration works.

The CUDA (`nvptx`) backend does not carry over: it is not built for the packaged track (see [CUDA backend](#cuda-backend)).
`hipfort_cuda_errors`, which only serves that backend, has no packaged counterpart.
`roctx` does carry over, as `use roctx` and `roc::roctx_fortran`, from `rocm-systems`' `bindings/fortran/`.

The user-facing migration guides live in the hipfort documentation, outside this RFC, so that they follow the packaging as it lands.
They move to the Fortran entry point of the ROCm documentation (see [Tests and documentation](#tests-and-documentation)) before hipfort is removed, no earlier than 11.0, so they stay reachable.
They give the full per-library table, the compile and link lines, the cases where users have to build a binding themselves, and a scripted rename.
There are two guides, for two populations:

- hipfort users, who rename, and can overlap the two bindings until 11.0
- users of a library's own module, who change some call sites between its deprecation at 10.3 and its removal, no earlier than 11.0

## Roadmap and rollout

### Version targeting and roadmap

The bindings target the ROCm version of the headers the generator reads: the `develop` branch of the `rocm-systems` and `rocm-libraries` monorepos, or a ROCm install.

The release numbers below are targets, and the design does not depend on them.
What this RFC fixes is the destination and the order of the steps.
The bindings ship in two steps:

- **Tech preview, targeted for ROCm 10.3.** The bindings are in the monorepos, and TheRock's CI builds and tests them.
  They reach TheRock's tarballs only once TheRock's release builds enable them, and the OS packages only once their packaging is approved (see [Decisions requested](#decisions-requested)).
  Until then, users build a binding from its monorepo against a nightly ROCm install.
  Their interfaces can still change.
  The documentation is one short page on how to enable and build them and on their limitations.
- **Production, targeted for ROCm 10.4.** The bindings ship in the approved OS packages, with the full documentation and the compatibility promise below.

During the transition, both tracks are generated from the same headers.
The compatibility track is `ROCm/hipfort` with its current layout: the `hipfort_` prefix, the CUDA backend and separate interface, type and enum modules.
It is kept as source and not packaged.
Users who build it themselves keep its layout and module names through the 10.x series, until ROCm 11.0.
The bindings team maintains it beside the packaged track through the whole 10.x series, regenerating it against each 10.x release's headers and keeping its HIP backend building and passing its tests.
Its [CUDA backend](#cuda-backend) stays in its source as it is, best effort, with no NVIDIA CI.
From 10.3 to 11.0, users can therefore migrate on their own schedule, one file at a time.
New features land only in the packaged track.

The compatibility promise is a proposal (see [Decisions requested](#decisions-requested)).
A packaged binding is versioned with its library, and its compatibility follows the C library's.
From production, code that compiles and links against a 10.4 binding keeps compiling and linking against any later 10.x binding, unless the C library itself changes the bound function.
The 10.3 tech preview carries no such promise.
A binding's CMake package accepts any version of its library with the same major number (`SameMajorVersion`), and a library's major version changes only at a ROCm major release.
A precompiled `.mod` still needs the compiler that wrote it.
Users therefore revisit their build at a major release, not at every point release.

At link time, this promise relies on each archive having one section per procedure and on the program being linked with `-Wl,--gc-sections`.
A wrapper whose C function is missing from the installed library then breaks only the programs that call it, since the linker drops the sections no program uses.
The bindings' CMake compiles the archives with one section per procedure for GNU, LLVM Flang (including `amdflang`), Classic Flang and NVHPC.
Cray CCE and the Intel compilers do not put each procedure in its own section, so with those a wrapper whose C function is missing breaks every program that links the binding.
The CMake packages do not add `-Wl,--gc-sections` to the consumer's link, so a program adds it itself.
With `amdflang` and `ld.lld`, an OpenMP offload program linked that way also needs `-Wl,-z,nostart-stop-gc` until the toolchain is fixed (see [Open questions and risks](#open-questions-and-risks)).

Within a major series, a binding only grows: new routines, new specifics in a generic, new enumerators and types.
An existing Fortran signature changes only when its C function does, or at a major release, even when the change would come from the generator or a metadata entry.
The one exception is a fix.
An interface that does not match its C function (a wrong type or kind, a scalar bound as an array) is corrected in the release that finds it, since code compiled against it was already calling the library incorrectly.

The compatibility track covers only one of the four ways users get hipfort today, and the table lists all four.

| How you get hipfort today                                                            | What the transition does to you                                                                                                                                                                                                                                                   |
| :----------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| You build it from source                                                             | Nothing. The compatibility track keeps its layout and module names, and its CUDA backend stays in its source, best effort.                                                                                                                                                        |
| You install a package                                                                | Nothing: ROCm 10.0 to 10.2 ship no hipfort package, `.mod` or archive, so there is nothing to lose. Once the bindings' packaging is approved (targeted for 10.4), the proposed `-fortran-devel` packages carry the new modules, not hipfort's: adopting them is the rename above. |
| You use a library's own Fortran module (`rocblas_module.f90`, `rocsparse.f90`, ...)  | Deprecated from 10.3, when the generated module supersedes it, and removed no earlier than ROCm 11.0. See [What changes for a library repository](#what-changes-for-a-library-repository).                                                                                        |
| You build one code base on several ROCm versions, some before 10.3 (for example 6.x) | Nothing until your oldest system reaches 10.3: use the hipfort release tagged for each ROCm (`rocm-<version>`). See the note below.                                                                                                                                               |

A hipfort package from an earlier ROCm keeps working against that ROCm, and once such a system moves to 10.3 or later, it falls in the last row of the table.
For that row, a thin wrapper module can also select `use hipfort_<lib>` or `use <lib>` per system at compile time, and the packaged bindings can be generated from an older release's headers on request.

No row is a break during the 10.x series.
For the third row, ROCm's packages ship no compiled hand-written module (see [Background](#background)).
No ABI is at stake, so removing these modules is a source-level change, made no earlier than ROCm 11.0 and coordinated with each library.
If a distribution still publishes its own hipfort package, that package continues through the transition, and any removal is announced together with that distributor.
Keeping both tracks until 11.0 costs little, because neither is written by hand.

### Rollout and success criteria

The rollout has four stages, each with a clear exit.

1. **Bring current (7.14 / 10.0 / 10.1 / 10.2).** The committed bindings match the current headers.
   This is done for 7.14, 10.0 and 10.1 (hipfort 0.8.0, 0.9.0 and 0.10.0), and the 10.2 regeneration is merged and awaits its release.
   hipfort now also builds with the Cray compiler, CCE, tested locally and not in CI.
   The bindings' first pull requests were produced by regenerating each binding and committing the result by hand.
   The regeneration workflows, one per binding, run the public generator, [`ROCm/rocm-fortran-generator`](https://github.com/ROCm/rocm-fortran-generator), pinned by `GENERATOR_REF`, and open their pull requests through each monorepo's existing GitHub App.
   They are merged in both `rocm-systems` and `rocm-libraries`, and until a binding's directory exists, its workflow runs only the header check.

1. **Tech preview (targeted for 10.3).** Each in-scope library has a binding in its monorepo's `bindings/fortran/`.
   TheRock enables the bindings in steps, each a separate change:

   - build support, with `THEROCK_ENABLE_FORTRAN_BINDINGS` off by default, even with `THEROCK_ENABLE_ALL`, and one feature per family (`THEROCK_ENABLE_RUNTIME_FORTRAN`, `THEROCK_ENABLE_BLAS_FORTRAN`, and so on for `SOLVER`, `SPARSE`, `FFT` and `RAND`)
   - CI: the `fortran-bindings` build stage and one GPU test job per binding subproject
   - releases, with the packaging, in a later change

   Until the releases step, TheRock's release workflow skips the `fortran-bindings` stage, so its nightly and release tarballs do not carry the bindings.
   The hand-written library modules are deprecated.
   The regeneration workflows are merged in both monorepos: they already check header pull requests, and once a binding has merged, its workflow also regenerates it daily and opens the bot's pull request.
   Before the bindings team relies on them for the tech preview, they still have to:

   - pin the source of the HIP headers that `rocm-libraries` reads, which today come from `rocm-systems`' default branch
   - open an issue for the header's owners when a regeneration fails, such as for a header that is valid C++ but not valid C
   - list the changes that need a decision (removed names, changed signatures, renamed or unclassified arguments) in the bot's pull request, for the bindings team to resolve before merging
   - run on release branches

   Before 10.3 ships, every `bind(C, name=)` label must resolve against the shipped libraries.

1. **Production (targeted for 10.4).** The bindings ship in the approved OS packages, and the compatibility promise starts.
   The Fortran entry point under *ROCm HPC SDK* carries the full documentation, and the migration guides move there.
   The choice of static archives is confirmed with large HPC sites (see [Open questions and risks](#open-questions-and-risks)).

1. **Retire the compatibility track (11.0).** `ROCm/hipfort` is removed, no earlier than this release, and the packaged track is the only deliverable.
   The hand-written library modules are removed no earlier than this release.

The rollout is done when:

- `rocm-systems`' `bindings/fortran/` (`hip`, `roctx`) and the ten `bindings/fortran/<lib>/` of `rocm-libraries` (twelve modules) carry generated source, tests and, where needed, curated metadata
- a header change that changes a binding is followed by the bot's pull request on its next daily run when the regeneration succeeds, and the bindings team aims to merge that pull request within five business days (a target, see [Decisions requested](#decisions-requested))
- a regeneration on each release branch reports any diff to the release owners, who decide whether it blocks the release (see [Decisions requested](#decisions-requested))
- `find_package(<lib>-fortran)` exposes `<ns>::<lib>_fortran` when the binding is built, where `<ns>` is the C library's namespace (`roc::` for ROCTx)
- TheRock's CI builds and tests every binding, and its release builds enable and package them once the packaging is approved (see the [Packaging in TheRock](#appendix-packaging-in-therock) appendix)
- each array tier compiles at its declared standard (see [Language standard](#language-standard))
- an external program compiles and links against the installed package with at least `amdflang` and gfortran
- the bindings ship in the OS packages, and the Fortran entry point carries their full documentation and the migration guides (10.4)
- the hand-written library modules are removed, and the libraries' Fortran clients no longer use them (no earlier than 11.0)
- `ROCm/hipfort` is removed (no earlier than 11.0)

Adding the bindings is one pull request per library, twelve in all: one for HIP and one for ROCTx in `rocm-systems`, independent of each other, and one per `rocm-libraries` binding.
Each one only adds its binding to `bindings/fortran/<lib>/`, with its own copy of the `amdflang` probe, its CODEOWNERS line and the entry that lets the monorepo's CI detect changes to it, and leaves the library untouched.
Every `rocm-libraries` binding test uses `hip`, except rocRAND's and hipRAND's version checks, so these tests need the HIP binding installed.
Until it is, each such test suite is skipped at configure time, so the tests only run in TheRock once its `rocm-systems` pin contains the HIP binding.
The pull requests therefore merge in this order:

1. TheRock's build support, with the bindings off by default
1. TheRock's CI stage and its GPU test jobs
1. the bump of the TheRock commit that `rocm-systems`' and `rocm-libraries`' CI pin, so that their CI builds the `fortran-bindings` stage and the binding pull requests test their own binding
1. the HIP and ROCTx bindings in `rocm-systems`, in either order
1. TheRock's bump of its `rocm-systems` pin, which TheRock's bot opens daily
1. the ten `rocm-libraries` bindings, and then TheRock's bump of its `rocm-libraries` pin

All the binding pull requests share one shape:

- the options of [Building the bindings](#building-the-bindings)
- a static `<lib>_fortran` target in the C library's namespace (`roc::` for ROCTx)
- the `<lib>-fortran` package, installed in its own `<libdir>/cmake/<lib>-fortran/` so that `find_package(<lib>-fortran)` resolves
- the install tree of [Layout after the split](#layout-after-the-split)
- the test convention of [Tests and documentation](#tests-and-documentation)

Each pull request already carries its binding's `metadata/` (ROCTx's is an empty placeholder).
hipBLAS's no longer builds a temporary `hip` stub module.
Each is regenerated with the current generator when it is rebased for merging.

The deprecation pull requests follow, one per library with a hand-written module, once its binding has merged.

## Open questions and risks

The decisions this RFC asks for are collected in [Decisions requested](#decisions-requested).
The points below are either settled with a residual risk, still open, or risks to watch.

These points are settled, with a residual risk:

- **Cross-repository header dependencies.**
  Some bindings depend on another library's headers (rocSOLVER on rocBLAS) or on the HIP runtime, so the modules have a compile order.
  This mirrors the existing C dependency graph and is settled the same way: the order of the bindings in TheRock's build stage, and `find_package` for a standalone build (see [Building the bindings](#building-the-bindings), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).
  Each binding's regeneration workflow also watches the headers it depends on (rocBLAS's for rocSOLVER, rocRAND's for hipRAND, hipblas-common's for hipBLAS), so it catches drift across headers.
- **Assumed-rank interfaces (Fortran 2018).**
  The default stays the `assumed-shape` tier, whose per-rank forms work on older compilers, and a build option, `-DFORTRAN_ARRAY_INTERFACES=assumed-rank`, selects assumed-rank instead.
  The tiers exclude each other, so this is one option with three values.
  Two booleans could express an illegal combination.
  Enabling assumed-rank *replaces* the per-rank forms, because the compiler cannot tell an assumed-rank argument from the per-rank ones by rank, so they cannot legally coexist in one generic.
  The residual risk is compiler support: Fortran 2018 assumed-rank is not equally mature in gfortran, LLVM Flang and Cray CCE, and HPC sites adopt new standards late.
  It compiles with `amdflang` today, but corner cases are a risk.
  It can become the default once the compilers are proven, which would also shrink the generated sources and their build time.

These points are still open:

- **Minimum compiler versions.**
  The minimum standard is settled at Fortran 2003 (see [Language standard](#language-standard) and [Fortran 77 code](#fortran-77-code)), but the minimum compiler versions are not.
  The question worth asking the large HPC sites is which compiler versions they pin.
- **Size of the generated files.**
  A library's binding is one generated file, and for the large APIs it is very large, around 165k lines each for rocBLAS, hipBLAS and rocSOLVER.
  A concern raised during the review of this RFC is that this is too large for GitHub to show in a diff.
  There are two options:
  - split the output per C header: one `.F90` per header, still assembled into one module (with Fortran `include` or submodules), which keeps diffs reviewable at the cost of more files
  - keep one file and treat it like any generated artifact, as the bindings already do by marking their `.F90` `linguist-generated`, and review the generator and the metadata instead of the file line by line.
    The regeneration workflow makes this option safer, because the regenerated diff then comes from the tool and not from a person.
- **Docstrings and the documentation site (Sphinx and Doxygen).**
  The layout is decided (see [Tests and documentation](#tests-and-documentation), in the [build appendix](#appendix-building-installing-and-testing-the-bindings)).
  A binding has no `docs/` subdirectory, and each library documents its Fortran API from its existing top-level `docs/`.
  For HIP and ROCTx, the documentation is built from `rocm-systems`' `bindings/fortran/hip/` and `bindings/fortran/roctx/`.
  It could instead be built from HIP's and rocprofiler-sdk's `docs/`, but only if their owners agree, since the bindings otherwise stay out of those projects.
  Still undecided is whether to emit the headers' Doxygen comments as `!>` docstrings and publish a Fortran reference from them.
- **The libraries' Fortran clients at removal.**
  rocBLAS, hipBLAS, hipSOLVER, rocSPARSE and hipSPARSE have Fortran code in their `clients/`: samples, and Fortran shims that their C++ tests call.
  While the hand-written modules are deprecated, those clients keep compiling them.
  Once the modules are removed, the clients can no longer use the generated binding in the same build, since the binding builds after its library.
  The samples can move into their binding, beside its tests, since what they show is how to call it.
  For the shims of the C++ tests, the options are to drop their Fortran variants, since each binding has its own tests, or to compile the binding's `.F90` into the clients, which ties the library's build to the binding again.
  Each removal pull request settles this for its library.
- **Staffing of the bindings team.**
  hipfort stalled when no one maintained it.
  The split removes the hand-copying but not the work: the bot's pull requests, the metadata, the compatibility track and the generator still need people, and the release-branch regeneration adds a report to each release.
  If that team is short of people, the bot's pull requests wait and the bindings lag their headers, although no C pull request waits on a regeneration.
  Two things limit that risk.
  CODEOWNERS shares each binding with the library's team, which can review and merge a bot pull request.
  The generator documents how to classify an argument, so a library developer can do it.
  A fallback, under which the library's CODEOWNERS can merge a bot pull request after ten business days, is proposed in [Decisions requested](#decisions-requested).
- **Static archives for HPC users.**
  The bindings ship as static archives, as hipfort already does, so a program needs no Fortran shared library at run time.
  This is to be confirmed with large HPC sites and AMD's HPC application teams before 10.4.

These are the risks to watch:

- **Required binding test jobs.**
  Once TheRock's CI builds the bindings, their test jobs are to be required, and they pass as long as the C API only grows (see [What this means for a library maintainer](#what-this-means-for-a-library-maintainer) and [Decisions requested](#decisions-requested)).
  A binding test fails only when a change breaks the C API or ABI.
  If a binding test fails for a reason the bindings team cannot fix quickly, TheRock can mark that job `expect_failure` until it is fixed, as for any component's test.
  The regeneration workflow's header check on library pull requests is informative and not required.
- **CI cost.**
  TheRock's CI adds the `fortran-bindings` build stage and twelve GPU test jobs, one per binding subproject (hipFFT's job covers hipFFTW).
  Consumer-based test selection runs only the jobs of the bindings whose library a pull request changes, once TheRock's committed consumer graph includes the bindings, which needs the graph's drift check to configure with the bindings on.
  The jobs' wall time is measured during the CI step, before the releases step enables the bindings.
- **Building the bindings in a generic stage.**
  TheRock builds the math libraries once per GPU architecture, but the bindings are host-only and are built once, in a generic stage.
  That stage builds against the generic part of the math libraries' artifacts, which their split into generic and architecture-specific components provides.
  It is the first generic stage to depend on artifacts built per architecture: their subprojects are declared there as pre-built, with no AMDGPU target selected.
  TheRock's CI change for the bindings allows that, and is not merged yet, so this precedent is for TheRock's maintainers to accept.
  If the split is ever turned off, the stage needs another source for the libraries' headers and host libraries.
- **`--gc-sections` and OpenMP offload.**
  The compatibility promise and the bindings' tests rely on `-Wl,--gc-sections`.
  With `amdflang` and `ld.lld`, that option removes the OpenMP offload entry table, so an OpenMP offload program aborts at its first target construct.
  Such a program must also link with `-Wl,-z,nostart-stop-gc` until the toolchain fix lands.
- **Headers that are not valid C yet.**
  The generator parses every header as C23 (`-std=gnu2x`), because rocSPARSE's attributes on enumerators are valid only from C23.
  ROCTx declares a struct in a prototype, which C accepts only with a warning, and hipFFT's `hipfftXt.h`, which its Xt and Mp headers rely on, includes a C++ header.
  The generator keeps a workaround only for `hipfftXt.h`, a stand-in header of its own, and the others are to be fixed in the headers themselves.
- **Compile time of the large modules.**
  Compile time depends on the compiler as much as on the size of a module, and a construct that costs nothing in one compiler can be very slow in another.
  With Cray CCE, a large hipfort module took about an hour to compile while each interface carried its own `use` statements, and a few seconds once the generator `use`d each module once at module level and `import`ed the needed entities in each interface body.
  No CI tracks the build time of the bindings, and none builds them with CCE: hipfort and the bindings are tested with CCE locally only.
  Splitting the output per header with submodules (see the size point above) would also let a build compile one library's binding in parallel.
- **CODEOWNERS and write access.**
  Each binding's CODEOWNERS line names the library's team and the bindings team, but it takes effect for the latter only once that team has write access to both monorepos, which it does not have today.
  With both teams on one line for the whole binding, `metadata/` included, an approval from either team satisfies the review requirement (see [Decisions requested](#decisions-requested)).
- **CUDA backend.**
  The compatibility track keeps hipfort's CUDA backend until 11.0 on a best-effort basis: no NVIDIA CI covers it, so a break there may go unnoticed until a user reports it.

## Alternatives considered

**A single bindings super-repository.**
All bindings could stay in one dedicated repository: today's hipfort, but generated.
That keeps a clean boundary and simple ownership, and it remains a reasonable option.
With a single bindings super-repository:

- the headers must be copied or fetched from two monorepos, at pinned revisions, before every regeneration
- the metadata is separated from the header it describes
- a developer cannot regenerate a binding in the PR that changes its header
- neither the library's CI nor TheRock builds and tests the binding against the same commit's C library

Keeping each binding in its library's repository, but in a `bindings/fortran/` tree of its own, keeps most of that boundary while avoiding these costs (P1, P2).

**Bindings inside each library's project.**
An earlier version of this RFC put each binding in its library's project, `projects/<lib>/fortran/`, added to the library's own build and packaged in the library's development package.
That keeps a binding closest to its header, but it ties the binding to the library's implementation: its build options, its CMake and its packaging.
The libraries' implementations are expected to be reorganized, while their public APIs are not, and the bindings depend only on the APIs.
Building them inside the libraries would make that reorganization harder, so they live in their own tree and are built after the libraries (P2).

## Appendix: The generator (rocm-fortran-generator)

`rocm-fortran-generator` is a Julia program that regenerates the whole binding set from the current C headers, which it parses with [Clang.jl](https://github.com/JuliaInterop/Clang.jl), the Julia bindings to libclang.
It keeps hipfort's routine, enum and check-routine names, so existing hipfort calls compile unchanged.
Only regenerating the bindings needs Julia.
Using the modules does not.

```text
   Header sources (one of)
     a rocm-systems checkout         (HIP, ROCTx)
     a rocm-libraries checkout       (the math libraries, HIP from a rocm-systems checkout or an installed ROCm)
     an installed ROCm               (/opt/rocm/include)
            │  parse as C  (clang -x c, via Clang.jl)
            ▼
   Generator (Julia)
     generate_all.jl  →  common/fortran_gen.jl, common/hip_generics.jl,
                         <lib>/generator.jl + its configuration
                         + the binding's metadata/ from the checkout
            │  emit
            ▼
   bindings/fortran/  (committed, packaged track: one module per library, no prefix)
     rocm-systems bindings/fortran/hip/, bindings/fortran/roctx/:  hip.F90, roctx.F90
     rocm-libraries bindings/fortran/<lib>/:  rocblas.F90, hipblas.F90, rocsparse.F90,
       hipsparse.F90, rocfft.F90, hipfft.F90 + hipfftxt.F90 + hipfftw.F90 (in bindings/fortran/hipfft/), rocsolver.F90,
       hipsolver.F90, rocrand.F90, hiprand.F90
            →  HIP runtime + ROCTx + 10 libraries (12 library modules)
```

For each library, the generator parses the headers as C and emits the interface blocks, the enum and `#define` constants, the derived types and the check routine.
Each library is handled in a separate process.
Each monorepo's regeneration workflow runs the generator with no ROCm installed: `rocm-systems` on its own checkout, and `rocm-libraries` on its checkout plus a checkout of `rocm-systems`' default branch for the HIP headers.
The headers a build would generate (the version, config and export headers, and HIP's `hip_version.h`) are synthesized from the checkout's templates and CMake files.
The feature macros a header tests, such as rocSPARSE's `ROCSPARSE_WITH_*`, are read from the library's CMake options.
A Clang error fails generation for that library only, and the other libraries still generate.

The profiling binding (`roctx`) is a project of its own beside HIP's, `rocm-systems`' `bindings/fortran/roctx/`.
It is built against the installed `rocprofiler-sdk-roctx` and exports `roc::roctx_fortran` from a `roctx-fortran` package.
It is a dozen `bind(C)` interfaces with no array overloads, so the array option `FORTRAN_ARRAY_INTERFACES` (see [Building the bindings](#building-the-bindings)) changes nothing for it.

Regeneration is cheap, which is what lets the regeneration workflows regenerate every binding of a monorepo each day and check each pull request that changes a public header.
It also means a release's final headers can be picked up at the last moment instead of weeks ahead.
The generator is deterministic: the same headers and metadata produce byte-identical output.

Only the emit stage is Fortran-specific.
The front end parses headers with Clang.jl into a language-agnostic description of functions, enums and structs, so swapping the emitter would retarget it to another language.

### Free side effect: C-header hygiene

Because a `bind(C)` interface mirrors the C ABI, the generator parses every public header as C (`clang -x c`).
Valid C headers are therefore a hard prerequisite, which every run checks.
ROCm headers are meant to be C-includable, but in practice they are only pulled from C++, so C-only breakage can go unnoticed for several releases.
Typical cases are:

- a C++ keyword used as an identifier
- a missing `extern "C"`
- a C++-only include

The generator fails on such a header, the regeneration check of the pull request that changed it fails without blocking it, and the bindings team can then follow up with the header's owners.
The C libraries benefit too.
Some public headers are not valid C yet.
rocSPARSE marks deprecated enumerators with the `[[deprecated]]` attribute, valid only from C23, so the generator parses every header as C23 (`-std=gnu2x`).
ROCTx declares a struct inside a function prototype, which C accepts only with a warning.
hipFFT's `hipfftXt.h`, which its Xt and Mp headers rely on, includes `<cstddef>`, a C++ header.
The generator keeps a workaround only for that last case, a stand-in header of its own, until `hipfftXt.h` is fixed.

### Ambiguous or unmappable C

Some valid C constructs are ambiguous or have no Fortran equivalent.
The generator maps each of them to a defined, ABI-correct form instead of dropping it or guessing.
A struct that is not field-mappable, such as one with bit-fields, becomes an opaque byte blob of the exact C size and alignment, so the structs that embed it keep their layout.
A union becomes one such field of its enclosing struct.
Opaque handles, workspace pointers and C strings stay `type(c_ptr)`, as they were in hipfort, so code written against hipfort compiles unchanged.
ROCTx's routines also accept a `character(kind=c_char)` string.
Where `NULL` selects a mode, as in the buffer-size query/execute idiom of rocSPARSE, rocBLAS and rocSOLVER, the metadata lets the caller omit the argument, through a plain (non-`bind(C)`) procedure in the same generic, since `OPTIONAL` is not allowed on the raw surface.
A rocBLAS or rocSPARSE complex maps to Fortran's intrinsic `complex`, which assumes the two have the same size and alignment, an assumption no check verifies yet.
Function-like and internal macros are skipped, and `#define` constants become parameters.
The HIP vector types are skipped too, except where a library's API takes one by value: rocRAND's `uint4`, which a small header of the generator declares so that `rocrand_set_seed_uint4` is bound.
rocSOLVER is generated from its umbrella header, `rocsolver.h`, and a deprecated alias that re-exports a rocBLAS function is bound only in rocBLAS's module.
The [generator repository](https://github.com/ROCm/rocm-fortran-generator) documents each case.

Two ambiguities the C type system cannot resolve on its own are left to the curated metadata: a scalar host value versus a data buffer (`const double* alpha` versus `double* x`, identical in C), and device-side output scalars.
Invalid C is a separate case: the generator fails on it (see [C-header hygiene](#free-side-effect-c-header-hygiene)).

### Curated metadata (the manual surface)

The generator emits everything the C ABI carries: signatures, enums and structs.
A few Fortran-side facts are not in the C headers and cannot be inferred, so they live in a small amount of per-library metadata that a maintainer curates.
These files sit in a `metadata/` directory beside the library's module, `bindings/fortran/metadata/` in `rocm-systems` and `bindings/fortran/<lib>/metadata/` in `rocm-libraries`.
The generator reads that directory when it runs on a checkout, so the metadata is curated in the same repository as the header it describes (P1).
It is the one part of a binding that stays manual.

The main question the metadata answers is whether a pointer argument is one scalar or a data buffer, since `const double* alpha` and `double* x` are identical in C.
For a scalar such as `alpha` or `beta`, it also records whether the library reads it from host or device memory according to the handle's *pointer mode*, a per-handle runtime setting.
Other entries record arguments that accept `NULL`, scalars the device writes, and pointers whose C type hides what they point to.
The metadata format belongs to the generator and can change without this RFC.
It is documented in the generator repository, `ROCm/rocm-fortran-generator`, along with how the classifications were bootstrapped from hipfort's sources.
When a new ROCm function has a scalar pointer argument the bootstrap did not cover, the generator reports it, and that argument must be classified by hand.

The generator configuration holds the project conventions: the name prefix, the headers to parse and their Clang flags, the opaque handles, the modules a library uses, the functions to skip or alias, and the check overrides.
Those are not facts from the headers, so the configuration stays with the generator, which owns the code that reads it.
Every binding's pull request carries its `metadata/` (ROCTx's is an empty placeholder).
The generator keeps its own copy of each library's metadata, which it uses when it runs on an installed ROCm, and a checkout's `metadata/` replaces it.

**Keeping the classifications correct.**
The generator's report catches an *unclassified* argument, but not a *stale* one, which stays well-formed after a function is renamed or its signature changes.
So on each run the generator also checks the entries against the parsed C declarations.
An entry that matches no function or argument (a typo, a removed API, or one newer than the headers read) is reported as a warning, since it changes no binding.
An entry that the headers make inapplicable, such as a scalar classification on an argument that is no longer a pointer to one value, is reported and, by default, ignored: the argument binds as its C declaration says.
`ROCM_FORTRAN_STRICT=1` turns that report into an error, and the regeneration workflows set it.
Such an entry therefore fails both the header check of the pull request, which is informative and does not block it, and the daily regeneration of that binding, until the metadata is fixed.
A header change thus never blocks a library pull request, and the bindings team fixes the metadata before the binding is regenerated.
The generator also cross-checks the metadata against the documentation: when the Doxygen `@param` comment of a pointer argument says it is an array, or a scalar on the host or the device, and the binding disagrees, it reports a warning.
That check and the unmatched-entry check stay warnings in strict mode.
A comment never changes a binding, because the metadata has the last word.
A comment known to be wrong is listed as an erratum in the generator.
Neither check proves an entry right: one that matches, and that no comment contradicts, can still be wrong.
That case is covered two ways:

- a one-time audit of the classifications bootstrapped from hipfort
- runtime tests that exercise **both host-pointer and device-pointer modes** for a representative set of entry points

Those tests prioritize the pointer-mode-ambiguous families (`alpha`/`beta`/`result`), where a wrong classification turns into a fault or a silently wrong answer.

### Testing the generator

The generator is tested at three levels.
A unit-test suite in the generator repository checks its logic with small fixture headers and no ROCm, including a byte-for-byte golden test of the emitters.
The same CI checks that the output compiles and that the array tiers agree, on every supported Julia version.
In each monorepo, the regeneration check compiles the regenerated modules with gfortran in each array tier, on a CPU-only runner.
TheRock's `fortran-bindings` stage then builds each binding with `amdflang` and compiles its Fortran tests (its `test/`).
At runtime, TheRock's binding test jobs run those tests on an AMD GPU, as GPU-labeled CTest tests of each binding.
That runtime suite must cover both pointer modes, host and device, for the entry points whose scalar arguments the metadata classifies.
A misclassified scalar pointer compiles and links cleanly, so it is the one correctness property no compile-only gate can see (see the *Keeping the classifications correct* paragraph of [Curated metadata](#curated-metadata-the-manual-surface)).
The generator repository's README lists the tests and how to run them.

### Dependencies

The generator needs Julia, Clang.jl and the C headers.
It is a standalone, build-time tool that regenerates the committed `.F90`, and it is never part of a consumer build or of the compiler build.
No Julia dependency reaches the compiler, ROCm or any consumer: using the committed `.F90` needs only a Fortran compiler.
Consumers are unchanged: they still link `libamdhip64` and the relevant math-library `.so`.

The coupling to Clang is shallow: the generator parses public headers at the C API surface (`clang -x c`), not Clang's internal AST.
It supports Julia 1.11 to 1.13, each with its own pinned manifest, and Clang.jl 0.19, and it uses the libclang bundled with Clang.jl, with no system Clang involved.

Across repositories, `rocm-systems` and `rocm-libraries` gain a `bindings/fortran/` tree (one project in `rocm-systems`, one `bindings/fortran/<lib>/` per library in `rocm-libraries`) and one regeneration workflow per binding, which invokes the generator.
TheRock builds the bindings but never runs the generator.

## Appendix: Building, installing and testing the bindings

This appendix records the build, install and test decisions, and the bindings' pull requests (see [Rollout and success criteria](#rollout-and-success-criteria)) implement the details.

### Building the bindings

Each binding is a self-contained CMake project, `bindings/fortran/<lib>/` in `rocm-libraries` and `bindings/fortran/hip/` and `bindings/fortran/roctx/` in `rocm-systems`, which builds its sources into the `.mod` and `.a` against the installed library.
It requires CMake 3.19, and 3.20 to run its tests with `ctest --test-dir`, which the versions in use already satisfy.
No library build adds it, so a binding is only ever built that way, by TheRock or by a user, and never forces a Fortran compiler on a C build.

```bash
cmake -S bindings/fortran/rocblas -B build/rocblas-fortran \
  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build/rocblas-fortran
# rocm-systems: HIP, against the installed runtime (ROCTx likewise, from bindings/fortran/roctx)
cmake -S bindings/fortran/hip -B build/hip-fortran -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build/hip-fortran
```

A binding builds behind one switch, `BUILD_FORTRAN_BINDINGS`.
Its default is `${UNIX}`, because ROCm ships no Fortran compiler on Windows, which is out of scope (see [Non-goals](#non-goals)).
The switch is guarded: it builds the binding when a Fortran compiler is present, and otherwise skips it with a `STATUS` message and never fails the configure.
When neither `CMAKE_Fortran_COMPILER` nor `FC` is set, the binding prefers the `amdflang` found beside the C/HIP compiler or under `ROCM_PATH`.
That probe is a `cmake/ROCmFortranCompiler.cmake` in each binding, so that each one builds on its own.

`FORTRAN_ARRAY_INTERFACES` selects the array tier: which ergonomic array specifics each generic carries besides the raw form.
Its values are `none`, `assumed-shape` and `assumed-rank`, and its default is `assumed-shape`.
It replaces hipfort's two booleans `HIPFORT_USE_FPOINTER_INTERFACES` and `HIPFORT_ASSUMED_RANK`.
See [Open questions](#open-questions-and-risks) for why assumed-rank replaces rather than adds.
ROCTx has no array overloads, so the option has no effect on it.

`BUILD_FORTRAN_CLIENTS` builds the binding's own tests in its `test/`.
Its default is `ON`.

Each archive is compiled with one section per procedure (`-ffunction-sections -fdata-sections`) when the compiler is GNU, LLVM Flang (`amdflang`), Classic Flang or NVHPC.
Cray CCE and the Intel compilers do not put each procedure in its own section.
The binding's tests link with `-Wl,--gc-sections`, but the CMake packages do not add it for a consumer: a program that wants to link only the C functions it calls adds it itself.
With `amdflang` and `ld.lld`, that option currently also breaks OpenMP offload programs, which therefore also link with `-Wl,-z,nostart-stop-gc` until the toolchain is fixed (see [Open questions and risks](#open-questions-and-risks)).

For discovery, the Fortran target ships as its own per-library config package, `<lib>-fortran`.
A consumer writes `find_package(rocblas-fortran)`, which calls `find_dependency()` on the C library when the binding was built against an installed one.
Every binding installs it in its own directory, `<libdir>/cmake/<lib>-fortran/` (`<libdir>/cmake/rocblas-fortran/rocblas-fortran-config.cmake`), where `find_package` looks by default.
Nothing goes into the C library's `<libdir>/cmake/<lib>/`: that directory belongs to the C package, and rocRAND and hipRAND already install a `<lib>-fortran-config.cmake` of their own there for their hand-written wrappers.
A separate directory also keeps the C and Fortran distro packages from shipping the same file.

A later step stays possible, if the rocm-cmake maintainers accept it.
Once rocm-cmake grows a hook in the generated C config that optionally includes `../<lib>-fortran/<lib>-fortran-config.cmake`, `find_package(rocblas)` alone will expose `roc::rocblas_fortran` when the binding is installed, and ignore it otherwise.
The hook is not emitted for rocRAND and hipRAND while they include their legacy `<lib>-fortran-config.cmake`.
The Fortran config skips its `find_dependency()` when the C target exists or the C package is the one being found, so the hook cannot recurse.
Until then the `-fortran` package is the entry point, and consumer code written against it keeps working afterwards.
The same is possible for HIP and ROCTx, whose C configs come from clr and rocprofiler-sdk rather than from rocm-cmake, but it touches two projects the bindings otherwise stay out of.
It is a possibility to propose to their owners later, not part of this RFC, and until then `find_package(hip-fortran)` and `find_package(roctx-fortran)` are the entry points.

Each library's Fortran target sits in its C library's existing CMake namespace (`roc::rocblas_fortran`, `hip::hip_fortran`, `hip::hipfft_fortran`, ...), with no hip-versus-roc rule, except ROCTx's `roc::roctx_fortran`.
The full list is in [Migration and compatibility](#migration-and-compatibility).

The consumer-facing presence flag keeps the variable name already in the tree, `<pkg>_FORTRAN_FOUND`, now defined by the Fortran config package.
The config requires Fortran to be enabled before `find_package`.
When no `.mod` matches the consumer's compiler, it sets `<pkg>_FORTRAN_FOUND` and the package's own `_FOUND` to `FALSE`, with a message that names the installed compilers and the `<LIB>_FORTRAN_COMPILER_DIR` override, so a user can tell a missing rebuild from a wrong path.
Each per-library Fortran package is versioned with its C library (`SameMajorVersion`), so a binding resolves to the same major version as the C library it was generated from.
The version is read from the library's sources in the monorepo (`projects/<lib>/CMakeLists.txt`, `projects/hip/VERSION`, ...), so a binding built outside the monorepo installs no version file and a versioned `find_package` does not match it.

The generated hipSOLVER binding takes the name of hipSOLVER's existing Fortran target, `roc::hipsolver_fortran`, as a static `libhipsolver_fortran.a` in the `hipsolver-fortran` package.
The hand-written `hipsolver_module.f90` and its shared `libhipsolver_fortran.so`, which a standalone hipSOLVER build exports under that name today, are deprecated with the other hand-written modules (see [What changes for a library repository](#what-changes-for-a-library-repository) for the clash while both exist).
ROCm's packages do not ship that library, since TheRock builds hipSOLVER with `EXPORT_FORTRAN_BINDINGS=OFF`.
A consumer moves to `find_package(hipsolver-fortran)` until the rocm-cmake hook lands.

Cross-module dependencies are handled the way the C libraries already handle them.
`rocsolver` uses `rocblas`, because the rocSOLVER API takes `rocblas_handle` and rocBLAS enums, which is the existing C dependency graph, now in Fortran.
TheRock builds rocBLAS's binding before rocSOLVER's, and a standalone build finds it with `find_package(rocblas-fortran)`.
Without it, the rocSOLVER binding, and so its tests, is skipped with a `STATUS` message.
The rocBLAS `.mod` must come from the same compiler as the rocSOLVER binding, so a non-`amdflang` site builds the two bindings in that order.

### Layout after the split

Each binding is `bindings/fortran/<lib>/` in its monorepo: `hip` and `roctx` in `rocm-systems`, and one per math library in `rocm-libraries`.
Each holds the generated modules, their curated metadata, their build files and their tests.
The generator code is never vendored here (P3).
Only the metadata it reads is kept beside the binding, in `metadata/` (see [Curated metadata](#curated-metadata-the-manual-surface)).

```text
rocm-systems/  bindings/fortran/hip/     a standalone project, built against the installed HIP
   hip.F90                       generated: module hip (runtime + types/enums/check/auxiliary)
   metadata/                     curated metadata
   CMakeLists.txt                → libhip_fortran.a + hip.mod
   hip-fortran-config.cmake.in   find_package export → hip::hip_fortran
   cmake/ROCmFortranCompiler.cmake   the amdflang probe
   cmake/toolchains/             example toolchain files (amdflang, gnu, cray, intel, intel-classic, nvhpc)
   test/                         Fortran tests
   CHANGELOG.md, .gitattributes  the .F90 are marked linguist-generated
rocm-systems/  bindings/fortran/roctx/   the same, built against the installed ROCTx
   roctx.F90                     generated: module roctx
   CMakeLists.txt                → libroctx_fortran.a + roctx.mod
   roctx-fortran-config.cmake.in find_package export → roc::roctx_fortran

rocm-libraries/  bindings/fortran/rocblas/   a standalone project, built against the installed rocBLAS
   rocblas.F90                   generated: module rocblas (one module)
   metadata/                     curated metadata
   CMakeLists.txt                → librocblas_fortran.a + rocblas.mod
   rocblas-fortran-config.cmake.in
   cmake/ROCmFortranCompiler.cmake   the amdflang probe
   cmake/toolchains/             example toolchain files
   test/                         Fortran tests (use hip: need the HIP binding)
   CHANGELOG.md, .gitattributes  the .F90 are marked linguist-generated
```

HIP's and ROCTx's bindings are outside `projects/hip/` and `projects/rocprofiler-sdk/`, like every binding outside its library's project, and either builds without the other.
hipFFT's binding directory holds two bindings and installs two packages from one build, `hipfft-fortran` (`hip::hipfft_fortran`, modules `hipfft` and `hipfftxt`) and `hipfftw-fortran` (`hip::hipfftw_fortran`, module `hipfftw`).
Its build takes `HIPFFT_` options for both, and a consumer of `hipfftw-fortran` can override its directory with `HIPFFTW_FORTRAN_COMPILER_DIR`.
`hipfft-fortran` also loads `hipfftw-fortran` when it is installed.
In `rocm-libraries`, each binding is owned in the monorepo's root `.github/CODEOWNERS` by the library's review team and the bindings team, `@ROCm/rocm-fortran-maintainers`.
In `rocm-systems`, the one `bindings/fortran/` entry is owned by the HIP reviewers and the bindings team.
Each binding is also declared in the monorepo's `.github/repos-config.json`, so that the monorepo's CI detects changes to it: one entry per library in the `bindings/fortran` category in `rocm-libraries`, and one `fortran` entry in the `bindings` category in `rocm-systems`.
A team needs write access to a repository for its CODEOWNERS entry to take effect, so the bindings team needs it in both monorepos.

A `.mod` only works with the compiler that wrote it, so artifacts install under a per-compiler subdirectory, one archive and one module set per library:

```text
<prefix>/
   <libdir>/fortran/<compiler>/        librocblas_fortran.a, libhip_fortran.a, libroctx_fortran.a, ...
   <includedir>/fortran/<compiler>/    rocblas.mod, hip.mod, roctx.mod, ...
   <datadir>/rocblas/fortran/          rocblas.F90          ← the generated source
   <datadir>/hip/fortran/              hip.F90
   <datadir>/roctx/fortran/            roctx.F90
   <libdir>/cmake/rocblas-fortran/     rocblas-fortran-config.cmake  → roc::rocblas_fortran
   <libdir>/cmake/hip-fortran/         hip-fortran-config.cmake      → hip::hip_fortran
   <libdir>/cmake/roctx-fortran/       roctx-fortran-config.cmake    → roc::roctx_fortran
```

`<libdir>`, `<includedir>` and `<datadir>` are the GNUInstallDirs values (`lib64` on Fedora, RHEL and SUSE in a standalone build, while TheRock and its packages always use `lib`).
A ROCm tarball unpacked in `/opt/rocm` gives `/opt/rocm/include/fortran/amdflang/` and `/opt/rocm/lib/fortran/amdflang/`.
The distro packages install under a versioned prefix, `/opt/rocm/core-<X.Y>`, so ROCm 10.4 installed from packages gives `/opt/rocm/core-10.4/include/fortran/amdflang/` and `/opt/rocm/core-10.4/lib/fortran/amdflang/`.

Every file of a binding installs in its development component (`devel` in `rocm-libraries`, and `dev` in `rocm-systems`, through `HIP_FORTRAN_INSTALL_COMPONENT`), and its tests in a `tests` component.
The proposed packaging puts each library family's bindings in their own `amdrocm-<family>-fortran-devel` package (see [Building and shipping a binding](#building-and-shipping-a-binding), the [Packaging in TheRock](#appendix-packaging-in-therock) appendix and [Decisions requested](#decisions-requested)).

`<compiler>` is the name a user types, derived from the compiler ID (`amdflang` for any LLVM Flang, `gfortran`, `ftn`, `ifx`, `ifort`, `nvfortran` or else the lowercased basename), so several compilers can share one prefix.
It ignores the compiler version, and `<LIB>_FORTRAN_COMPILER_DIR` (for example `ROCBLAS_FORTRAN_COMPILER_DIR=gfortran-13`, and `HIP_FORTRAN_COMPILER_DIR` and `ROCTX_FORTRAN_COMPILER_DIR` in `rocm-systems`) gives a second version of one compiler its own directory.
The array tiers produce `.mod` files of the same name that are not interchangeable: the precompiled `amdflang` set is the default tier, `assumed-shape`, and a site that wants another tier builds its own set under its own prefix or `<LIB>_FORTRAN_COMPILER_DIR`.
`find_package(rocblas-fortran)` resolves the subdirectory matching the consuming project's compiler, so CMake users never write these paths, and a non-CMake build passes them as `-I` and `-L`.

The generated `.F90` installs too, under `share/<lib>/fortran/` (`share/hipfft/fortran/` holds `hipfft.F90`, `hipfftxt.F90` and `hipfftw.F90`).
A user of another compiler builds the binding from it without cloning a monorepo, and the installed source always matches the installed library.
The user compiles it with `-DUSE_ASSUMED_SHAPE=1` for the `assumed-shape` tier, or `-DUSE_ASSUMED_RANK=1` for the `assumed-rank` tier.
Without a macro, the source yields the `none` tier.
The hand-written modules keep their installed sources (`include/rocblas/rocblas_module.f90`, `include/hipblas/hipblas_module.f90`, `include/rocsparse/rocsparse.f90` and `include/rocsparse/rocsparse_enums.f90`) until their removal, no earlier than 11.0.
The generated source does not reuse those paths, so a consumer that compiles them changes its path when it moves.

### Tests and documentation

hipfort's test suite splits across the libraries it covers, and each library's Fortran tests move to its binding's `test/`.
Each test is one Fortran program and its own CTest test, labeled `gpu` since it needs an AMD GPU.
A `<name>.f03` test calls the raw and typed forms and is registered as `<lib>_fortran_<name>`.
A `<name>.f08` test calls the array overloads and is registered as `<lib>_fortran_<name>_f2008`.
Tests are built under `BUILD_FORTRAN_CLIENTS`, and installed: the executables in `bin/`, and a `CTestTestfile.cmake` with the same tests and properties in `bin/<lib>-fortran/`, so that `ctest --test-dir` runs them from an install.
hipFFTW's tests are named `hipfftw_fortran_<name>` and install with hipFFT's, under `bin/hipfft-fortran/`.
The tests `use hip`, so without `hip::hip_fortran` a binding's whole test suite is skipped at configure time with a `STATUS` message, except rocRAND's and hipRAND's version tests.
rocRAND, hipRAND and the `rocm-systems` tests follow the same convention, with differences only where they must:

- rocRAND and hipRAND each add a vendored FRUIT suite, registered as one CTest test
- ROCTx's test and rocRAND's and hipRAND's version tests need no GPU
- a HIP test whose device lacks a feature exits with 77, which CTest reports as skipped (`SKIP_RETURN_CODE`)
- two HIP tests load a code object built with `hipcc`, which TheRock points at its device libraries through `HIP_FORTRAN_GENCO_FLAGS`

The regeneration check on a library's pull request compiles the regenerated modules without a GPU, which catches a header that no longer gives a binding that compiles.
TheRock's `fortran-bindings` stage compiles every binding's tests, and TheRock runs the installed tests on an AMD GPU, in one test job per binding subproject (see [Packaging in TheRock](#appendix-packaging-in-therock)).
A test that spans libraries, such as a rocSOLVER test that `use`s rocBLAS, follows the same dependency order as the bindings.

A Fortran routine has the same name, arguments and semantics as the C function it binds, so the library's C reference documents it.
Each library documents its Fortran API from its existing top-level `docs/`, stating how it maps onto the C one: module name, generic forms, link target.
Any docstrings live in the generated `.F90`, and a binding holds `test/` but no `docs/`: the content stays in the library's repository (P1), not in the documentation toolchain.
The headers' Doxygen comments can be emitted as `!>` docstrings, but turning that on for the packaged modules, and publishing a Fortran reference from them, is not decided yet.
If a Fortran reference is published, cross-references between libraries resolve through Doxygen tag files or Sphinx inventories, so the site is built in the aggregating doc job that already has every project's output.
The remaining mechanics, portal integration and the exact tag-file wiring, need the ROCm documentation team.

The ROCm documentation gets one Fortran entry point, a section under *ROCm HPC SDK* in the *Components* tab of the documentation portal.
It lists the libraries that ship a binding, the package that carries each binding, the interface variants (see [Language standard](#language-standard)), the compiler support, the installation of the packages and the runtime libraries a tarball install needs.
It also carries the status of hipfort and the migration guides, which move there from the hipfort documentation before hipfort is removed, no earlier than 11.0.
hipfort's tutorials move to the documentation of the matching library, and its install and how-to pages become this Fortran page.
Its API reference is not carried over.
For the tech preview, targeted for 10.3, the Fortran page is one short page on how to enable and build the bindings and on their limitations.
The full documentation comes with production, targeted for 10.4.

## Appendix: Packaging in TheRock

TheRock builds the bindings from a top-level `bindings/fortran/` tree of its own, apart from the libraries they wrap, and never inside the libraries' builds (P2).
The libraries' builds pass no Fortran option, except hipSOLVER's `EXPORT_FORTRAN_BINDINGS=OFF`, and need no Fortran compiler for the bindings.

TheRock enables the bindings in steps, each a change of its own that starts once the previous one has run without issues:

- **Build support.** TheRock can build the bindings, but `THEROCK_ENABLE_FORTRAN_BINDINGS` is off by default, even with `THEROCK_ENABLE_ALL`, so a local build turns them on explicitly.
- **CI.** TheRock's CI builds the `fortran-bindings` stage and runs one GPU test job per binding subproject, and nothing is shipped yet.
- **Releases.** The release builds enable the bindings and stop skipping the `fortran-bindings` stage and its test jobs, so the tarballs carry the bindings' `dev` components, and the packages of item 5 are declared once the product and release owners approve the packaging (see [Decisions requested](#decisions-requested)).
  How the release builds turn the bindings on, a CMake option in the release workflow or a per-branch flag, is TheRock's choice.

1. **Subprojects and artifacts.**
   Each binding directory is one subproject, `<lib>-fortran`, built from its monorepo's `bindings/fortran/<lib>/`: `hip-fortran` and `roctx-fortran` from `rocm-systems`, and one per `rocm-libraries` binding.
   They are grouped into one target-neutral artifact per library family: `runtime-fortran`, `blas-fortran`, `solver-fortran`, `sparse-fortran`, `fft-fortran` and `rand-fortran`.
   All are in the `FORTRAN_BINDINGS` feature group, so `THEROCK_ENABLE_FORTRAN_BINDINGS` turns every binding on or off.
   Each family also has its own feature: `THEROCK_ENABLE_RUNTIME_FORTRAN`, `THEROCK_ENABLE_BLAS_FORTRAN`, `THEROCK_ENABLE_SOLVER_FORTRAN`, `THEROCK_ENABLE_SPARSE_FORTRAN`, `THEROCK_ENABLE_FFT_FORTRAN` and `THEROCK_ENABLE_RAND_FORTRAN`.
   A binding absent from its monorepo is skipped and its artifact left empty, so TheRock can carry the subprojects before the bindings land.

1. **Build stage.**
   The bindings build in their own generic stage, `fortran-bindings`, after `math-libs`.
   The math libraries are built once per GPU architecture, and the stage builds the bindings once, against the generic part of the math artifacts.
   A generic stage that depends on per-architecture artifacts needs TheRock to accept a pre-built subproject with no AMDGPU target: the CI step selects no target for a subproject that is never configured.
   The bindings are host code, but some HIP binding tests build a GPU code object with hipcc, so the HIP and math-library subprojects take the test GPU targets, while ROCTx's, host-only, needs neither a HIP compiler nor a GPU target.
   The math-library binding tests `use hip`, so every math-library binding depends on `hip-fortran`, not on `roctx-fortran`, and rocSOLVER's also depends on rocBLAS's.
   The stage runs on Linux only, and a sanitizer build (`THEROCK_SANITIZER`) declares no binding in it (see [Non-goals](#non-goals)).

1. **One Fortran compiler: the `amdflang` TheRock builds.**
   TheRock builds flang on Linux, and `base/aux-overlay` exposes it as `bin/amdflang`, which ships in `amdrocm-llvm`.
   The bindings use the `amd-hip` toolchain, and their own probe picks the `amdflang` beside the toolchain's C compiler, so the shipped `.mod` files match the shipped compiler, version included.
   If the Fortran compiler is later built in a stage of its own, the bindings stage comes after it, and the subprojects can take `CMAKE_Fortran_COMPILER` from the toolchain instead of probing.

1. **Artifact descriptors.**
   The default `dev` component already collects `**/*.a`, `**/include/**` and `**/cmake/**`, so the `.mod` files, the archives and the config packages land in `dev` with no change.
   Each descriptor adds `share/<lib>/fortran/**`, where the generated sources install, as an optional entry.
   Its `test` component lists the test executables (`bin/<lib>_fortran_test_*`) and their CTest directory (`bin/<lib>-fortran/`).
   The Fortran entries the libraries' own artifacts carry, hipSOLVER's `lib/libhipsolver_fortran.so*` in `artifact-solver.toml` and hipBLAS's `lib/libhipblas_fortran.so` in `artifact-blas.toml`, already match nothing, since TheRock builds neither shared library.
   They go away with the hand-written modules, in the pull requests that remove them.

1. **Packages.**
   This item is a proposal for the releases step, pending the product and release owners (see [Decisions requested](#decisions-requested)).
   `build_tools/packaging/linux/package.json` declares the six `amdrocm-<family>-fortran-devel` packages of [Building and shipping a binding](#building-and-shipping-a-binding), each with its artifact's `dev` component and its dependencies.
   It enables `amdrocm-fortran`, declares it per-architecture (`Gfxarch`), and adds the six packages to its dependencies.

1. **CI.**
   Each monorepo declares its bindings in `.github/repos-config.json`, so its CI reports a change under `bindings/fortran/` as a changed project.
   TheRock maps `bindings/<name>/` paths like `projects/` and `shared/` ones, and every binding artifact belongs to the `fortran-bindings` stage, so such a change rebuilds that stage.
   The stage is part of TheRock's Linux multi-arch CI.
   `rocm-libraries`' CI does not reuse the stage from a baseline, since it depends on the math libraries it rebuilds.
   Until the releases step, TheRock's release workflow skips the `fortran-bindings` stage and its test jobs.
   `rocm-systems` and `rocm-libraries` run TheRock's CI from a pinned TheRock commit, so a binding pull request builds the stage only once that commit includes the CI step.
   The test jobs install a family's bindings and their tests from the CI artifacts (`install_rocm_from_artifacts.py --fortran-bindings`).

1. **Tests.**
   Each binding installs its tests, and they ship in the `test` component of its family's artifact.
   TheRock runs them in one test job per binding subproject, twelve in all: `hip-fortran`, `roctx-fortran`, and one per `rocm-libraries` binding, hipFFT's covering hipFFTW.
   Each job installs its family's bindings and the C libraries their tests load, and runs that binding's installed CTest suite.
   The jobs are skipped in sanitizer builds.
   Once the CI step lands, the bindings' test jobs are to be required, like every component's tests in TheRock (see [Decisions requested](#decisions-requested)).
   They test the committed binding against the library the pull request builds, so they pass as long as the C API only grows, which is the rule within a major release.
   A new function is simply not bound until the bot regenerates the binding, and that regeneration is never a condition for merging.
   A binding test fails only when a change breaks the C API or ABI: within a major release that is a bug the test catches, and at a major release, the pull request that breaks the API regenerates the binding or asks the bindings team to do so.
   Consumer-based test selection runs a changed subproject's tests and those of its direct consumers, and a binding consumes the library it binds, so a change to HIP, ROCTx or a math library also runs its binding's tests.
   That selection reads TheRock's committed consumer graph, which its drift check regenerates with `THEROCK_ENABLE_ALL` and therefore without the bindings.
   A follow-up change, once the pins contain the bindings, turns the bindings on in that check and commits the regenerated graph.
   The math-library binding tests `use hip`, so they run only once TheRock's `rocm-systems` pin contains the HIP binding (see [Rollout and success criteria](#rollout-and-success-criteria) for the merge order).
   While the monorepo pins do not contain the bindings, each test job warns that no tests are installed and passes.
   That is to become an error once both pins contain the bindings, so that a missing or empty test set then fails the job.

So until the releases step, neither the nightly nor the release tarballs carry the bindings.
Until then, a user tries a binding by building it from its monorepo against a nightly install.
The releases step puts the bindings' `dev` components in the tarballs, and, if the packaging is approved, declares the packages of item 5.
Whether the tech preview, targeted for 10.3, ships the bindings in the tarballs only or also in the OS packages depends on when that step lands.

## Revision History

- 2026-07-20: Alexis Montoison: Initial draft.
- 2026-10-14: Alexis Montoison: Merged as a draft, with each binding in a `fortran/` directory of its library's project, built by that project.
- 2026-10-22: Alexis Montoison: Each binding's CODEOWNERS shared with its library's team; one section per procedure in each archive; Fortran 2008 tests; the distro package that ships each binding.
- 2026-10-27: Alexis Montoison: Packaging per library family after review; `rocm-fortran-generator` is public.
- 2026-11-03: Alexis Montoison: Restructured. The bindings move to `bindings/fortran/` as standalone projects, built by TheRock in a stage after the libraries they wrap, with their tests in TheRock's test jobs; only sanitizer builds skip them; per-library build options dropped; status back to draft.
- 2026-11-07: Alexis Montoison: hipFFT's binding split into `hipfft` and `hipfftxt`; each CMake package installs in `lib/cmake/<lib>-fortran`.
- 2026-11-12: Alexis Montoison: Brought in line with the implementation: staged TheRock rollout (off by default, then CI, then releases, which skip the stage until then); required binding test jobs; the regeneration workflows as merged in both monorepos; merge order; hand-written modules deprecated at 10.3 and removed no earlier than 11.0; packaging, release gate, review target and compatibility promise written as proposals, collected in Decisions requested; the generic stage on per-architecture artifacts; the `-Wl,--gc-sections` and OpenMP offload risk, with `-Wl,-z,nostart-stop-gc`; a header workaround only for `hipfftXt.h`; HIP and ROCTx in separate projects, `bindings/fortran/hip/` and `bindings/fortran/roctx/`, with twelve TheRock test jobs; review fallback after ten business days; one approval from either team, `metadata/` included.
