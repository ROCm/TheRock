---
author: Alexis Montoison (amontoison)
created: 2026-07-20
modified: 2026-09-09
status: draft
---

# Distributing Fortran bindings across rocm-systems and rocm-libraries

## Overview

`hipfort` provides the Fortran `bind(C)` interfaces to the HIP runtime and the ROCm math libraries.
Today it lives in a separate GitHub repository, and the interfaces are partially written by hand.
Coverage is only ever as current as the last manual update: it drifts, and it stalls when no one maintains it.

This RFC co-locates the generated bindings in the same repository and project as the C headers they mirror.
They live in a dedicated `fortran/` directory, not in the `include/` tree, so a change to a `.h` can flow to its `.F90` in the same repository.
The HIP runtime bindings go to `rocm-systems`, next to HIP.
Each math library's bindings go to `rocm-libraries`, next to that project.
The generator (`rocm-fortran`) is a standalone build-time tool, in its own repository.
CI keeps everything in sync: when a C header changes, it regenerates the affected module and suggests the diff on the PR, like `clang-format`.

For ROCm 7.14 / 10.0 / 10.1, the goal is narrow: repair the existing hipfort without breaking anything, bringing its bindings current with the headers (they lag by one release).
The real change lands in ROCm 10.2: packaging returns, and the bindings split across `rocm-systems` and `rocm-libraries`.
That is when hipfort, under its current name and layout, goes away, replaced by the co-located, generated bindings this RFC describes.
This document fixes the direction, not a date, and is the source of truth for the design.

## Background

hipfort is a thin Fortran-to-C translation layer.
Its modules declare `interface ... bind(C)` against the ROCm C symbols, plus a few ergonomic wrappers.
It holds no GPU compute.

Everything in this section is the current situation, the hipfort of today.
The RFC changes or removes much of it (the separate repository, the `hipfort_` prefix, the split modules, and the CUDA backend); the roadmap says when.

```
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
| `hipfort_roctx.F90`, `hipfort_cuda_errors.F90`                 | profiling, CUDA-backend errors       |

After `make install`, under `$CMAKE_INSTALL_PREFIX`:

```
├── lib/fortran/<compiler> (HIPFORT_MULTITOOLCHAIN_LAYOUT=ON)
│   ├── libhipfort-amdgcn.a              ← AMD backend  (default)
│   └── libhipfort-nvptx.a               ← NVIDIA backend (deprecated; compat track only)
├── lib/cmake/hipfort/                   ← find_package(hipfort): hipfort::rocblas ...
├── include/fortran/<compiler>/hipfort/{amdgcn,nvptx}/*.mod ← read on `use hipfort...`
```

The install layout has a toolchain dimension, because a `.mod` is specific to the compiler, and even the compiler version, that produced it.
Two Fortran compilers cannot share one `.mod`.
`HIPFORT_MULTITOOLCHAIN_LAYOUT` selects the layout.
ON, the default, installs per-compiler, under a `<compiler>` subdirectory (for example `lib/fortran/<compiler>/`), so several toolchains coexist on the same prefix (the diagram above shows this case).
OFF installs a single set for one compiler: one `.mod` tree and the `.a` under `lib/`, which works for one compiler but a second one would conflict.
The backend axis (`amdgcn` vs `nvptx`) is separate, because the CUDA switch also produces a different `.mod`.
`find_package(hipfort)` resolves to the subdirectory matching the consuming project's compiler and AMD backend, and a presence check reports which pieces are installed for the user's compiler, so a user can confirm the Fortran extension is available before building.
(Under the split this becomes a per-library check with per-library targets (`find_package(rocblas)`, `roc::rocblas_fortran`), described in Building the bindings; this section is about today's hipfort.)

There is no portable Fortran ABI across compilers, which is the root of all this: `.mod` and compiled `.a` are compiler- and version-specific, so the only stable ABI is the C one, the vendor `.so`, which the `bind(C)` layer rides on.
That constraint drives the whole packaging design (ship source, plus a precompiled set for `amdflang`), set out in full in the appendix on `.mod`/`.a` incompatibility.

Both `.mod` and `.a` are needed, for different phases.

```
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
   your_app.o ── unresolved symbols ──► libhipfort-amdgcn.a  (rocblas_dgemm_, rocblas_dgemm_rank_)
                                       ► librocblas.so       (rocblas_dgemm)
                                       ► libamdhip64.so      (HIP runtime)
```

The `.mod` is a compile-time artifact.
It carries declarations.
The `.a` is a link-time artifact and it carries compiled bodies.
A math-library call has two forms.
The raw `bind(C)` interface (`rocblas_dgemm_`) is a pure declaration: it sits in the `.mod`, adds nothing to the `.a`, and its symbol is resolved from the vendor `.so`.
The ergonomic variants (`rocblas_dgemm_rank_` taking Fortran arrays) are real wrappers: they call `c_loc` and forward to the raw form, so their bodies are compiled into the `.a`.
These wrappers are numerous, around 1900 for rocBLAS alone, alongside the HIP runtime wrappers (`hipMalloc`, `hipMemcpy`, `hipCheck`).
So the `.a` is not just the memory wrappers: every math library with native or typed variants contributes bodies to it.
Both the `.mod` and the `.a` must ship.

**Why a static archive, not a shared library.**
The `.a` holds thin forwarding shims (each wrapper does `c_loc` and calls the raw `bind(C)` form), so a shared library buys nothing here: there is nothing substantial to share in memory; the caller is bound to the module at compile time through the `.mod`, so changing a wrapper means recompiling anyway; and a prebuilt `.so` is compiler- and version-locked exactly like a `.mod`.
Against those non-benefits, a `.so` only adds a runtime dependency and a PLT indirection that blocks inlining a one-line shim.

**Fusing with the vendor library.**
We keep the Fortran `.a` separate from the vendor `.so` rather than fusing them: fusing would tie the binding's version to the vendor release and make that `.so` carry Fortran symbols, for no gain (the wrapper object is well under half a percent of `librocblas.so`).
A separate static `.a` is also lazy at the member level, so unused wrappers are never pulled in, and there is no C-side clash, since the C and Fortran objects compile in different directories.

**One source set, two backends.**
The same sources compile twice under the CUDA switch (`-DUSE_CUDA_NAMES`), producing a per-backend archive and `.mod` (`libhipfort-amdgcn.a` for AMD, `libhipfort-nvptx.a` for NVIDIA); a compiled `.mod`/`.o` is backend-specific, so the two cannot share one archive.
The packaged track keeps only the AMD backend.

The hand-written model lags for a simple reason.
A new function, enum, or struct field in a header must be copied by hand, later, by someone who is not the header's author.
The result is delay and silent drift, a missing `_64` variant, a stale struct layout.
It stops entirely when there is no maintainer.

Worse, the hand-written work is already duplicated.
Several math libraries ship their own Fortran module inside their include tree.
rocBLAS ships `rocblas_module.f90`, hipBLAS ships `hipblas_module.f90`, rocSPARSE ships `rocsparse.f90` (over 7000 lines) plus `rocsparse_enums.f90`, and hipSPARSE and hipSOLVER ship `hipsparse.f90` and `hipsolver_module.f90`.
Five of these define a module named exactly after the library (`module rocblas`, `hipblas`, `rocsparse`, `hipsparse`, `hipsolver`), and four install it as a public file, so the packaged track's unprefixed modules supersede and remove them rather than coexist (see What changes for a library repository).
Each is maintained independently of hipfort's binding for the same library.
So `rocblas_dgemm` has two separate Fortran interfaces, in two repositories, kept current by two different people.
Either copy can drift from the C header and from the other.
Co-located generation collapses these parallel copies into one generated source of truth per library, next to the header they both used to track by hand.

rocSPARSE goes further.
It also ships fourteen Fortran example programs (`clients/samples/example_fortran_*.f90`) written against its own module.
Yet its online documentation is entirely C/C++ and surfaces none of them, pointing Fortran users to hipfort instead.
So the same library carries a second Fortran binding and a second example set, maintained apart from hipfort and invisible in its own docs.
Generating the binding in place, with the examples and docstrings alongside it, folds that duplicated, undiscoverable work back into the library.

## Design principles

**P1. Co-locate bindings with their headers.** The binding for `rocblas_dgemm` should live in the same repo, and change in the same PR, as `rocblas.h`.
Proximity removes the lag.
The binding regenerates when, and where, the C API changes.

**P2. Keep packaging separate.** The split changes how packaging works, not what ships.
The deliverable is per-library: each library ships its own `.F90`, `.mod`, `lib<lib>_fortran.a`, and CMake target (`roc::rocblas_fortran`, `hip::hip_fortran`, ...).
Fusing the pieces into one package, or building only a subset, stays possible, but the default is per-library.
What moves is integration, not packaging: the sources move next to the headers, the packaging job stays put.
That split fixes the sync problem without disrupting the deliverable.

**P3. Keep the generator standalone.**
It is build-time tooling, not a runtime dependency.
It lives in its own repository (`rocm-fortran`), which is internal today; where it ultimately belongs, and whether it opens up, are recorded in Open questions.
What this RFC does commit to is that its *output* is committed, reviewable Fortran in the library repositories, so reading and building the bindings never depends on reaching the generator.
Wherever it lives, it is owned and maintained by the Fortran-bindings team, not the compiler developers, and it is not part of the compiler build.
Library repositories invoke it as a standalone tool.
They do not vendor it.

**P4. CI maintains the generated surface, not hand-copying.** A C-library developer should not have to learn Fortran to keep a binding current.
When a header changes, CI regenerates the `.F90` and proposes the diff as a suggested commit, like an auto-formatter, so the generated surface is never stale after merge.
The one manual part is the curated metadata: most changes need none, and when a new function needs a classification the CI flags it for the Fortran-bindings team, who add the one line.

## Non-goals

To keep the scope sharp, the following are explicitly out of scope for this RFC.

- **Windows.** The target is Linux HPC. ROCm's existing Fortran clients are already gated off on Windows in-tree (rocBLAS, hipBLAS, hipSOLVER, and rocSPARSE all condition Fortran on `NOT WIN32` / `UNIX`), so a Windows Fortran product is not a goal.
- **Fortran under sanitizers.** The ASAN and coverage build variants ship no Fortran, since Fortran linking is unsupported under ROCm's sanitizer configuration.
- **Fortran 77, and any consumer below Fortran 2003.** The bindings are modules built on `iso_c_binding`, which is F2003, so a translation unit compiled below Fortran 90 has no `use` statement and cannot consume them (see Language standard). This is a property of the standard level, not of the source form: fixed-form sources compiled by a current compiler consume the bindings normally.
- **The generator as a runtime or shipped product.** `rocm-fortran` is build-time tooling; no Julia, Clang.jl, or generator code reaches a consumer, the compiler build, or the deliverable.
- **Non-CMake consumer tooling as a first-class path.** CMake `find_package` is the supported integration; a `pkg-config` file or a flags script is a convenience, not a guarantee.
- **A new build system or test framework.** This RFC defines where the bindings live and how they stay in sync, reusing the monorepo's existing CMake, CODEOWNERS, and CI substrate rather than introducing a parallel one.

## The generated bindings

The generated modules are what you `use` and link.
This section describes their surface: the module names, how you call them, the Fortran standard they need, and the CUDA option.
The generator that produces them is described next.

### Naming and module structure

The compatibility track keeps today's names and layout: the `hipfort_` prefix and the split into separate interface, type, and enum modules per library.
So `use hipfort_rocblas` keeps working, and the prefix avoids clashing with the vendor's own `rocblas` module.
The packaged track modernizes both.
It drops the prefix, matching NVIDIA's convention (`use cublas`, not `nvfort_cublas`).
This does not clash with the vendor's own same-named modules (`rocblas_module.f90`, `rocsparse.f90`, ...): under the split those shipped modules are superseded and removed, so the generated module is the only `rocblas` in the tree.
And it collapses each library's binding into a single Fortran module, instead of the separate interfaces / types / enums modules.
The HIP runtime module becomes `hip` (so `use hip`, not `use hipfort`), and the cross-cutting pieces fold in rather than staying separate modules: `hipfort_check`, `hipfort_handles`, and `hipfort_auxiliary` disappear, their symbols coming from the library module you already `use` (`hipCheck` and the HIP handles from `hip`, `rocblasCheck` and `rocblas_handle` from `rocblas`).
Cross-library sharing still works: a library that needs another's types simply `use`s that library's module (rocSOLVER uses rocBLAS).
One generator emits both layouts: the packaged, collapsed, unprefixed set committed next to the headers, and the split, `hipfort_`-prefixed compatibility set emitted into `ROCm/hipfort`. The two schemes coexist at no extra cost.

### Calling styles

One rule, shaped by M. Klemm, B. Cornille, M. Kurz, and J. Potyka.
Each entry point is a single generic over its public name.
The compiler picks the form from the actual arguments, so callers never touch `iso_c_binding`.

- Raw `type(c_ptr)`: today's hipfort surface, and what OpenMP target data needs.
  Existing code keeps working, nothing breaks.
- Ergonomic forms under the same name: a plain Fortran array, a typed handle, or a normal string just work.

The ergonomic array forms carry a contract, because they take a Fortran array and forward it with `c_loc`:

- The actual argument must be **contiguous** and have the `TARGET` attribute; otherwise the processor may pass a temporary.
- The rocBLAS/rocSPARSE math APIs are **stream-ordered** (they enqueue and return before the device is done), so for any argument the device reads or writes after the call returns, the **raw `type(c_ptr)` surface is canonical**: a temporary would dangle past the call.
- The generator therefore never emits an ergonomic wrapper that passes `c_loc` of wrapper-local storage (a materialized shape or stride array, a null-terminated string copy) for an argument the device touches after return.

For a device-resident array (`hipMalloc` + `c_f_pointer`) a non-contiguous section is worse than a lifetime bug: packing it is a host read of device memory, which faults on a non-XNACK configuration, so contiguity there is a correctness requirement, not a performance one.

### Language standard

The floor is Fortran 2003, and two features raise it in well-defined steps:

- **Fortran 2003 (`iso_c_binding`)** is the minimum: `bind(C)`, `type(c_ptr)`, `c_loc`, `VALUE`, interoperable derived types, and `enum, bind(c)`. The raw `type(c_ptr)` interface surface needs nothing more, so a site restricted to F2003 can use it as-is.
- **Fortran 2008** is needed for the ergonomic array forms: assumed-shape array dummies and passing a Fortran array through `c_loc`.
- **Fortran 2018** is needed only for the opt-in assumed-rank variants (`-DUSE_ASSUMED_RANK=ON`, `dimension(..)`).

So the raw surface is F2003, the array ergonomics are F2008, and assumed-rank is F2018.
Minimum compilers: gfortran, amdflang, and Cray CCE.
See the appendix for why a `.mod` cannot be shared across compilers, or even across versions of the same compiler.

Fortran 77 is therefore out of scope for the bindings: `iso_c_binding` is F2003, and below Fortran 90 the language has no modules, so there is no `use` statement to write and nothing to bind to.
Note that this is a property of the standard level and not of the source form: a fixed-form source compiled by a current compiler consumes the bindings like any other caller.
Reaching the API from below Fortran 90 would need a separate artifact, a C shim library with F77 linkage plus an include file for the constants.
That is not planned, and would be generated for a specific set of entry points if industrial demand appeared.

### CUDA backend

The generated modules are `.F90` (uppercase), so the compiler runs the C preprocessor and resolves the CUDA switch.
Compiled with `-DUSE_CUDA_NAMES`, the same sources bind to CUDA instead of HIP.
The generator cannot infer CUDA names from the AMD headers, so hand-written data files feed the `#ifdef` branches.
The main one is a per-library `hipName cuName` map.

The NVIDIA backend lives only in the compatibility track (`ROCm/hipfort`),
kept there for HIP-versus-CUDA comparison until ROCm 11.0.
The packaged track does not carry it: not co-located, not packaged, not shipped in ROCm.

## The generator (rocm-fortran)

`rocm-fortran` regenerates the whole binding set from the current C headers with [Clang.jl](https://github.com/JuliaInterop/Clang.jl).
It keeps hipfort's module, enum, and check-routine names, so existing hipfort code compiles unchanged.
Using the modules needs no Julia.
Only regenerating does.

```
   Header sources
     rocm-libraries/<lib>/include    (latest API headers)
     /opt/rocm/include               (HIP runtime + fallback headers)
            │  parse as C  (clang -x c, via Clang.jl)
            ▼
   Generator (Julia)
     generate_all.jl  →  common/fortran_gen.jl, common/hip_generics.jl,
                         <lib>/generator.jl + <lib>.toml
            │  emit
            ▼
   fortran/  (committed, packaged track: one module per library, no prefix)
     hip.F90 (rocm-systems), rocblas.F90, hipblas.F90, rocsparse.F90, hipsparse.F90,
     rocfft.F90, hipfft.F90, hipfftw.F90, rocsolver.F90, hipsolver.F90,
     rocrand.F90, hiprand.F90
            →  hip runtime + 11 libraries, ~5800 functions
```

The engine parses each header as C and emits, per library, the interface block, the enum and `#define` constants, the derived types, and the docstrings.
Each library runs in its own process.

Profiling (`roctx`) is deferred: for now it is not migrated into `rocm-systems`, because rocTX is being refactored upstream into rocprofiler-sdk and its home is not yet settled. It stays in the compatibility track until that lands.

Regeneration is cheap.
That is what makes the CI-as-formatter model practical, and it means a release's final headers can be picked up at the last moment, not weeks ahead.
The generator is also deterministic: the same headers and metadata produce byte-identical output.

The generator is about 6.6k lines of Julia.
Only the emit stage is Fortran-specific.
The front end, which parses headers with Clang.jl into a language-agnostic description of functions, enums, and structs, could be retargeted to another language by swapping the emitter.

### Free side-effect: C-header hygiene

The generator parses every public header as C (`clang -x c`), because a `bind(C)` interface mirrors the C ABI.
So "the headers are valid C" is a hard prerequisite, checked for free on every run.
ROCm headers are meant to be C-includable, but in practice they are only pulled from C++.
C-only breakage therefore hides for releases.
Typical cases are a C++ keyword used as an identifier, a missing `extern "C"`, or a C++-only include.
The generator fails on such a header, and the CI hook flags the PR that introduced it.
This is value for the C libraries themselves.
Several such issues were already caught this way.

### Ambiguous or unmappable C

Not every valid C construct maps cleanly to a `bind(C)` interface.
Where a construct is ambiguous or has no Fortran equivalent, the generator falls back to a defined, ABI-correct form rather than dropping it or guessing.

- **Bit-fields, unions, anonymous members.** A struct that is not field-mappable is emitted as an opaque byte blob of the exact C size and alignment.
  Structs that embed it keep their field offsets, so nothing downstream breaks.
  Its own members are simply not individually accessible.
  This is how `hipDeviceProp_t` survives, even though it embeds the bit-field struct `hipDeviceArch_t`.
  A struct with an incomplete or zero-size layout is genuinely skipped.
- **Opaque handles.** A pointer to an incomplete struct, such as `rocblas_handle`, becomes a distinct thin derived type, or `type(c_ptr)` when typed handles are off.
- **Opaque workspace pointers.** A typed pointer that is really scratch memory is forced to `type(c_ptr)` by value, not a native array buffer.
- **C strings.** `char*` plays three roles that the C type cannot distinguish, so each is handled separately rather than with one raw binding.
  An input string (`const char*`) is kept out of the raw `bind(C)` block and given a Fortran wrapper that null-terminates.
  A `const char*` return value (the status-to-string helpers) is exposed as a Fortran function returning a string.
  A caller-supplied output buffer is bound with its length argument so the wrapper can copy back and trim; where the C API documents no length, it stays a raw binding rather than a wrapper that cannot bound the library's write.
- **Null pointers as a mode selector.** Across rocSPARSE, rocBLAS, and rocSOLVER, `NULL` is sometimes not an error but a mode selector (the buffer-size query/execute idiom: call once with `nullptr` to get the size, again with the buffer).
  A Fortran array dummy has no absent state, and `OPTIONAL` on the raw surface is illegal (F2018 constraint C865 forbids `OPTIONAL` with `VALUE`, and every raw pointer is passed by value).
  So a `<lib>.nullable` metadata category records which parameters accept `NULL` and what it selects; optionality lives only in the non-`bind(C)` wrapper layer (where `OPTIONAL` is legal and the wrapper passes `c_null_ptr` for an absent argument), and query/execute pairs are generated as two named entry points rather than one optional-argument generic.
- **Complex numbers.** A rocBLAS or rocSPARSE complex is a two-float C struct, which the C type system does not distinguish from any other two-float struct, so mapping it to a Fortran intrinsic `complex` is a curated fact, not something inferred.
  It is an ABI assumption (the struct is passed the same way as `complex(c_float_complex)` on the supported platforms), checked by the size-and-alignment gate; the alternative, a `bind(C)` derived type, would break every existing user and the shipped examples that pass a plain `complex`.
- **Macros and vector types.** Function-like and internal (`__`-prefixed) macros are skipped, while `#define` constants are extracted as parameters.
  The low-level HIP vector types (`float4`, `int4`, and the rest) are skipped, and functions taking them stay `type(c_ptr)`.

Two ambiguities the C type system cannot resolve on its own are left to the curated metadata (see Curated metadata): a scalar host value versus a data buffer (`const double* alpha` versus `double* x`, identical in C), and device-side output scalars.
Invalid C, as opposed to merely ambiguous C, is a separate case: the generator fails on it (see C-header hygiene).

### Curated metadata (the manual surface)

The generator emits everything the C ABI carries: signatures, enums, structs, and docstrings.
A few Fortran-side and CUDA-side facts are not present in the C headers and cannot be inferred.
They live in small per-library files that a maintainer curates.
This is the honest counterpart to "the generator removes the hand-copying": the bindings are generated, but this thin layer of per-argument intent stays manual, because it encodes information the C API does not.

| File                   | What it encodes                                                                          | Why the headers cannot give it                                        |
| ---------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `<lib>.argkinds`       | which pointer args are scalar host values (`alpha`, `beta`) vs data buffers              | `const double* alpha` and `double* x` are identical in C              |
| `<lib>.cptrargs`       | device output scalars kept as pass-through `type(c_ptr)` (`info`, a dot `result`)        | pointer-mode outputs live on the device, so arrayifying them is wrong |
| `<lib>.native`         | which functions get the array (`_rank_*`) overloads, and the max rank                    | rank is a Fortran-side choice, absent from the ABI                    |
| `<lib>.txt`            | the `hipName cuName` map for the CUDA backend                                            | AMD headers hold no CUDA names                                        |
| `<lib>_enums_cuda.txt` | enum values where CUDA differs from HIP                                                  | two different libraries                                               |
| `<lib>.toml`           | per-library generator config (name prefix, opaque handles, own headers, check overrides) | project conventions, not header facts                                 |

In the packaged track, only four of these are maintained: `argkinds`, `cptrargs`, `native`, and `toml`.
The CUDA map files (`<lib>.txt`, `<lib>_enums_cuda.txt`, and the per-type `.cuda.finc`) belong to the compatibility track only, since the packaged track has no CUDA backend.

The maintenance cost is bounded, because these files are bootstrapped and tool-validated, not written from scratch:

- A generator tool derives every `.argkinds` from the original hipfort sources in one run.
  The `.txt` and `.cuda.finc` files were seeded from hipfort's `#ifdef USE_CUDA_NAMES` blocks.
- `audit_cuda_mappings.jl` validates every emitted `cu*` name against a checked-in CUDA symbol dump (offline, no CUDA install) and drops the ones that do not exist.
  `apply_argkinds` warns about scalar-looking args that are still unclassified.

The residual manual work is small, and it is maintenance rather than authoring.
A new ROCm function whose scalar pointer argument the bootstrap did not cover must be classified by hand (the warning points at it).

**Keeping the classifications correct.**
The `apply_argkinds` warning catches an *unclassified* argument, but not a *stale* one: an entry that stays well-formed after a function is renamed or its signature changes.
So the classification files carry a machine-readable schema keyed to function identity and parameter position and type, and the generator validates every entry against the parsed C declaration on each run: an entry that matches no current function or parameter **fails generation**, it does not merely warn.
That turns "nothing merges *unclassified*" into "nothing merges *unmatched*".
The one case a schema still cannot catch (an entry that matches but is wrong) is covered two ways: a one-time audit of the classifications bootstrapped from hipfort, and runtime tests exercising **both host-pointer and device-pointer modes** for a representative set of entry points, prioritizing the pointer-mode-ambiguous families (`alpha`/`beta`/`result`), since those are the ones a wrong classification turns into a fault or a silent wrong answer.

### Testing the generator

The generator is validated at three levels: unit tests over its pure transforms; output goldens plus the CI compile gates (the committed modules must compile under amdflang and gfortran at `-std=f2018` and survive the Doxygen pass); and a runtime run of the bindings on an AMD GPU.
That runtime run must cover both pointer modes for the `argkinds`-classified entry points, since a misclassified scalar pointer compiles and links cleanly and no compile-only gate can see it (see Keeping the classifications correct).
The full test inventory (file names, gates, and the end-to-end golden) is in the appendix.

## Proposal

```
   BEFORE  (super-repo, always downstream)
     rocm-systems (HIP headers) and rocm-libraries (rocBLAS, ...)
            │  headers copied downstream
            ▼
     hipfort  (separate repo, updated by hand, stalls when unmaintained)
            │
            ▼
     packaging

   AFTER  (co-located sources, separate packaging)
     rocm-systems/         .../hip/*.h          + fortran/hip.F90
     rocm-libraries/  projects/rocblas/*.h      + fortran/rocblas.F90
                      projects/hipfft/*.h       + fortran/hipfft.F90
            │  the generator regenerates each .F90 in the SAME PR as its header
            ▼
     packaging (separate job)
            collect *.F90  →  per-library *.mod + lib<lib>_fortran.a
                           →  per-library targets (roc::rocblas_fortran, hip::hip_fortran, ...)
            fuse into one package or ship a subset (P2, optional)
```

Where each piece goes:

| Piece                                                                            | Repo                                                 | Why                                                  |
| :------------------------------------------------------------------------------- | :--------------------------------------------------- | :--------------------------------------------------- |
| HIP runtime module `hip` (absorbing HIP types, enums, check, auxiliary, handles) | `rocm-systems`                                       | HIP lives here (P1)                                  |
| Each math library's `<lib>` module (`rocblas`, `hipfft`, ...)                    | `rocm-libraries`, by project                         | next to its C headers (P1)                           |
| Curated metadata (`argkinds`, `cptrargs`, `native`, `.toml`)                     | with its library (`rocm-systems` / `rocm-libraries`) | manual, but moves with the header (P1)               |
| CUDA maps (`<lib>.txt`, `<lib>_enums_cuda.txt`, `*.cuda.finc`)                   | compatibility track (`ROCm/hipfort`)                 | CUDA is not in the packaged track (see CUDA backend) |
| The generator (`rocm-fortran`)                                                   | its own repository                                   | build-time Fortran tooling (P3)                      |
| Packaging (per-library `.mod`/`lib<lib>_fortran.a`, CMake targets)               | separate job                                         | per-library deliverable; optional fused package (P2) |

The CI hook runs the generator on the changed headers only:

```
   Developer opens a PR editing  projects/rocblas/.../rocblas.h
            │
            ▼
   CI hook: run the generator on the changed header(s) only
            │
            ├─ generated .F90 unchanged  → OK, check passes, nothing to do
            │
            ├─ generated .F90 differs    → bot pushes a SUGGESTED CHANGE:
            │                                the regenerated rocblas.F90 diff
            │                                (same UX as clang-format-diff)
            │                                dev clicks "commit suggestion" → in sync
            │
            └─ header no longer valid C   → FAIL, generation stops and flags the
                                             C-header regression (see C-header hygiene)
```

The hook is path-filtered.
It runs only when a PR touches a library's public C headers (`*.h`, the files the generator parses) or its `fortran/` directory (the generated binding and the curated metadata), so unrelated PRs never pay for it.
The `.hpp` C++ headers are not parsed, so they do not trigger it.
Some changes reach beyond one library: a shared header, such as the HIP runtime or the rocBLAS headers that rocSOLVER includes, can shift other libraries' bindings, and a change to the generator itself can shift all of them.
A nightly full regeneration backstops the per-PR hook and catches that cross-header drift.

This rolls out in two stages.
First advisory, where the bot suggests and merging is not blocked.
Then enforcing, a required `bindings-in-sync` check, like a `clang-format` gate, that fails a header change missing its regenerated binding.

### Maintaining the bindings under the split

A binding is more than its `.F90`, and the split keeps all of it next to the library.

- **Generated `.F90`.** Derived from the header.
  The CI hook regenerates it and suggests the diff, and no one edits it by hand (P4).
- **Curated metadata.** The manual part, the intent the C API cannot express (see Curated metadata).
  Co-located in the same `fortran/` directory, so a header change carries its metadata tweak in the same PR (P1).
  Most header changes need none, and when a new function's scalar pointer argument is still unclassified the generator's checks flag it on the PR, so an unclassified argument cannot merge silently.
  This catches a missing classification rather than a stale one: an entry that still matches its function after a rename or signature change is not re-verified by this check, so correctness of the existing classifications rests on the runtime tests and CODEOWNERS review of the metadata, not on this gate.
- **Tests.** A library's Fortran tests live in its `fortran/test/`.
  The per-PR hook compile-checks them without a GPU; the packaging or validation job runs them on an AMD host.
- **Docs and examples.** Docstrings are generated in place with each binding.
  The published doc site and the examples are assembled by the packaging job from every project, so hipfort keeps one doc set and one example set despite the split.
  The current hipfort documentation and tests can likewise be split across the libraries they cover.

The packaging job runs nightly and on release branches.
It collects the committed `fortran/*.F90` from `rocm-systems` and each `rocm-libraries` project and compiles them in dependency order for the AMD backend.
It produces, per library, a `.mod` set and a static archive `lib<lib>_fortran.a` (`librocblas_fortran.a`, `libhip_fortran.a`, ...) with its CMake target (`roc::rocblas_fortran`, `hip::hip_fortran`, ...), installed under `lib/fortran/<compiler>/` and `include/fortran/<compiler>/`.
The deliverable is per-library: `find_package(rocblas)` then gives the C library and its Fortran binding together.
Fusing everything into one package, or shipping only a subset, is an option (P2), not the default.
The packaged track does not build the nvptx backend (see CUDA backend); that backend stays source-only in the compatibility track until ROCm 11.0.
Because the package has no CUDA backend, the `.mod` files land directly in `include/fortran/<compiler>/`, without the compatibility track's inner `hipfort/<backend>/` nesting.

### Building the bindings

The Fortran sources live in each library's `fortran/` directory, and a dedicated, self-contained `fortran/CMakeLists.txt` builds them.
It does `enable_language(Fortran)`, compiles the modules, and produces the `.mod` and `.a`.
It requires a recent CMake: the `rocm-libraries` projects sit at 3.16 today (and `next-cmake` at 3.25.2), which already covers Fortran module dependency ordering and `Fortran_MODULE_DIRECTORY`; a build using the Ninja generator wants CMake >= 3.20 for reliable Fortran module dyndep.
Every `rocm-systems` and `rocm-libraries` project exposes one uniform switch, `BUILD_FORTRAN_BINDINGS`, **ON by default but guarded**: it builds the bindings when a Fortran compiler is present, and silently skips them when one is not, so a C-only site with no Fortran compiler still configures and builds.
The guard reuses the probe the monorepo root already runs (`check_language(Fortran)` sets `ROCM_LIBS_HAVE_FORTRAN` in `rocm-libraries/CMakeLists.txt`), and a standalone per-project build falls back to its own `check_language(Fortran)`.
When the switch is ON and a compiler exists, the project calls `enable_language(Fortran)` and `add_subdirectory(fortran)`; otherwise it prints a `STATUS` line and moves on, never a fatal error.
That is how ON-by-default stays compatible with P4: no C build is ever *forced* to acquire a Fortran compiler, it just builds the bindings wherever it already can, and a C developer who has a Fortran compiler but does not want them passes `-DBUILD_FORTRAN_BINDINGS=OFF`.
To keep the twelve integrations identical rather than copy-pasted, the guard, `enable_language`, and `add_subdirectory(fortran)` live in one shared CMake helper (a single function, e.g. `rocm_add_fortran_bindings()`, in rocm-cmake or a small shared module) that each project calls in one line.
The helper sets a per-library result variable `<LIB>_HAVE_FORTRAN_BINDINGS` (for example `ROCSPARSE_HAVE_FORTRAN_BINDINGS`), true only when this library's bindings were actually built.
This is deliberately distinct from the root's `ROCM_LIBS_HAVE_FORTRAN`, which only reports that a Fortran compiler exists: the compiler can be present while a given library has `BUILD_FORTRAN_BINDINGS=OFF`, so "a compiler is available" and "this binding was built" are two different facts with two different names.
The packaging job, which always has amdflang, builds every library this way.

Concretely, the shared helper is:

```cmake
# shared helper, called in one line per library: rocm_add_fortran_bindings()
option(BUILD_FORTRAN_BINDINGS "Build the generated Fortran bindings" ON)
if(BUILD_FORTRAN_BINDINGS)
  # reuse the probe the root already ran, otherwise probe locally
  if(NOT DEFINED ROCM_LIBS_HAVE_FORTRAN)
    include(CheckLanguage)
    check_language(Fortran)
    set(ROCM_LIBS_HAVE_FORTRAN ${CMAKE_Fortran_COMPILER})
  endif()
  if(ROCM_LIBS_HAVE_FORTRAN)
    enable_language(Fortran)
    add_subdirectory(fortran)
    set(<LIB>_HAVE_FORTRAN_BINDINGS TRUE)  # e.g. ROCSPARSE_HAVE_FORTRAN_BINDINGS; feeds <pkg>_FORTRAN_FOUND
  else()
    message(STATUS "BUILD_FORTRAN_BINDINGS=ON but no Fortran compiler; skipping")
  endif()
endif()
```

For discovery, the Fortran target ships as its own per-library config package (`rocblas-fortran-config.cmake`, following the in-tree rocRAND/hipRAND convention), which the C library's config pulls in with `include(<lib>-fortran-config.cmake OPTIONAL)`.
So `find_package(rocblas)` exposes `roc::rocblas_fortran` when the bindings are installed and ignores it otherwise, with no separate export switch needed.
Each library's Fortran target sits in its own C library's existing CMake namespace: `roc::rocblas_fortran` and `roc::hipsolver_fortran`, but `hip::hip_fortran`, `hip::hipfft_fortran`, and `hip::hiprand_fortran`. This is per-library, not a hip-versus-roc rule (hipBLAS, hipSPARSE, and hipSOLVER are `roc::`); the full table is in Migration and compatibility. There is no per-library `<lib>::` namespace and deliberately no umbrella `rocm-fortran::` one, since `rocm-fortran` is the generator, not the deliverable.
The consumer-facing presence flag keeps the convention already in the tree, `<pkg>_FORTRAN_FOUND` (rocRAND and hipRAND already set `rocrand_FORTRAN_FOUND` / `hiprand_FORTRAN_FOUND`), which the Fortran config package defines.
There is a single source of truth: the shared helper derives the config package's `<pkg>_FORTRAN_FOUND` directly from the build-tree `<LIB>_HAVE_FORTRAN_BINDINGS`, so the two can never disagree.
Each per-library Fortran package is versioned with its C library, so `find_package(rocblas 5.1)` resolves the binding to the same version as the C library it was generated from; a static archive carries no soname, so this package version is the version handle.
This generalizes what hipSOLVER already does: hipSOLVER defaults the switch to `${UNIX}` (ON on Linux) today, so ON-by-default matches its behavior; the design adds the compiler guard so it is safe everywhere, redefines the target as the generated binding (the hand-written `hipsolver_module.f90` is the superseded module removed under the split), and replaces hipSOLVER's `EXPORT_FORTRAN_BINDINGS` export-set toggle with the optional-config include above.
The reference build of the shipped bindings is the packaging job, which collects every `fortran/` and compiles them in dependency order.

Cross-module dependencies are handled the same way the C libraries already handle them.
`rocsolver` uses `rocblas`, because `rocsolver-functions.h` includes `rocblas.h` and the rocSOLVER API takes `rocblas_handle` and rocBLAS enums.
That is the existing C dependency graph, now in Fortran, not a new problem.
The packaging job compiles rocBLAS before rocSOLVER; a standalone per-repo build resolves it with `find_package(rocblas)` on rocBLAS's Fortran package, exactly as the C build depends on `librocblas`.
Because each library is a single module under the packaged track, `rocsolver` simply `use`s `rocblas`; the dependency is kept as-is, matching the C graph.

### Layout after the split

**Source tree (committed).** Each library owns a `fortran/` directory next to its C headers, holding the generated module, its curated metadata, its build files, and its tests.
The generator *code* is never vendored here (P3); only the *data* it reads (the `.toml` and the metadata) is co-located.

```
rocm-systems/  .../hip/fortran/
   hip.F90                       generated: module hip (runtime + types/enums/check/handles/auxiliary)
   hip.toml, hip.argkinds        curated: generator config + metadata
   CMakeLists.txt                self-contained build → libhip_fortran.a + hip.mod
   hip-fortran-config.cmake.in   find_package export → hip::hip_fortran
   test/                         Fortran tests

rocm-libraries/  projects/rocblas/fortran/
   rocblas.F90                   generated: module rocblas (one module)
   rocblas.toml                  curated: generator config (required)
   rocblas.argkinds/.cptrargs/.native   curated metadata
   CMakeLists.txt                → librocblas_fortran.a + rocblas.mod
   rocblas-fortran-config.cmake.in
   test/  examples/
```

The CUDA metadata (`<lib>.txt`, `<lib>_enums_cuda.txt`, `*.cuda.finc`) does **not** live here: the packaged track has no CUDA backend, so those files stay in the compatibility track (`ROCm/hipfort`).
The CODEOWNERS entry is not a file in `fortran/` either; it is a line in the project's own `.github/CODEOWNERS` (each project keeps one, and a merge job assembles the monorepo root file from them).

**Install tree.** Because a `.mod` is compiler- and version-specific (see appendix), artifacts install under a per-compiler subdirectory, one archive and one module set per library:

```
<prefix>/
   lib/fortran/<compiler>/       librocblas_fortran.a, libhip_fortran.a, ...
   include/fortran/<compiler>/   rocblas.mod, hip.mod, ...
   lib/cmake/rocblas/            roc::rocblas_fortran
   lib/cmake/hip/                hip::hip_fortran
```

There is no `hipfort/<backend>/` nesting: the packaged track has a single backend, so the `.mod` files sit directly under `include/fortran/<compiler>/`.
`find_package(rocblas)` resolves the subdirectory matching the consuming project's compiler.

### Tests and documentation

**Tests.** hipfort's test suite splits across the libraries it covers: each library's Fortran tests move to its `fortran/test/`, next to the binding they exercise.
The per-PR hook compile-checks them without a GPU (this is what catches interface drift), and the packaging or validation job runs them on an AMD host.
A test that spans libraries (a rocSOLVER test that `use`s rocBLAS) follows the same dependency order as the bindings, resolved the same way: build order in the aggregating job, `find_package` for a standalone build.

**Documentation.** The API reference *is* the docstrings, and the generator emits them in place from the C headers into each `<lib>.F90`, so the reference content is already co-located in `fortran/` and there is no separate doc source to maintain. Everything a user reads travels with the binding automatically.

**Do the docs get their own subfolder in `fortran/`?** No. The docstrings live in the `.F90`, and the *doc build* is a whole-library concern (the C API and the Fortran API belong in one site), so it stays at the library's existing top-level `docs/`, which simply adds `fortran/*.F90` to its Doxygen input. We deliberately do not stand up a second doc-build system inside `fortran/`: co-locate the content (P1), not a parallel toolchain. So `fortran/` holds `test/` and `examples/`, but no `docs/`.

**Dependency impact on the doc build.** Cross-references are where library dependencies reach the docs: a rocSOLVER docstring that mentions `rocblas_handle` resolves only if rocBLAS's Doxygen tag file (or Sphinx inventory) is available when rocSOLVER's docs are built. So the doc build carries the same dependency graph as the code, expressed as tag-file and inventory dependencies rather than `.mod` dependencies. That is the argument for assembling the site in one aggregating job (the doc job that already has every project's output) rather than building each library's Fortran doc in isolation: an isolated build cannot resolve a cross-project reference without the other project's tag file. The remaining mechanics (portal integration, the exact tag-file wiring) need the ROCm documentation team; see Open questions.

## What changes for a library repository

For a `rocm-systems` or `rocm-libraries` project, the delta is small and bounded.

- A `fortran/` subdirectory holds the generated binding, its curated metadata, and its Fortran tests.
- A path-filtered CI check runs the generator, but only when a PR touches that library's `*.h` or its `fortran/` directory.
  Unrelated PRs are untouched.
- A CODEOWNERS entry gives the Fortran-bindings team ownership of `fortran/`.
- C and C++ developers do not write Fortran.
  The bot proposes regenerated bindings, and the bindings team owns the manual metadata.
- In the advisory stage a Fortran failure never blocks a C-only PR.
  In the enforcing stage the gate only fails a header change that skips its regenerated binding.
- Any Fortran module the library ships today (for example `rocblas_module.f90`) is superseded by the generated binding.
  Its removal is coordinated with the bindings team, not forced on the C developers.

## Dependencies

The generator needs Julia, Clang.jl, and the C headers.
It is a standalone, build-time tool.
It runs on its own to regenerate the committed `.F90`, and it is never part of a consumer build or of the compiler build.
No Julia dependency reaches the compiler, ROCm, or any consumer.
The `.F90` are committed artifacts, and using the bindings needs only a Fortran compiler.
Consumers are unchanged.
They still link `hip-runtime-amd` and the relevant math-library `.so`.

The coupling to Clang is deliberately shallow.
The generator parses public headers at the C API surface (`clang -x c`), not Clang's internal AST, so it is insensitive to the Clang version beyond parsing C.

Across repos, `rocm-systems` and `rocm-libraries` gain a `fortran/` subdirectory and a CI hook that invokes the generator, and the packaging job depends on all three.

## Version targeting and roadmap

The target ROCm version is simply whichever headers the generator reads, from the `rocm-libraries` monorepo (`develop`) or a ROCm install.

The immediate step is bringing the committed bindings current with ROCm 7.14 / 10.0 / 10.1.
Packaging for 10.2 is best-effort.
This RFC fixes the destination and the order of steps, not a packaging date.

The transition runs on two tracks, both emitted by the same generator.
The compatibility track is `ROCm/hipfort` with its current layout: the `hipfort_` prefix, the CUDA backend, and the split modules (interfaces, types, enums).
It is kept as source, not packaged, so existing users who build it themselves see no break through ROCm 7.14, 10.0, and 10.1.

That covers one population out of three, so here are all of them.

| How you get hipfort today                                                           | What the transition does to you                                                                                                                                     |
| :---------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| You build it from source                                                            | Nothing. The compatibility track keeps its layout, module names, and CUDA backend.                                                                                  |
| You install a package                                                               | Nothing, because there is no package to lose: packaging *returns* at 10.2, it does not continue (a ROCm 10.0 install ships no hipfort package, `.mod`, or archive). |
| You use a library's own Fortran module (`rocblas_module.f90`, `rocsparse.f90`, ...) | A real break at 10.2, when the generated module supersedes it. See What changes for a library repository.                                                           |

Only the third row is a break, and it is an owned, coordinated one rather than a side effect.
If some distribution does still publish a hipfort package, it keeps being published through the transition, and any de-packaging is announced with that distributor.
The packaged track lands in ROCm 10.2: packaging returns and the bindings split across `rocm-systems` and `rocm-libraries`, dropping the `hipfort_` prefix, collapsing each library to a single Fortran module, and dropping the CUDA backend from the package.
Emitting both is cheap, because the generator already supports multiple layouts.
The compatibility track then lingers as source only, and is retired at ROCm 11.0, leaving the packaged track as the sole deliverable.

## Migration and compatibility

Existing hipfort users face two mechanical edits, and nothing else.
Routine names, argument lists, and calling styles are unchanged, so this is find-and-replace rather than a rewrite: what breaks is the `use` line and the link target, not the call sites.

**1. Module names.** The `hipfort_` prefix is dropped and each library collapses to a single module.

| Old                                                | New           |
| :------------------------------------------------- | :------------ |
| `use hipfort` (+ `hipfort_types`, `hipfort_enums`) | `use hip`     |
| `use hipfort_rocblas` (+ `_enums`, `_types`)       | `use rocblas` |
| `use hipfort_<lib>` (+ `_enums`, `_types`)         | `use <lib>`   |

The HIP memory helpers (`hipfort_hipmalloc`, `hipfort_hipmemcpy`, `hipfort_hiphostregister`) fold into `hip`, and the cross-cutting modules (`hipfort_check`, `hipfort_handles`, `hipfort_auxiliary`) fold into the library module the caller already `use`s: `hipCheck` comes from `hip`, `rocblasCheck` and `rocblas_handle` from `rocblas`.

**2. Link target.** One `libhipfort-*.a` for everything becomes one archive and one CMake target per library, in the C library's own namespace, so `find_package(<lib>)` yields both halves:

```cmake
# before
find_package(hipfort REQUIRED)
target_link_libraries(app PRIVATE hipfort::rocblas)

# after
find_package(rocblas REQUIRED)
target_link_libraries(app PRIVATE roc::rocblas roc::rocblas_fortran)
```

The namespace follows each C library rather than a hip-versus-roc rule: `hip::` for the HIP runtime, hipFFT, and hipRAND; `roc::` for rocBLAS, hipBLAS, rocSPARSE, hipSPARSE, rocFFT, rocSOLVER, hipSOLVER, and rocRAND (matching the already shipped `roc::hipsolver_fortran`).

**Both can be installed at once**, so a codebase migrates one file at a time: the module names differ, the `.mod` files land in different subpaths, the archives have different filenames, and the object symbols are mangled per module name (`__hipfort_rocblas_MOD_...` against `__rocblas_MOD_...`).
The single rule is not to `use` both bindings for the same library in one scope, since they export the same public names and the reference becomes ambiguous.
That clash is source-level and scope-local, which is exactly why a file-by-file migration works.

Two capabilities do not carry over.
The CUDA (`nvptx`) backend is not built for the packaged track (see CUDA backend), and `roctx` is deferred rather than shipped in the initial packaged set, because rocTX is moving into `rocprofiler-sdk` upstream and its home is not settled; both keep working on the compatibility track in the meantime.

A user-facing migration guide, with the full per-library table and a scripted rename, ships with the bindings rather than living in this RFC, so that it tracks the packaging as it lands instead of freezing with this document.

## Rollout and success criteria

The rollout is staged, and each stage has a clear exit.

1. **Bring current (7.14 / 10.0 / 10.1).** The committed bindings match today's headers; the CI hook runs in *advisory* mode, suggesting regenerated diffs without blocking a merge.
1. **Co-locate and package (10.2).** Each in-scope library has a `fortran/` directory, the packaging job builds per-library `.mod` and `lib<lib>_fortran.a`, and the `bindings-in-sync` gate becomes *enforcing*.
1. **Retire the compatibility track (11.0).** `ROCm/hipfort` under its current name is removed, and the packaged track is the sole deliverable.

It is done when `hip` and all eleven in-scope libraries carry a co-located `fortran/` (generated source, curated metadata, tests); a header PR that changes a binding fails the enforcing gate until the regenerated `.F90` is committed; `find_package(<lib>)` exposes `<ns>::<lib>_fortran` when built; and an external consumer compiles and links against the installed package for at least amdflang and gfortran.

## Open questions and risks

This is a draft.
Several points are still open.

- **Cross-repo header dependencies and build order.** Some bindings depend on another library's headers (rocSOLVER on rocBLAS) or on the HIP runtime, and the modules have a compile order.
  This mirrors the existing C dependency graph and is resolved the same way: build order in the packaging job, and `find_package` for isolated per-repo builds (see Building the bindings).
  The nightly full regeneration is the backstop for cross-header drift.
- **Assumed-rank interfaces (F2018).** Decision: ship assumed-rank OFF by default.
  The default stays the explicit, per-rank variants, which work on older compilers, with a build option (`-DUSE_ASSUMED_RANK=ON`) to enable it.
  Enabling it *replaces* the per-rank variants rather than adding to them: an assumed-rank dummy is not distinguishable by rank from the per-rank specifics, so the two cannot legally coexist in one generic.
  F2018 assumed-rank is not uniformly mature across gfortran, flang, ifx, and cray, and HPC sites lag the standard; it compiles with amdflang today, but corner-cases are a risk.
  Turn it on by default later, once the compilers are proven.
  Assumed-rank would also shrink the generated sources and their build time (for example the ~24k-line `hipfort_hipmalloc` on today's hipfort).
- **The Fortran 2003 floor: settled for the standard, open for the compiler version.** The design sets F2003 as the minimum (see Language standard), and no consuming application in view needs less than Fortran 90.
  What remains is the version axis rather than the standard axis: the raw `type(c_ptr)` surface is already the smallest interoperable subset, and nothing below F2003 can express a standard-conforming C binding at all, so the question worth asking the large HPC sites is which compiler versions they pin, not which standard they target.
- **Where the generator ultimately lives, and whether it opens up.** It stays a standalone repository (P3), and that is the part this RFC settles.
  What is not settled is whether it eventually moves next to the Fortran toolchain that consumes its output, and whether the repository becomes public.
  Today it is internal, which bounds who can reproduce a regeneration from scratch; it does not bound who can read, review, or build the result, since the emitted `.F90` are committed in the library repositories and compile with any Fortran compiler.
  Opening it would mainly buy external reproducibility of the generation step, and is worth deciding on that basis rather than by default.
- **Assembling the documentation site (Sphinx + Doxygen).** The layout is decided (see Tests and documentation): docstrings are generated in place, `fortran/` has no `docs/` subfolder, and the Fortran API is documented from the library's existing top-level `docs/`.
  What stays open is assembling one coherent site from twelve projects, and the cross-project reference wiring (a rocSOLVER docstring referencing rocBLAS types needs the dependency's Doxygen tag file or Sphinx inventory at build time).
  Two shapes: each library builds its Fortran doc section, aggregated into the ROCm doc portal; or the doc job assembles a single combined site from all sources.
  This needs the ROCm documentation team.
- **A second error model (owned error objects).** rocSPARSE now ships, alongside the status code, an owned error object: several entry points take a `rocsparse_error*` out-parameter producing an opaque descriptor the caller must release with `rocsparse_destroy_error`, whose message is a borrowed `const char*` tied to the descriptor's lifetime.
  The inherited status-code model has no vocabulary for an owned object, a destroy obligation, or a borrowed string.
  Open: how the descriptor is represented in Fortran, who owns the destroy, and the lifetime of the borrowed message pointer.
- **Deprecation and removal policy.** A C `deprecated` attribute is a compiler diagnostic, not a docstring, so a Fortran caller of a deprecated entry point currently gets silence where a C caller gets a warning; and because the generated surface tracks the header exactly, removing a C function would reach the Fortran module in the same merge, with no release in which the Fortran user was warned.
  Plan: capture the attribute as a generated docstring marker (Fortran has no deprecation attribute below F2023), and forbid an entry point from leaving the generated surface in the same release it leaves the C header (the in-sync gate fails a regeneration that drops a public Fortran name unless the PR acknowledges it).
- **Are dummy-argument names public API?** A Fortran keyword actual argument makes every dummy name part of the module's public contract, yet those names come from C parameter names, which are not API in C: renaming a C prototype parameter (invisible to C callers) would break a Fortran caller using that keyword.
  Open: decide whether dummy names are supported API; if so, the in-sync gate fails on a dummy-name rename rather than suggesting it, and unnamed C parameters get a deterministic naming rule so they do not drift between generator versions.

## Alternatives considered

**A single bindings super-repo.** All bindings in one dedicated repository (today's hipfort, generated) keeps a clean boundary and simple ownership, and it stays a reasonable option.
Its tradeoff is that the bindings sit downstream of the header change, so they update in a separate, later PR, and headers must be copied out to regenerate.
Co-location gives up that clean separation for updates that move in lockstep with the C API, which is the property this RFC prioritizes.

## Appendix: How NVIDIA does it (baseline)

The closest analog is NVIDIA's Fortran story for its GPU math libraries, and the contrast is instructive.

NVIDIA ships precompiled Fortran modules for cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuTENSOR, and more, inside the HPC SDK, next to `nvfortran`.
You write `use cublas` and link with `-cudalib=cublas`.
The decisive difference is that NVIDIA owns the only compiler its users use.
With a single supported compiler the "`.mod` incompatible across compilers" problem simply does not exist for them, so shipping prebuilt `.mod` is fine.
We cannot copy that: our users are on gfortran, flang, ifx, cray, and amdflang, so we must ship source and compile per compiler, which is exactly what this RFC does.

A few more contrasts, point by point:

- **Location.** NVIDIA's interfaces live in the compiler/SDK, not co-located in each library's source repo.
  Our co-location in `rocm-libraries` is, if anything, more tightly integrated.
- **Naming.** No prefix: `use cublas`, not `nvfort_cublas`.
  The module name is the library name.
- **Backward compatibility.** They keep the legacy and the v2 calling conventions side by side (`cublas` and `cublas_v2`).
- **Array handling.** They do not use assumed-rank.
  They use the CUDA Fortran `device` attribute: `nvfortran` distinguishes host from device arrays, and the interfaces declare device arrays, so host/device dispatch is resolved by the language.
  gfortran and flang do not know `device`, so our `bind(C)` plus `c_loc` plus generated variants is the portable substitute for what `nvfortran` does natively.
- **Scalar host-vs-device (alpha/beta).** Because `nvfortran` knows host from device, its non-v2 modules set and restore the library pointer mode implicitly, at a small overhead.
  We cannot rely on the compiler for this, so `.argkinds` records which scalar pointers are host values.
  One caveat static metadata cannot remove: the host-versus-device choice for `alpha`/`beta` is also *runtime* handle state (`rocblas_set_pointer_mode`), so the ergonomic host-scalar form is only correct when the handle is in host-pointer mode.
  The wrappers therefore must not silently set or restore that mode (it is shared, mutable handle state, so touching it would race other threads), and the raw `type(c_ptr)` surface stays the escape hatch for device-pointer mode.
- **Tooling.** Their "wizard" is the compiler driver: `-cudalib` plus an automatic module search path.
  There is no separate installation checker because the bindings are bundled with the compiler.
  Ours are not bundled with a single compiler, which is why an explicit packaging and a presence check are worth having.

The through-line: most of the hard questions here (cross-compiler `.mod`, ABI across compilers, where the bindings live, generation) exist for us precisely because we do not own the one compiler our users run.
NVIDIA answers them by monoculture; we answer them by generation, source shipping, and co-location.

## Appendix: Fortran `.mod` and `.a` incompatibility across compilers and versions

This is the constraint that shapes the whole packaging design, so it is set out here in full.

Unlike C, Fortran has no standard ABI and no standard module format.
Two artifacts are affected.

- **The `.mod` file.** A `.mod` is a compiler-private serialization of a module's interface (names, kinds, generic-resolution tables).
  Its format is undocumented and specific to the compiler, and it is **not stable across versions of the same compiler**: gfortran, for instance, bumps its module-format version between releases, so a `.mod` written by one gfortran is rejected by another.
  There is no interoperability across compilers either: a gfortran `.mod` is meaningless to ifx, amdflang, or Cray, and vice versa.
- **The compiled `.a`.** The wrapper bodies compile to objects whose symbols are **name-mangled** by a compiler-specific scheme (trailing underscores, module-name decoration such as `__rocblas_MOD_`), so the archive is tied to the same compiler, and version, as the `.mod` it accompanies.

So there is exactly one portable thing to ship: **the Fortran source**.
A prebuilt `.mod`/`.a` set is usable only by the exact compiler, at essentially the exact version, that produced it.

One relaxation exists on the compiler we ship.
amdflang is gaining an opt-in `-fmodule-mismatch-check=warn` (in `amd-staging`, landing in ROCm 10.1) that downgrades the module-version check to a warning, so a newer amdflang can read a `.mod` written by an older one.
That softens the *version* axis for the precompiled amdflang set, which is the set we ship and the skew our users are most likely to meet (a ROCm-shipped `.mod` against a slightly different amdflang).
It does nothing for the *compiler* axis, which is the one that forces us to ship source: no flag makes a gfortran `.mod` meaningful to amdflang.

This is why the design:

- ships **source** so any site can compile with its own compiler, and additionally ships a **precompiled set for the default compiler (`amdflang`)** for the common case;
- keys the install tree by **compiler** (and, where a site keeps several versions side by side, by version) under `lib/fortran/<compiler>/` and `include/fortran/<compiler>/`;
- treats the only stable ABI as the **C one**, the vendor `.so`, which the `bind(C)` layer rides on and which every compiler agrees about.

NVIDIA escapes all of this by owning the single compiler (`nvfortran`) its users run, which is why it can ship prebuilt `.mod` and we cannot (see How NVIDIA does it).

## Appendix: Implementation backlog and open risks

The direction is set; the items below are implementation and CI work that does not change the design, recorded here so they are not lost. None is a blocker.

**CI and reproducibility.**

- Pin what the generator resolves (its revision, the Julia and Clang.jl versions, the include roots, the macro profile, and the C standard level it parses at), so a regeneration is reproducible and the fan-out across `hip` and the eleven libraries has a containment boundary.
- Some public headers do not parse from a source checkout alone: a few are generated into the build tree, and feature-gate headers (`*.h.in`) can change the visible API without touching a tracked `.h`. Define the generator's input as a named, configured artifact per library, or run the hook nightly for the libraries whose headers need a configured or partly-built tree.
- The per-PR hook reads PR-head content, runs a parser over it, and needs a write-capable token to post the suggestion. Use the standard two-workflow split: an unprivileged job on the PR head produces the diff as an artifact, and a trusted job on the base ref posts it and never executes PR code.

**Release integrity.**

- Resolve every `bind(C, name=)` label in the generated sources against the symbol tables of the libraries actually shipped in a release, and fail on any unresolved name, so a feature-gated API cannot ship as a Fortran interface with no code behind it. Linking the built archive with `--whole-archive -Wl,--no-undefined` catches the wrapper half.
- Add a CI check comparing each emitted struct's size and alignment against the C compiler's `sizeof`/`_Alignof` on the target, since those ABI facts are frozen into a committed artifact. This also covers the complex-number mapping, which maps a two-float C struct to a Fortran `complex` on an ABI assumption.

**Type mapping.**

- Integer scalars map to named `iso_c_binding` kinds, not host-measured ones. State the assumption that a default-`INTEGER` site building with `-i8` must suffix its literals, and that the `iso_c_binding` kinds are the companion C processor's, which need not be the compiler that built the `.so`.
- Function-pointer and callback types map to `c_funptr` with an emitted `abstract interface` where the signature is mappable; a callback that must be a device function pointer is marked unsupported-from-Fortran in its docstring rather than emitted as if usable.
- For the few handle types that differ across platforms (for example `hipfftHandle`, a 32-bit int on NVIDIA but a pointer on AMD), either record a per-backend type substitution or drop that library from the nvptx track.

**Cross-library and metadata consistency.**

- Typed handles, the "own headers" marker, and the name prefix are global decisions, not per-library ones (rocSOLVER's whole API takes `rocblas_handle`). Promote them to one repository-level config, and gate on two modules claiming the same declaration, or a module referencing a type no module claims.
- Assert the 63-character Fortran identifier limit in the generator (the longest C symbol is 52 characters), and fail rather than truncate.

**Packaging and build variants.**

- Version each per-library Fortran package with its own C library, and state the versioning scheme (a static archive carries no soname).
- The ASAN build variant ships no Fortran, since Fortran linking is unsupported under ROCm's ASAN configuration; the presence check reports that distinctly rather than as "not installed for your compiler". Same for the coverage variant.

**Consumer experience and policy.**

- Offer a non-CMake consumer path (a `pkg-config` file, or a small flags script) with one worked command line per compiler, since not every HPC site builds with CMake.
- Decide whether to ship a supported migration helper that rewrites `use hipfort_*` lines to the packaged module names. The transition guide documents the `sed` recipe such a tool would automate (delete the folded modules, rename the rest, leave `roctx` and `cuda_errors` alone), so the open question is support surface and the residue the recipe leaves to the user (fixed-form sources, continuation lines, `use hipfort_check` in a file that never used `hipfort`), not effort.
- The CMake config package requires Fortran enabled before `find_package`, errors with a named diagnostic when the Fortran compiler identity is unknown, and carries the module directory as an interface include selected from the resolved compiler axis.
- State a default-inclusion policy and who answers support tickets for a public Fortran API shipped under a library's name (bug reports arrive against rocBLAS, while the bindings team owns `fortran/`).
- State the non-goals explicitly (Windows, Fortran under ASAN, the bounds of non-CMake tooling), and keep the generator-repo name consistent throughout.

## Appendix: Testing the generator

The generator is tested at three levels: its own logic, the shape of its output, and the output at runtime.

Its logic has a unit-test suite in the generator repository.
It exercises the pure transforms, with no ROCm headers and no Clang parse: C-type to Fortran mapping, name and spec helpers, config loaders, the bind, wrapper, enum, and struct emitters, 132-column line wrapping, docstring formatting, and the hip-to-cu name heuristic.
A byte-golden in that suite asserts the emitters still produce the same output as before, and `compile_golden.jl` runs `gfortran -fsyntax-only` on regenerated fixtures to prove that output is valid Fortran.
`runtests_split.jl` covers the default split layout, which goes through separate interface and wrapper emitters.

The output is also gated in CI, and those gates double as regression tests.
The committed modules must compile under amdflang and gfortran (`-std=f2018`).
They must survive hipfort's Doxygen pass.
The hipfort test suite must still compile against the regenerated interfaces, which catches interface drift without a GPU.
The CUDA bind names are validated offline by `audit_cuda_mappings.jl` against a checked-in CUDA symbol dump, so a wrong `cu*` name cannot slip through and no CUDA install is needed.
The end-to-end golden is `regenerate.sh` plus `git diff` against hipfort's `develop`.

These gates are compile-only, by design, so they need no GPU.
The bindings themselves are exercised at runtime by the hipfort test suite on an AMD GPU.
That runtime suite must cover both pointer modes (host and device) for the `argkinds`-classified entry points: a misclassified scalar pointer compiles and links cleanly, so it is the one correctness property no compile-only gate can see (see Keeping the classifications correct).

Because the transforms are now unit-tested, the trust the enforcing `bindings-in-sync` gate needs is largely in place: a C developer who accepts a suggested `.F90` relies on a tool whose logic is checked, not only on its output.
