---
author: Nirmal Unnikrishnan (nunnikri)
created: 2026-10-27
modified: 2026-10-28
status: draft
---

# RFC0016: PowerPC (ppc64le) Build-Only Host Support

## Overview

Add `ppc64le` as a new **host CPU architecture** in TheRock's CI: compile the
full ROCm stack (host runtime + existing GPU device targets) on `ppc64le`
without running any GPU tests, since no `ppc64le` + AMD GPU test hardware is
expected to exist in the fleet. This is a build-verification target, not a
supported/shipped `ppc64le` package yet.

Host CPU architecture is orthogonal to GPU ISA (`gfx*`) target selection — the
AMDGPU device compiler backend already targets GPU ISAs independent of host
CPU, and no coupling between the two was found anywhere in the matrix code.

## Motivation

A request came in for PowerPC (`ppc64le`) support in ROCm/TheRock. Both GCC
and LLVM/clang have long-standing, mature `powerpc64le-linux-gnu` cross- and
native-compilation support (it's the standard toolchain target for
Linux-on-Power distros such as RHEL/Fedora/Ubuntu ppc64le) — compiler
capability is not the gating factor. This RFC scopes what else is actually
needed in TheRock.

## Requirements

What "done" means for v1:

- **R1 — Build-only CI signal.** Compile the existing ROCm/GPU target matrix
  on `ppc64le`; do not execute GPU tests (no `ppc64le` + AMD GPU test
  hardware expected in the fleet).
- **R2 — GPU target parity.** Whatever `gfx*` families build on `x86_64`
  should build on `ppc64le`, with one known v1 exception (R3).
- **R3 — Known v1 scope exclusion.** `gfx95`/`gfx125` target families are
  blocked by an unrelated AVX-flag bug in composablekernel (Appendix A,
  §composablekernel). Excluded from the v1 `ppc64le` build matrix until
  fixed — tracked as a gap to close, not a permanent narrowing, since these
  are mainstream CDNA3+/MI300 target families.
- **R4 — Package format support, phased:**
  - Phase 1: tarball, RPM, Debian package (`.deb`) — rpm and deb share the
    same underlying code change (see Packaging & Versioning), so they ship
    together rather than being split across phases.
  - Phase 2: Python wheel
- **R5 — No version-numbering changes.** The existing date-based version
  scheme is already architecture-agnostic.
- **R6 — Scope.** Build-verification only in v1 — not a supported or shipped
  `ppc64le` product target.
- **R7 — Fix the `clr` `currentStackPtr()` correctness bug.** The
  uninitialized-stack-pointer fallback in
  `clr/rocclr/os/os_posix.cpp:781-797` on non-x86/non-ARM hosts (Appendix A)
  must be fixed as part of v1, not deferred as opportunistic cleanup — it is
  a silent correctness bug, not a compile blocker, but `ppc64le` is the host
  that exposes it today.

### Definition of Build Success

For this RFC, "full ROCm stack" means all projects enabled in the `ppc64le`
build topology successfully configure and compile.

A successful build does not require all projects present in the repository
to be enabled. Projects explicitly excluded by documented architecture-
specific exceptions (Requirement R3 and the exclusions listed below) remain
outside the success criteria until the underlying issues are resolved.

## Out of Scope

This RFC, and the Appendix A audit it's based on, covers only `rocm-systems`
and `rocm-libraries` — the ROCm platform/runtime/math-library stack itself.
Explicitly out of scope for v1:

- **PyTorch, JAX and other downstream ML frameworks** (`torchvision`, `torchaudio`,
  `apex`, `triton`) — built via `external-builds/pytorch` as a separate
  downstream pipeline (checkout + build against an already-produced ROCm
  SDK), with their own dedicated, `x86_64`-specific CI workflows
  (`multi_arch_build_portable_linux_pytorch_wheels*.yml`,
  `multi_arch_build_windows_pytorch_wheels*.yml`, etc.). None of this was
  swept by the Appendix A audit, and upstream PyTorch's own `ppc64le`
  support is a separate, much larger unknown. Getting the base ROCm SDK
  building on `ppc64le` (this RFC's scope) is a prerequisite for ever
  evaluating PyTorch-on-`ppc64le` — not something to scope together with it.
- **ROCm Validation Suite (RVS)** — not built by TheRock at all; it's an
  external dependency `rdc` optionally links against
  (`rocm-systems/projects/rdc/rdc_libs/rdc_modules/rdc_rvs`, gated behind a
  `BUILD_RVS` option and `find_package(rvs REQUIRED)` against a
  separately-installed package, off by default). Out of scope by
  construction — nothing to change here unless `BUILD_RVS` is ever turned on
  by default.
- **MIGraphX and MIVisionX** — neither exists in TheRock's current submodule
  set (`rocm-systems`, `rocm-libraries`); both remain separate upstream ROCm
  repos not yet folded into this monorepo (see RFC0012's "Computer Vision
  SDK" as a forthcoming, separate expansion stream). Not auditable or
  actionable from within TheRock today — would need revisiting only once
  (if) either is integrated into TheRock itself.

## Current State

TheRock's build-topology/CMake layer already recognizes `ppc64le` as a
canonical host architecture name, alongside `x86_64`/`aarch64` — this is
further along than a blank slate:

- The build-topology schema (`cmake/therock_features.cmake`,
  `build_tools/_therock_utils/build_topology.py`,
  `docs/development/build_system.md`) already lists `ppc64le` as one of
  three canonical processor names.
- `cmake/therock_subproject_dep_provider.cmake` already has a `ppc64le` →
  `powerpc64le` arch-suffix branch for sanitizer runtime lookup.
- Two modules are **already explicitly disabled on `ppc64le` today**
  (`rocprofiler-systems`, `rocprofiler-systems-examples`, via
  `BUILD_TOPOLOGY.toml`'s `disable_processors` key), implying the rest of
  the tree is expected to build there.

A full per-module audit of every project in `rocm-systems` and
`rocm-libraries` was completed (full detail in **Appendix A**; exact commits
audited in **Appendix A's "Audit Reference Commits"**). Summary:

- **6 confirmed hard blockers** — none structural; each is a narrow,
  isolated fix or exclusion (rocprofiler-systems, rccl, rocr-runtime test
  code, shared/tensile, rocfft, composablekernel). RPP was initially flagged
  as a seventh, independent blocker (unconditional AVX2/FMA intrinsics with
  no portable fallback) — closer review during RFC discussion found it
  already has a complete SIMDE-based portability layer for non-x86 hosts;
  see the RPP note under "Unblocking the Build" below and Appendix A.
- **A further ~10 minor gaps** — silent no-ops, perf-only degradations, or
  narrow library-search-path issues. None block compilation.
- **Real existing positive precedent already in the tree**, not just a
  theoretical possibility: rocr-runtime's intrinsics compatibility shim,
  rdc's live `__powerpc64__` branch, rocdbgapi's PPC64-aware type-layout
  probe, rccl's write-combine-fence handling, RPP's SIMDE-based AVX2/F16C/FMA
  emulation layer, hipsolver's OpenBLAS `TARGET=POWER8` selection,
  emulation/rocjitsu's arch branch, and kpack's ELF surgery layer already
  defining `EM_PPC64`/`R_PPC64_RELATIVE` and the correct 64KB page size for
  PPC64 (though currently untested — no PPC64 fixtures exist in kpack's own
  test suite).
- **CI coverage: zero.** No `.github/workflows/*.yml` references `ppc64le`
  anywhere today — nothing has ever actually exercised this configuration in
  CI.
- **Packaging: greenfield.** No existing non-`x86_64` packaging precedent at
  all, not even for `aarch64`.

## Build Environment : Alternatives Considered

How to actually produce `ppc64le` builds — three options, evaluated in
practice as part of this RFC (an exploratory build using Option 2 below was
run in parallel with this analysis and successfully installed a working
`ppc64le` toolchain: gcc-toolset-13/cmake/ninja via yum on
`manylinux_2_28_ppc64le`).

### Option 1 (Primary recommendation): Native `ppc64le` hardware or full-system VM

- **Pros:** Matches how every other host architecture is already treated.
  `compiler/pre_hook_amd-llvm.cmake`'s `LLVM_TARGETS_TO_BUILD "AMDGPU;Native;SPIRV"` already auto-resolves `Native` to `PowerPC` on a
  real/virtualized `ppc64le` host — zero compiler-config changes needed. No
  `CMAKE_CROSSCOMPILING`/`try_run` friction anywhere in the dependency tree,
  since the environment is genuinely native. Full build speed, no emulation
  tax on the (large) LLVM compile or anything downstream.
- **Cons:** Slowest to provision — requires real POWER hardware or a
  full-system VM, neither of which exists in the fleet today. Ongoing
  hosting/maintenance cost for dedicated infrastructure.

### Option 2 (Secondary — immediate bridge): QEMU user-mode emulation

- **Pros:** Available today on the existing `x86_64` fleet — no new runner
  procurement. Looks native to the build system: `uname` genuinely reports
  `ppc64le` inside the emulated environment (via `binfmt_misc`), so
  `LLVM_NATIVE_ARCH` resolves correctly and CMake `try_run` checks execute
  transparently — no cross-compile toolchain-file engineering needed.
  Already validated in practice for this RFC.
- **Cons:** Meaningfully slower for CPU-heavy work (commonly 3-8x) — most
  consequential for the from-source LLVM/amd-llvm build, the single largest
  compile unit in the stack. Not representative of real hardware
  performance/timing characteristics (immaterial for a build-only
  correctness signal; would matter if timing-sensitive behavior ever needed
  validating).

### Option 3 (Fallback only — if real/virtualized `ppc64le` hardware access proves entirely unobtainable): True toolchain cross-compilation

- **Pros:** Fastest in principle — every build step, including the LLVM
  compile, runs at native `x86_64` speed on the existing fast fleet, with no
  emulation tax at all.
- **Cons:** Requires real new engineering, not a flag flip — and does not
  actually save any CI-infrastructure cost that Option 2 doesn't already
  capture. Option 2 already reuses the existing `x86_64` fleet entirely
  (just a `binfmt_misc` registration step, no new runner procurement — see
  Appendix C); Option 3's only advantage over Option 2 is raw build-time
  speed (avoiding the ~3-8x QEMU emulation tax on the LLVM compile), not
  infrastructure reuse. `LLVM_TARGETS_TO_BUILD "AMDGPU;Native;SPIRV"` would
  need explicit cross-bootstrap plumbing (LLVM's own `CrossCompile.cmake`
  two-stage pattern — confirmed via
  [LLVM's own cross-compile documentation](https://llvm.org/docs/HowToCrossCompileLLVM.html)
  to require a native-`tblgen` side-build plus
  `CMAKE_FIND_ROOT_PATH_MODE_*`/`LLVM_HOST_TRIPLE`/`CMAKE_SYSROOT` plumbing,
  an area with [open upstream issues](https://github.com/llvm/llvm-project/issues/125402)
  even for its best-trodden cases), and TheRock's own `cmake/` layer has no
  existing toolchain-file pattern for this today — this bootstrap work, not
  probe-seeding below, is the dominant cost of this option. It is also the
  less-common direction: essentially all documented/tested LLVM
  cross-compile cases are `x86_64` host targeting ARM/AArch64/RISC-V
  *object code*. TheRock's case is the reverse (`x86_64` host, `ppc64le`
  target) and harder still, because ROCm ships its own compiler toolchain
  rather than just prebuilt libraries — the cross-compiled `amd-llvm`/clang/
  offload-bundler binaries themselves would need to end up as working,
  natively-executing `ppc64le` programs (a self-hosting-toolchain bootstrap),
  not merely emit `ppc64le` object code from an x86_64-resident compiler.
  Separately, a full-tree re-audit (both submodules plus TheRock's own
  `cmake/`) for `try_run`/`check_*_source_runs`-style feature detection —
  which hard-errors under pure cross-compilation without a pre-seeded cache
  variable or a narrowly-scoped `CMAKE_CROSSCOMPILING_EMULATOR` — found 4
  raw hits, of which only **1 is actually active for this build**:
  composablekernel's AVX-512 probe. The other 3 are inert here (a vendored
  `timemory` CUDA-only `try_run` never entered since TheRock builds HIP not
  CUDA, a libomp probe that fails to compile by design so nothing ever
  executes, and a Windows-only `glog` check) — so this specific risk is
  confirmed narrow, not an open-ended hedge. No existing precedent anywhere
  in TheRock for cross-compiling its own toolchain for a different host
  architecture than the build machine.

### Recommendation

Primary target: **Option 1**, once procurable — matches how the project
already treats every other host architecture and avoids all of the friction
above. **Option 2** is the right immediate choice to start producing a
build-only signal now, on existing infrastructure, while native hardware is
pursued. **Option 3** should only be pursued if real/virtualized `ppc64le`
hardware genuinely cannot be obtained at all — for this project specifically
it is strictly more engineering effort than either of the other two, and
that effort buys only build speed, not infrastructure savings: Option 2
already reuses the same existing `x86_64` fleet with no new procurement (see
Appendix C), so the "existing CI infra" argument does not favor Option 3
over Option 2 the way it might first appear to. Option 3 would also be a
net-new, largely unprecedented build pattern for this project (a
cross-compiled toolchain that must itself run natively on the target, not
just emit target object code — see Cons above), versus Options 1 and 2,
which both reuse the exact same build pattern every other host architecture
already uses.

## Unblocking the Build — Scope Decisions

6 confirmed hard blockers were found (full detail in Appendix A, concrete
per-file fix list in Appendix B). Key v1 scoping decisions:

- **RPP:** not excluded from v1. Initial audit flagged `rppdefs.h`/
  `rpp_cpu_simd_load_store.hpp`'s intrinsics includes and
  `add_fma_property()`'s `-mno-fma` flag as blockers; closer review during
  RFC discussion found both already arch-guarded with a working fallback —
  `#elif __x86_64__` → SIMDE-based AVX2/F16C/FMA emulation
  (`simde/x86/{avx2,f16c,fma}.h`) on other hosts, wired in end-to-end via
  `pkg_check_modules(SIMDE REQUIRED simde)` in RPP's top-level
  `CMakeLists.txt`. All 92 AVX2-using kernel files in `tensor/cpu/kernel/`
  route through this guarded path with zero direct x86 header includes. The
  one open item is whether the `simde` pkg-config package is available on
  the `ppc64le` build image — tracked as a Phase 0 Dockerfile task
  (Appendix B), not a code exclusion.
- **composablekernel `gfx95`/`gfx125` data-type tests:** excluded from the
  v1 target matrix (Requirement R3) until the unguarded AVX flags are fixed.
- **`rocprof-sys-rt` (rocprofiler-systems):** gate the subcomponent off in
  v1; a real Dyninst-based port is tracked as follow-up (upstream Dyninst
  already supports `ppc64le`, and a sibling vendored copy in the same module
  already does too, so source exists to port from).
- **Everything else** (rccl's `timer.h`, rocr-runtime's test-only headers,
  Tensile's hardcoded host triple, rocFFT's forced FFTW AVX2 flag): small,
  isolated fixes, not exclusions — fix directly rather than carve out.
- **`clr`'s `currentStackPtr()` bug (Requirement R7):** required fix, not
  opportunistic cleanup — see Definition of Build Success and Requirements
  above.

### Temporary Architecture Exclusions

The v1 exclusions described in this RFC are not intended to become permanent
feature reductions for `ppc64le`.

The composablekernel `gfx95`/`gfx125` exclusion (Requirement R3) is treated
as a tracked architecture-enablement gap and should be revisited as the
underlying AVX-flag issue is resolved. The long-term objective remains
host-architecture parity with `x86_64` wherever practical.

## CI Integration

- Reuse the existing build-but-don't-test matrix pattern already used 18
  times today (`"tests_on_trigger": []`, `"test-runs-on": ""`) — add a
  `ppc64le` entry following the same shape.
- Add a `ppc64le` runner pool to `BUILD_RUNNER_LABELS`, pointed at whichever
  Build Environment option above is in effect.
- Thread an `arch` parameter from `configure_multi_arch_ci.py` through every
  workflow's container `image:` expression, so `ppc64le` selects a new build
  image instead of the ~40 hardcoded `x86_64` references (full file list in
  Appendix B).

## Packaging & Versioning

**Phasing:** Phase 1 — tarball, RPM, Debian package. Phase 2 — Python wheel.

### Tarball — naming gap

Today's tarball names (e.g.
`therock-dist-linux-gfx94X-dcgpu-10.3.0a20261026`) carry **no host CPU
architecture component at all** — only OS, GPU family, and version. Unlike
wheels (below), there is no existing slot to repurpose; a new naming
segment is needed so `ppc64le` and `x86_64` tarballs are distinguishable.

**Decided:** add the arch segment only for non-`x86_64` builds, leaving
today's `x86_64` tarball names unchanged (unlabeled continues to mean
`x86_64`) — e.g. `therock-dist-linux-gfx94X-dcgpu-...` stays as-is, while a
`ppc64le` build produces `therock-dist-linux-ppc64le-gfx94X-dcgpu-...`. This
avoids a breaking rename of an artifact naming scheme that's already
shipping, at the cost of an implicit "no label means `x86_64`" convention
going forward — accepted as the right tradeoff.

#### Tarball Naming Compatibility

Leaving existing `x86_64` tarball names unchanged avoids breaking existing
automation, documentation, and downstream consumers that already depend on
the current naming convention.

The resulting convention is:

- No architecture segment implies `x86_64`
- Non-`x86_64` architectures carry an explicit architecture segment

This asymmetry is intentional and prioritizes backward compatibility over
naming uniformity.

### RPM / Deb

- Architecture field mapping: `amd64` → `ppc64el` (deb), `x86_64` →
  `ppc64le` (rpm) — these are the correct per-format spellings (Debian and
  RPM diverge here).
- Both fields are set together, per package entry, in the same
  `build_tools/packaging/linux/package.json` (81 entries) — since the
  underlying code change isn't separable by format, rpm and deb ship
  together in Phase 1 (Requirement R4).

### Wheel

- Already has an existing arch-tag slot (unlike tarball) — today hardcoded
  to `linux_x86_64`/`manylinux_2_28_x86_64` in an allowlist. Needs the new
  value (`linux_ppc64le`/`manylinux_2_28_ppc64le`) added to that allowlist.
  The wheel filename itself gets its platform tag automatically from the
  build tool (auditwheel/wheel) when built on a `ppc64le` host — this isn't
  a manual rename, just an allowlist fix.

### Versioning

No changes needed — version strings (e.g. `10.3.0a20261026`) are date-based
and already architecture-agnostic.

## Risks

### Increased CI Resource Consumption

A `ppc64le` build running under QEMU user-mode emulation can be
significantly slower than a native `x86_64` build, especially for the
LLVM/amd-llvm portion of the stack. Build duration and runner utilization
should be expected to increase until native `ppc64le` infrastructure becomes
available.

### Build Stability During Initial Enablement

No CI currently exercises `ppc64le` paths anywhere in TheRock. Initial
enablement is therefore expected to uncover latent host-architecture
assumptions over time as code evolves. Some amount of early build churn
should be expected until the configuration becomes routinely exercised.

### Limited Validation Coverage

A successful build does not imply runtime correctness, performance
validation, or GPU functionality validation. Because no known `ppc64le` +
AMD GPU fleet exists, this effort establishes only a compilation signal.

### Maintenance Burden

Any architecture-specific exclusions introduced for v1 (for example the
composablekernel target-family exclusion) create an ongoing maintenance
obligation until the underlying issue is fixed. Future changes to that
project may require periodic revalidation of the excluded path.

## Open Questions

- Is there an actual `ppc64le` hardware partner/customer driving this
  request, or is it speculative? This shapes how soon Option 1 (native
  hardware) is realistic versus how long the project stays on Option 2
  (QEMU emulation).
- A related data point surfaced during RFC review: `hipsolver`'s existing
  `ppc64` → `POWER8` OpenBLAS handling (see Current State, positive
  precedent) was added to `rocm-libraries` *after* the `rocm-libraries`
  commit TheRock had pinned at the time of the original audit — i.e. real,
  `ppc64le`-specific work has already landed upstream, independent of this
  RFC. Worth checking the authorship/linked issue of that change as a
  possible lead on the question above.

______________________________________________________________________

## Appendix A: Detailed Per-Module Findings

### Audit Reference Commits

The original audit (2026-10-27/28) was performed against TheRock's
submodule pins at that time, and subsequently cross-checked against each
repository's `develop` HEAD to confirm nothing had materially changed in the
interim:

| Repo                  | Commit used for original audit             | Cross-checked against `develop` HEAD                                                                                                                                                                                                                                                                                                  |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ROCm/rocm-systems`   | `2a8fffb7d64c59f83fe0d2efa22ed31d87d8adbf` | `d47f191a352b3726133bda4a59d79204b514b4a5` — no discrepancies; all findings hold unchanged (minor line-number drift only).                                                                                                                                                                                                            |
| `ROCm/rocm-libraries` | `8a29d2e82b93f5065c7fe5630248d3689f0c3f06` | `c75cafc00812b2272110fd90ed9aaf1e38bbef95` — RPP's two cited blockers were found inaccurate at *both* the pin and `develop` (see RPP note above and below, not a staleness issue); `hipsolver`'s `ppc64`/`POWER8` precedent was absent at the pin but present on `develop` (real upstream change in the interim, not an audit error). |

This RFC document (in TheRock) was committed from commit
`9d5bc51330c08dbdfbe7e3f79cc7d72698b9425b`. TheRock's own submodule pins
move continuously (automated bump PRs land multiple times a week); the
commits above are the ones actually read during this audit, not necessarily
TheRock's current pin at whatever point this document is later read.

Every module in `rocm-systems` and `rocm-libraries` was swept for host-arch
(not GPU-arch) assumptions blocking a `ppc64le` build.

### rocm-systems

Modules swept: `clr`, `hip`, `hipother`, `hip-tests`, `hrr`, `cuid`,
`hipfile`, `rocr-runtime`, `rocr-debug-agent`, `runtimes/api-headers`,
`aqlprofile`, `rocprofiler`, `rocprofiler-compute`, `rocprofiler-register`,
`rocprofiler-sdk`, `rocprofiler-systems`, `rocprof-trace-decoder`,
`roctracer`, `rocdbgapi`, `rdc`, `rocm-smi-lib`, `amdsmi`, `rocminfo`,
`rocm-core`, `rccl`, `rccl-tests`, `rocdecode`, `rocjpeg`, `rocshmem`,
`emulation/*`, `experimental/*`, `profilers/profiler-hub`, `shared/*`,
`python/*`, `tools/systems_pr_bot`.

**Confirmed blockers (hard compile/configure failures):**

- **`rocprofiler-systems`** —
  `source/lib/rocprof-sys-rt/CMakeLists.txt:38` unconditionally compiles
  vendored x86-64 assembly (`RTthread-x86-64.c`, `RTtlsgetaddr-x86.S`, a
  Dyninst runtime-instrumentation shim). This module's own platform-detection
  layer (`cmake/platform.cmake`, `cmake/dynsysname`) already correctly
  autodetects `ppc64_linux`/`ppc64le`, and the sibling vendored
  `external/dyninst` copy in the same module already supports `ppc64le` per
  its `DyninstPlatform.cmake` arch list — but no `RTthread-ppc64.c` /
  `RTtlsgetaddr-ppc64.S` exists in this vendored RT-lib copy. This is a
  missing source file, not a missing CMake guard.
- **`rccl`** — `src/include/timer.h:12` unconditionally includes
  `<x86intrin.h>` and calls `__rdtsc()` with no arch guard (only a
  `#if ENABLE_TIMER` feature-flag guard). This header is pulled into core
  (non-test) library sources: `src/transport.cc`, `src/proxy.cc`,
  `src/plugin/profiler.cc`.
- **`rocr-runtime`** (test/sample code only, not the library itself) —
  `rocrtst/samples/rocm_async/hsatimer.hpp:17` and
  `samples/common/hsatimer.h:27-29` both unconditionally include
  `<x86intrin.h>` (the former with no guard at all, the latter under
  `__GNUC__`, neither with an arch guard), unlike
  `rocrtst/common/hsatimer.cc` which *is* correctly guarded with
  `#if defined(__x86_64__) || defined(__i386)`.

**Minor gaps (silent no-ops or narrow path issues — not build blockers):**

- `clr/rocclr/os/os_posix.cpp:781-797` — `Os::currentStackPtr()`'s
  x86/ARM-only inline-asm chain falls through to an empty asm string on
  `ppc64le`, leaving the returned stack-pointer value uninitialized. Builds
  fine, silently wrong at runtime — the one correctness (not compile) bug
  found in this audit. Elevated to a required v1 fix (Requirement R7) rather
  than left as opportunistic cleanup, despite not blocking compilation.
- `clr/opencl/cmake/FindAMD_ICD.cmake:46` and both
  `rocm-smi-lib`/`amdsmi` `cmake_modules/help_package.cmake` hardcode
  `/usr/lib/x86_64-linux-gnu` as a (non-exclusive) search path — degrades
  gracefully via other paths on RPM-style hosts, but could miss libs on
  Debian/Ubuntu `ppc64le` multiarch layout (`powerpc64le-linux-gnu`).
- `rocr-runtime/runtime/hsa-runtime/CMakeLists.txt:314` — `IMAGE_SUPPORT`
  defaults OFF on `ppc64le` (only auto-enables for
  `i?86|x86_64|amd64|AMD64|loongarch64`) — a feature narrowing (HSA
  image/texture extension), same class as the ESMI/`rocattach` narrowings
  below, not a bug.
- `rocr-runtime/libhsakmt/CMakeLists.txt:510` hardcodes
  `/opt/amdgpu/lib/x86_64-linux-gnu` for Ubuntu static-lib linking (should be
  `powerpc64le-linux-gnu` on that distro) — narrow, Ubuntu-specific.
- `rccl/src/proxy.cc:887-890` — CPU-yield hint has x86/aarch64 branches but
  no `ppc64le` one; silently no-ops (perf-only).
- `rocshmem/src/assembly.hpp:41` — an unconditional x86-only `asm volatile("sfence")` `SFENCE()`, but it's dead code with zero call sites
  today — a landmine if ever wired up, not a current blocker.
- `rocprof-trace-decoder`'s AVX "quick scan" trace decoder already has a
  documented non-x86 fallback (`GFX9_QUICK_SCAN_HAS_X86 0` →
  `SCAN_NOT_IMPLEMENTED`) — compiles clean, just loses a fast-path.
- `projects/hip-tests/samples/2_Cookbook/{16,17}_*/CMakeLists.txt` hardcode
  `x86_64-linux-gnu` as a clang target triple in two assembly-pipeline
  *sample* targets — sample-only, trivial fix.

**Existing positive precedent:** `rocr-runtime`'s `core/util/utils.h`
intrinsics shim; `emulation/rocjitsu`'s `CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64"` branch and generic `rj_sanitizers.cmake` arch derivation;
`rccl/src/include/gdrwrap.h`'s `__PPC__` write-combine-fence branch;
`rdc/rdc_libs/rdc/src/RdcPerfTimer.cc`'s live `__powerpc64__` branch
(`<sys/platform/ppc.h>`); `rocdbgapi`'s `check_cxx_source_compiles` probe
explicitly designed around PPC64's `LL64` type layout; and
`rocprofiler-sdk`'s vendored `pybind11` already special-casing `ppc64le`.
Feature narrowings that are correct, not bugs: `amdsmi`'s ESMI (AMD
EPYC-only CPU management library) and `rocprofiler-sdk`'s `rocattach` are
already disabled via existing `x86_64`-only guards.

**kpack** (`rocm-systems/shared/kpack`) deserves its own note: it hand-rolls
raw ELF parsing (via Python `struct.unpack`, not a library like pyelftools)
because its job is literal ELF surgery — splitting embedded GPU device-code
out of host binaries. `python/rocm_kpack/elf/types.py` already defines
`EM_PPC64 = 21`, `R_PPC64_RELATIVE = 22` (commented "ppc64le" on the
preceding line), and an `ArchConfig` entry with the correct 64KB PPC64 page
size (vs 4KB for x86_64) — deliberate prior design work, not a stub. Its
64-bit/little-endian assertions are compatible with `ppc64le` specifically
(would reject big-endian classic `ppc64`, out of scope here). Gap: zero
PPC64 test fixtures exist anywhere in kpack's own test suite — this is real,
unvalidated code, not confirmed-working. The C++ runtime side
(`runtime/src/*.cpp`) has no architecture-specific ELF logic at all — it's
compiled natively per-host using system structs, so it's architecture-agnostic
by construction.

Everything else — `hip`, `hipother`, `hip-tests` (library itself), `hrr`,
`cuid`, `hipfile`, `rocr-debug-agent`, `runtimes/api-headers`, `aqlprofile`,
`rocprofiler`, `rocprofiler-compute`, `rocprofiler-register`,
`rocprofiler-sdk` (core), `roctracer`, `rocminfo`, `rocm-core`, `rccl-tests`,
`rocdecode`, `rocjpeg`, `emulation/mirage`, `experimental/*`,
`profilers/profiler-hub`, `shared/*` (other than kpack), `python/*`,
`tools/systems_pr_bot` — came back clean or already-correctly-guarded.

### rocm-libraries

Modules swept: `rocblas`, `hipblas`, `hipblas-common`, `hipblaslt`,
`hipsparse`, `hipsparselt`, `rocsparse`, `rocsolver`, `hipsolver`,
`rocalution`, `hiptensor`, `hipthreads`, `rocfft`, `hipfft`, `rocrand`,
`hiprand`, `rocprim`, `hipcub`, `rocthrust`, `rocwmma`, `composablekernel`,
`miopen`, `hipdnn`, `rpp`, all 4 `dnn-providers/*`, and
`shared/{tensile,rocroller,mxdatagenerator,origami,primbench,stinkytofu}`.

**Confirmed blockers (hard compile/configure failures):**

- **`rocfft`** — `clients/tests/CMakeLists.txt:~218-220` (inside
  `FFTW_CMAKE_ARGS_COMMON`), when building the test client's FFTW reference
  dependency — which happens automatically whenever system FFTW isn't
  found, the default on a bare container with no `fftw-devel` installed —
  passes `-DENABLE_AVX2=ON` unconditionally into FFTW 3.3.9's own CMake
  configure, with no host-arch guard anywhere in the enclosing
  `if(BUILD_FFTW OR NOT FFTW_FOUND)` block. FFTW's own CMake build gates
  `ENABLE_AVX2` to x86 targets and will hard-fail when asked for AVX2 on a
  non-x86 target. No other AVX/SSE flags or unguarded intrinsics exist
  anywhere else in rocFFT's own source.
- **`composablekernel`** — a second, more severe AVX blocker, gated only on
  GPU target family (`gfx95`/`gfx125` — mainstream CDNA3+/MI300 targets) with
  zero host-arch guard: `test/ck_tile/data_type/CMakeLists.txt:11,15` apply
  `-mavx` unconditionally; `test/ck_tile/data_type/CMakeLists.txt:22-48` run
  a `check_cxx_source_runs` AVX-512 capability probe whose fallback branch
  still hardcodes `-mavx2` regardless of the probe result (line 48) — no
  non-x86 path exists at all; `test/data_type/CMakeLists.txt:59,66,73,80,87, 112,119` and `test/gemm_mx/CMakeLists.txt:6` repeat the same
  `-mavx512f`/`-mavx` pattern; `example/67_gemm_microscaling/CMakeLists.txt:77`
  applies `-mavx512f`.
- **`shared/tensile`** — `Tensile/BuildCommands/SharedCommands.py:52`
  hardcodes the clang offload-bundler host triple as the literal string
  `host-x86_64-unknown-linux-gnu`. Since Tensile underlies rocBLAS's and
  hipBLASLt's kernel generation, this would block those libraries' builds on
  `ppc64le` too.

**Minor gaps (not build blockers):**

- `rocalution/src/CMakeLists.txt:160,180,187` and ~15 spots in
  `hipthreads/examples/*` apply `-march=native` unconditionally. Accepted by
  both GCC and clang on `ppc64le` (auto-detects POWER9/10) — not a blocker,
  just not arch-tuned.
- `rocwmma/test/CMakeLists.txt:40-47` applies `-mcmodel=large` to everything
  except arm/aarch64 (so it does apply on `ppc64le`), but this is a
  recognized, documented flag on GCC/clang for `ppc64`/`ppc64le` — silently
  accepted, not a failure.
- `shared/tensile/Tensile/cmake/TensileConfig.cmake:380` hardcodes
  `libclang_rt.asan-x86_64.so` for ASan runtime lookup — only hit on
  ASan-enabled test builds, same pattern repeated in rccl, rocdecode,
  rocshmem.

**Existing positive precedent:** RPP already has a complete SIMDE-based
portability layer for non-x86 hosts — initially miscategorized as a blocker
in an earlier audit pass. `rpp/api/rppdefs.h` and
`rpp/src/include/common/cpu/rpp_cpu_simd_load_store.hpp` both gate their
`<immintrin.h>`/`<x86intrin.h>`/`<smmintrin.h>` includes behind
`#elif defined(__x86_64__) || defined(_M_X64)`, falling back to
`#include <simde/x86/avx2.h>`/`f16c.h`/`fma.h` with
`SIMDE_ENABLE_NATIVE_ALIASES` on other hosts (with an explicit
`__HIP_DEVICE_COMPILE__` guard to avoid an `altivec.h` "vector" keyword
clash during GPU compilation passes). The top-level `rpp/CMakeLists.txt`
(~lines 233-257) does `pkg_check_modules(SIMDE REQUIRED simde)` and wires an
imported `simde::simde` target into the library build on non-x86 hosts.
`add_fma_property()` (`src/modules/CMakeLists.txt:54-58`) already
arch-guards `-mno-fma` to `x86_64|amd64|AMD64`, falling back to the portable
`-ffp-contract=off` otherwise. Of the 92 files in `tensor/cpu/kernel/` using
`_mm*`/`__m*` intrinsics (82 of 92, 70 specifically AVX2), zero include an
x86 header directly — all route through the guarded fallback chain above.
The one unresolved item is whether the `simde` package itself is available
on the `ppc64le` build image; not yet confirmed (see Appendix B, Phase 0).

`hipsolver/CMakeLists.txt:206-209` also already forces OpenBLAS
`TARGET=POWER8` on `CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64"` (vs `GENERIC`
otherwise, citing OpenBLAS#5987) and is confirmed complete — note this was
absent at the commit originally audited and added upstream since (see
"Audit Reference Commits" above and Open Questions). `rocsolver` only does a
plain `find_package(LAPACK)` with no vendored OpenBLAS build of its own, so
it doesn't need to duplicate this fix. `hipblas` client/test CMakeLists
already gates `-mf16c` behind an explicit `x86_64` check, and
`hipblas/clients/include/type_utils.h` already guards its `<immintrin.h>`
include behind `__x86_64__`/`__i386__`. `hipsparse` has no `-mf16c`/`-mfma`
flags anywhere, so nothing to gate there. `hipsparselt` does apply `-mf16c`
in several client/library CMakeLists, but gated only by compiler-ID checks
(`Clang`/`GNUCXX`) plus a `check_cxx_compiler_flag` probe — not an explicit
arch check, though the probe itself will simply fail (and thus no-op) on a
compiler that doesn't support the flag, which incidentally covers most
non-x86 toolchains too.

Everything else — `rocblas`, `hipblas-common`, `hipsparse`, `rocsparse`,
`hiptensor`, `hipfft`, `rocrand`, `hiprand`, `rocprim`, `hipcub`,
`rocthrust`, `miopen`, `hipdnn`, all 4 `dnn-providers/*`, and
`shared/{rocroller,mxdatagenerator,origami,primbench,stinkytofu}` — came back
clean, with no x86-only build assumptions found.

### Packaging detail

- No existing non-`x86_64` packaging precedent at all, not even for
  `aarch64` — `aarch64`/`arm64` only appear at the CMake/build-topology
  level, never in deb/rpm/wheel packaging code.
- `build_tools/packaging/linux/package.json` hardcodes `"Architecture": "amd64"` (deb) and `"BuildArch": "x86_64"` (rpm) as literal strings across
  81 package entries, feeding `debian_control.j2`/`rpm_specfile.j2`
  directly via `deb_package.py:351` / `rpm_package.py:181`.
- `build_tools/packaging/python/mirror_python_dependencies.py:34` hardcodes
  an allowed-platform-tag allowlist (`{"any", "linux_x86_64", "win_amd64"}`)
  plus a `manylinux*_x86_64` suffix check (line 183).
- `build_tools/packaging/linux/get_url_repo_params.py` hardcodes `/x86_64/`
  in 7 repo-path-building f-strings (lines 374, 379, 383, 422, 427, 434,
  435).
- Tarball/wheel filenames (e.g. `therock-dist-linux-gfx94X-dcgpu-...`,
  `rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl`): the wheel case has an
  existing platform-tag suffix to substitute into; the tarball case has no
  arch component at all today (see main body, Packaging & Versioning).

### CI / build matrix detail

- The GPU-family matrix in `build_tools/github_actions/amdgpu_family_matrix.py`
  already has a build-but-don't-test pattern (`"tests_on_trigger": []`,
  `"test-runs-on": ""`), used 18 times today (e.g. the `gfx90a`/`windows`
  entry at lines 447-456).
- `BUILD_RUNNER_LABELS["linux"]` (amdgpu_family_matrix.py:121-137) only
  lists `aws-linux-scale-rocm-*` self-hosted x86_64 runners — no
  architecture dimension exists here at all today.
- The Linux build container image is a hardcoded x86_64 literal in 40+
  files (every `multi_arch_build_portable_linux*.yml`, several
  `rocm-systems`/`rocm-libraries` workflow and action files,
  `build_tools/linux_portable_build.py:148`,
  `build_tools/github_actions/manylinux_config.py`, the composable_kernel
  Dockerfile, etc.), all pointing at
  `ghcr.io/rocm/therock_build_manylinux_x86_64`
  (`dockerfiles/build_manylinux_x86_64.Dockerfile:12`, `FROM quay.io/pypa/manylinux_2_28_x86_64`). PyPA publishes an official
  `manylinux_2_28_ppc64le` counterpart.

## Appendix B: Implementation Change List

### Phase 0 — Build environment

- `dockerfiles/build_manylinux_ppc64le.Dockerfile` — drafted; exploratory
  build in progress using Option 2 (QEMU user-mode emulation). Confirmed
  working: `manylinux_2_28_ppc64le` base image resolves `gcc-toolset-13`
  (13.3.1), `cmake` (3.26.5, note: must repoint `/usr/local/bin/cmake`,
  which the base image pre-bakes at a newer, incompatible 4.x), `ninja`
  (1.8.2), `git`/`git-lfs` cleanly via yum/EPEL.
- Same Dockerfile — verify/add the `simde` pkg-config package. RPP's
  existing portability path (Appendix A) depends on it being resolvable via
  `pkg_check_modules(SIMDE REQUIRED simde)` at configure time. If
  unavailable via yum/EPEL on `manylinux_2_28_ppc64le`, vendoring or
  building it from source may be needed. Not yet confirmed either way.
- New `publish_build_manylinux_ppc64le.yml` workflow (mirroring the x86_64
  one) — not yet created.
- `compiler/pre_hook_amd-llvm.cmake:101` (`LLVM_TARGETS_TO_BUILD "AMDGPU;Native;SPIRV"`) — **only** needs a change if Option 3
  (cross-compilation) is adopted; no change needed for Option 1 or 2, since
  `Native` already resolves correctly in both.

### Phase 1 — Unblock the build

- `rocm-systems/projects/rocprofiler-systems/source/lib/rocprof-sys-rt/CMakeLists.txt:38`
  — gate off on `ppc64le`, or port `RTthread-ppc64.c`/`RTtlsgetaddr-ppc64.S`
  from upstream Dyninst.
- `rocm-systems/projects/rccl/src/include/timer.h:12` — guard or replace
  `__rdtsc()` with a portable cycle-counter abstraction.
- `rocm-systems/projects/rocr-runtime/rocrtst/samples/rocm_async/hsatimer.hpp:17`
  and `samples/common/hsatimer.h:27-29` — add the same arch guard their
  sibling `hsatimer.cc` already has.
- `rocm-libraries/shared/tensile/Tensile/BuildCommands/SharedCommands.py:52`
  — parameterize/detect the host triple instead of hardcoding
  `host-x86_64-unknown-linux-gnu`.
- `rocm-libraries/projects/rocfft/clients/tests/CMakeLists.txt:~218-220` —
  guard `-DENABLE_AVX2=ON` to `x86_64` only (mirroring FFTW's own
  `ENABLE_VSX`/`ENABLE_ALTIVEC` options), or pre-install system FFTW on the
  `ppc64le` build image to avoid the `ExternalProject_Add` path entirely.
- `rocm-libraries/projects/composablekernel/test/ck_tile/data_type/CMakeLists.txt:11,15,22-48`,
  `test/data_type/CMakeLists.txt:59,66,73,80,87,112,119`,
  `test/gemm_mx/CMakeLists.txt:6`,
  `example/67_gemm_microscaling/CMakeLists.txt:77` — guard all
  `-mavx`/`-mavx512f`/`-mavx2` flags to `x86_64` only; fix the fallback
  branch that hardcodes `-mavx2` regardless of probe result.
- **Required (Requirement R7):** `clr/rocclr/os/os_posix.cpp:781-797` — fix
  `currentStackPtr()`'s uninitialized stack-pointer fallback on `ppc64le`.
- Opportunistic, non-blocking: `clr/opencl/cmake/FindAMD_ICD.cmake:46`,
  `rocm-smi-lib`/`amdsmi` `cmake_modules/help_package.cmake`,
  `rocr-runtime/libhsakmt/CMakeLists.txt:510` — add
  `powerpc64le-linux-gnu` search paths alongside the existing
  `x86_64-linux-gnu` ones.

### Phase 2 — CI matrix wiring

- `build_tools/github_actions/amdgpu_family_matrix.py` — add `ppc64le` to
  `BUILD_RUNNER_LABELS`; add a `ppc64le` family-matrix entry using the
  existing `tests_on_trigger: []` pattern.
- `build_tools/github_actions/configure_multi_arch_ci.py` — thread an
  `arch` parameter through to `linux_build_config`/`windows_build_config`.
- ~40 files referencing `ghcr.io/rocm/therock_build_manylinux_x86_64`
  (every `multi_arch_build_portable_linux*.yml`, several
  `rocm-systems`/`rocm-libraries` workflow/action files,
  `build_tools/linux_portable_build.py:148`,
  `build_tools/github_actions/manylinux_config.py`, composable_kernel's
  Dockerfile, etc.) — parameterize the image reference by arch.

### Phase 3 — Packaging (tarball, rpm, deb land in Phase 1; wheel is Phase 2)

- `build_tools/packaging/linux/package.json` — parameterize
  `Architecture`/`BuildArch` fields (81 entries) by build arch (`amd64` →
  `ppc64el` deb, `x86_64` → `ppc64le` rpm). Covers both rpm and deb in one
  change (Phase 1).
- `build_tools/packaging/linux/get_url_repo_params.py:374,379,383,422,427, 434,435` — parameterize the `/x86_64/` repo-path strings (Phase 1).
- Tarball naming scheme — add an arch segment only for non-`x86_64` builds,
  per the decision in the main body; `x86_64` tarball names stay unchanged
  (Phase 1).
- `build_tools/packaging/python/mirror_python_dependencies.py:34,183` — add
  `linux_ppc64le`/`manylinux_2_28_ppc64le` to the platform-tag allowlist
  (Phase 2).

## Appendix C: Build Environment Options — Effort Breakdown

The pros/cons above can read as if Options 1 and 2 are near-zero-effort next
to Option 3. That's only true for one narrow claim — `LLVM_TARGETS_TO_BUILD`'s
`Native` auto-resolution needs no change under either (confirmed: traced to a
`uname`-derived string match in `compiler/amd-llvm/llvm/cmake/config-ix.cmake`,
and QEMU user-mode's emulated `uname` reports `ppc64le` identically to real
hardware for this mechanism — not yet confirmed end-to-end on an actual build).
It is not true of the total engineering cost. Breakdown:

**Common to all three options** — none of this is option-specific, and none
of it is optional regardless of which environment is chosen:

- The 6 confirmed hard blockers (Phase 1 above), plus confirming `simde`
  package availability for RPP (Phase 0).
- Phase 2 CI code wiring (`BUILD_RUNNER_LABELS`, `configure_multi_arch_ci.py`
  arch threading, the ~40 hardcoded `...manylinux_x86_64` image references)
  — identical shape regardless of option; only the runner pool/image tag
  each resolves to differs.
- Phase 3 packaging (tarball arch segment, rpm/deb arch fields, wheel
  platform-tag allowlist).

**Common to Options 1 and 2 only** (not Option 3, which stays on an
`x86_64` build host and wouldn't run a native `ppc64le` container):

- `dockerfiles/build_manylinux_ppc64le.Dockerfile` — drafted (3.3KB vs. the
  `x86_64` Dockerfile's 6.2KB), real but incomplete work, not placeholder
  scaffolding. It's a plain `FROM quay.io/pypa/manylinux_2_28_ppc64le` image
  with nothing QEMU-specific baked in — the emulation requirement is about
  which host runs `docker build`/`docker run`, not the image contents, so
  Option 1 would run this exact same Dockerfile natively, unchanged. Gap vs.
  the `x86_64` image: `install_cmake.sh`/`install_ninja.sh`/
  `install_ccache.sh`/`install_gh.sh`/`install_awscli.sh`/`install_rust.sh`/
  `install_patchelf.sh`/`install_googletest.sh`/`install_shared_pythons.sh`
  have no `ppc64le` case; the draft substitutes yum packages for some of
  these instead — an explicitly-flagged version-drift tradeoff versus the
  `x86_64` image's pinned installer scripts, not a finished parity state.
- New `publish_build_manylinux_ppc64le.yml` workflow — not yet created.

**Option 1 only:**

- Procuring and registering new self-hosted `ppc64le` runner(s) — real
  hardware or a full-system VM — in `BUILD_RUNNER_LABELS`. Infrastructure
  acquisition, not a code change.
- Open product decision, not resolvable from code: whether Option 1 builds
  inside the same manylinux container (run natively, no emulation) or goes
  fully bare-metal with no container — determines whether the ~40-file
  image-reference wiring above even applies to this path.

**Option 2 only:**

- A `binfmt_misc` registration step on the existing `x86_64` self-hosted
  runner pool — reuses current infrastructure, no new hardware procurement.
  Already validated in practice: the Phase 0 exploratory build confirmed
  `gcc-toolset-13`/`cmake`/`ninja`/`git`-`lfs` install cleanly via yum/EPEL
  on the `manylinux_2_28_ppc64le` base under this path.

**Option 3 only:**

- New LLVM `CrossCompile.cmake` two-stage toolchain-file bootstrap — no
  existing pattern anywhere in TheRock's own `cmake/` layer to build on.
  Genuinely new engineering, not a parameterization of existing code, and
  the dominant cost of this option (see Alternatives Considered above for
  the `try_run` re-audit showing the probe-seeding side of this is narrow,
  and for why this is a harder case than LLVM's typical documented
  cross-compile pattern).
- Note what this cost buys: unlike Options 1 and 2, it does **not** remove
  any CI-infrastructure cost — Option 2 already reuses the existing fleet
  with no new procurement. The entire return on this engineering investment
  is build-time speed (no QEMU emulation tax), nothing else.

## Revision History

- 2026-10-27: Initial draft (Nirmal Unnikrishnan)
