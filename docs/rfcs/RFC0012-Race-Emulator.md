---
author(s): James Newling (newling)
created: 2026-03-23
modified: 2026-03-23
status: draft
discussion: https://github.com/ROCm/TheRock/discussions/4124
---

> PR demonstrating the RFC: [rocm-libraries#4968](https://github.com/ROCm/rocm-libraries/pull/4968)
> RFC copied from [shared/race-emulator/README.md](https://github.com/newling/rocm-libraries/blob/race_emulator_integrate/shared/race-emulator/README.md).
> Note: this RFC is written as if the project has already landed, since
> the content is taken from the README of a working implementation.

# Race Emulator

A CPU-only emulator for AMD GPU assembly that detects race conditions.
It is currently used by hipBLASLt to validate GEMM kernel assembly.

## Motivation

AMD does not currently have a GPU emulator for public ISAs. A
long-term goal should be to unify this project with existing closed
source emulators, and to ensure that emulation of all public ISAs
is open source. This project fills an immediate gap: providing race
condition detection for AMD GPU assembly in use today.

Race conditions in GPU assembly are difficult to avoid, particularly
in direct-to-assembly code generation systems. They can lie dormant
through many iterations of a program, only to surface under specific
hardware scheduling states, producing non-deterministic results that
are difficult to reproduce and debug. The race-emulator serves as an
early detection system, providing diagnostics at the assembly level
that a developer can use to find and eliminate races.

The approach taken here, CPU-side emulation, is one of several
possible approaches. See the
[Alternative approaches](#alternative-approaches) section for a
discussion of some of the others. In the long-term, the race
detection tooling offered by AMD will likely draw from multiple
approaches, and this project's scope will evolve accordingly.
Feedback and suggestions are very welcome.

## What is a race condition?

A race occurs when the value read from a register, local memory
(LDS), or global memory is ambiguous due to unsynchronized access.
When this happens, the behaviour of a GPU program is undefined. On
AMD GPUs, correct use of `s_waitcnt` (to wait for a wave's own memory
operations to complete) and `s_barrier` (to synchronize waves
within a workgroup) is required to avoid races.

## Goals

Given AMD GPU assembly code, detect all intra-workgroup races. Some
examples are:

1) A single thread issues a load from global memory to a register, but
   does not wait for the load to complete before using the register.
2) Two threads in different waves (but the same workgroup) write to the
   same address in LDS, without using a workgroup barrier to specify
   their relative order. The LDS address is subsequently read and used
   by a third thread.

## Maybe goals

- Make the analysis value sensitive. For example in case (2) above if
  the two writing threads write the same value, it could be considered
  race free, because the value that the third thread reads is not
  ambiguous (it doesn't matter which thread wrote as the value is the
  same).
- Detect inter-workgroup race conditions in global memory. For example,
  atomic writes for variants of split-k GEMMs.

## Non-goals

- Emulating all AMD GPU programs. In this project, instructions will be
  added as needed for specific use cases. A hardware-based approach is a
  promising alternative (see [Level 3](#level-3)) that will be
  comprehensive.

## Current status

The majority of the effort so far has gone into emulating instructions.
The emulator supports gfx942 (MI300X) and gfx1151 (Strix), and has
numerically validated runs of

- hipBLASLt/TensileLite GEMM kernels (f32, bf16, f16)
- an IREE-generated GEMM kernel (f32)
- a HipKittens GEMM kernel (bf16 -> f32)
- a few simple HIP programs.

Being able to numerically validate all vector instructions is not
necessary for detecting race conditions, but it provides useful proof
that all instructions are correctly emulated. The project also has unit
tests for all the instructions it emulates, individually. Below we
present two simple examples of races that are currently detectable.

### Case 1: Single thread

The following assembly is modified from a HIP program where each thread
adds two integers and writes the result to global memory:

```c++
int tid = threadIdx.x;
auto v1 = data[threadIdx.x];
auto v2 = data[threadIdx.x+17];
auto v3 =  v1 + v2;
```

The modification that introduces a race is to increase the `vmcnt` wait
value from 0 to 1. As a result, the thread only waits for `v1` to be
loaded before performing the add, and so the value for the second operand
may not yet be in the register.

```
s_load_dwordx2 s[0:1], s[0:1], 0x0
v_lshlrev_b32_e32 v0, 2, v0
s_waitcnt lgkmcnt(0)
global_load_dword v1, v0, s[0:1]
global_load_dword v2, v0, s[0:1] offset:68
s_waitcnt vmcnt(1) ; <--- SHOULD WAIT FOR BOTH!
v_add_u32_e32 v1, v2, v1
global_store_dword v0, v1, s[0:1]
s_endpgm
```

The race-emulator detects the race, with diagnostic message

```asm
VGPR race detected on line 8. Conflicting events:

5     |     global_load_dword v1, v0, s[0:1]
6 --> |     global_load_dword v2, v0, s[0:1] offset:68
7     |     s_waitcnt vmcnt(1) ; <--- SHOULD WAIT FOR BOTH!
8 --> |     v_add_u32_e32 v1, v2, v1
9     |     global_store_dword v0, v1, s[0:1]
```

It highlights the 2 lines that are involved in the race with `-->`.

### Case 2: Threads in different waves

Race conditions of this sort can arise when `s_barrier` is not used to
synchronize threads in different waves (subgroups). In the following
example the LDS is used as a shared memory for threads to exchange data.

```asm
  ; Each thread loads a distinct 4 bytes from global to a vector register.
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  v_lshlrev_b32_e32 v0, 2, v0
  v_sub_u32_e32 v2, 0, v0
  s_waitcnt lgkmcnt(0)
  global_load_dword v1, v0, s[0:1]
  s_waitcnt vmcnt(0)

  ; Each thread writes its 4 bytes to LDS.
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)

  ;  s_barrier <--- MISSING BARRIER
  ; Each thread reads from LDS, from an address written by another wave.
  ds_read_b32 v1, v2 offset:1020
  s_waitcnt lgkmcnt(0)
  global_store_dword v0, v1, s[0:1]
  s_endpgm
```

The error message provided by `race-emulator` is

```asm
LDS race in byte 512 detected. Race between a pair in:

Wave 2 Lane 0:
11     |   ; Each thread writes its 4 bytes to LDS.
12 --> |   ds_write_b32 v0, v1
13     |   s_waitcnt lgkmcnt(0)

Wave 1 Lane 63:
16     |   ; Each thread reads from LDS, from an address written by another wave.
17 --> |   ds_read_b32 v1, v2 offset:1020
18     |   s_waitcnt lgkmcnt(0)
```
Above, race-emulator has detected that a thread in wave 2 is writing to
the same address that a thread in wave 1 is reading from, and that the
order is not specified. This means that the value read by wave 1 is
ambiguous.

## Implementation of the core race detection logic

Race detection tracks memory events (loads, stores) through their
lifecycle: creation, `s_waitcnt` completion, and `s_barrier` retirement.
When any instruction reads an LDS address or vector register whose
value is ambiguous (an outstanding conflicting event exists), a C++
exception is thrown immediately with a diagnostic message.

**LDS race detection** currently uses per-byte counters for fast-path
checks, with interval-based overlap scanning as a fallback. Live
events are split by direction so that RAW and WAR hazards are checked
independently. WAW detection is a future addition.

**VGPR race detection** tracks events per register, using the stored
exec mask to determine which lanes are affected. Per-register event
counts provide a fast-path. Tracking is at byte granularity within
each 32-bit VGPR so that d16 instructions do not cause false positives
when the other half is accessed independently.

**Event retirement** follows a two-stage model. When a wave executes
`s_waitcnt`, its events are marked `WAVE_COMPLETE` (safe for the
owning wave but still visible to other waves). Events are only fully
retired when ALL waves reach an `s_barrier`, ensuring that cross-wave
races are detected regardless of wave scheduling order.

Note: detecting race conditions in scalar registers is not yet
implemented.

## Performance

The emulator performs numerically validated emulation (f16/bf16
arithmetic is promoted to f32) while tracking all memory events and
detecting race conditions in both LDS and vector registers.
A 128×128×8192 f16 single-workgroup GEMM kernel (TensileLite)
completes full numerical emulation in approximately 0.1 seconds on
a single CPU thread, including all race checking. Since workgroups
are independent, multi-workgroup kernels parallelize naturally across
CPU threads.

Each assembly line is compiled into a C++ lambda on first encounter
and cached, so loop bodies avoid reparsing on subsequent iterations.

## Usage

Currently, this project does not have a tool that works as
`./race-emulator my-kernel.s`. This is because some kernel-specific
work must always be done to initialize the arguments for the kernel. In
hipBLASLt we automate this to some extent by dumping the kernel
arguments prepared for a GPU run, and use those. However for numerical
validation in the emulator, we still need to substitute GPU pointer
arguments with CPU pointers. For a full example, see the
[end-to-end tests](tests/e2e_hip_general.cpp).

## Alternative approaches

This project is still in an early stage, and may pivot direction. Some
alternative design approaches for developing a race detection tool are
outlined below, in order of the number of dependencies required.

### Level 0

Does not use LLVM for parsing assembly (or anything else). Does not use
an external emulator. Does not run on, or require, a GPU. i.e. it is
completely standalone. Pros: Full control over the implementation. Can
detect race conditions without needing a GPU. This is the approach taken
in this project.

### Level 1

Use LLVM to parse the assembly, and then run a custom emulator on the
LLVM Machine IR. Pros: Robust parsing.

### Level 2

Like level 1, but additionally use an existing emulator for AMD GPUs.
Pros: No need to reimplement an emulator. Contribute to making another
emulator more robust.

### Level 3

Execute on real hardware. The code would need to be instrumented to
record global/LDS reads and writes, as well as s_barriers and
s_waitcnts, and stream these back to CPU, for every wave. A host program
would then analyze the recorded accesses, and detect race conditions as
a post processing step. I suspect that this is the approach taken by
NVidia's [racecheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#racecheck-tool)
tool. Perhaps
[this](https://umr.readthedocs.io/en/main/index.html) open source AMD
tool could serve as a starting point. Pros: Would not need to emulate
the full GPU ISA. Faster runtime than emulation. Can detect races at the
application level, without any extra effort (if runtime instrumentation
is used).

## Integration status

race-emulator is integrated into rocm-libraries in two places. Both
require the cmake option `HIPBLASLT_ENABLE_RACE_EMULATOR` to be enabled,
which is the default when building from the monorepo (see
[Enabling race-emulator in hipblaslt](#enabling-race-emulator-in-hipblaslt)).

1. **tensilelite-client**: Race checking can be enabled in Tensile YAML
   test files by setting `CheckForRaces: 1` in `GlobalParameters`. The
   emulator runs on the generated kernel assembly after the first warmup
   iteration. See
   [`race_check_gfx942.yaml`](../../projects/hipblaslt/tensilelite/Tensile/Tests/common/gemm/race_check_gfx942.yaml)
   and
   [`race_check_custom_gfx1151.yaml`](../../projects/hipblaslt/tensilelite/Tensile/Tests/common/gemm/race_check_custom_gfx1151.yaml)
   for examples.

2. **hipBLASLt (direct assembly)**: When custom assembly kernels are
   loaded via the `HIPBLASLT_CUSTOM_ASM_DIR` environment variable, the
   race emulator automatically validates each kernel on first invocation.
   No code changes are needed — race checking is built into the direct
   assembly dispatch path. Note: this approach will change when the more
   robust custom assembly approach is integrated into hipBLASLt.

If race checking is requested at runtime (e.g. `CheckForRaces: 1` in a
YAML file) but the build was compiled without
`HIPBLASLT_ENABLE_RACE_EMULATOR`, the client will throw a runtime error
rather than silently skipping the check.

## CI

race-emulator has its own CI workflow
(`.github/workflows/race-emulator-ci.yml`) that runs on PRs touching
`shared/race-emulator/**`. It uses sparse checkout for fast iteration.
For the fastest feedback, add the `noCI` and
`skip-therockci` labels to your PR to disable Jenkins and TheRock CI,
leaving only the race-emulator jobs. The jobs are:

| Job | Purpose |
|-----|---------|
| Linux (gcc) | Release build with `-Werror` |
| Linux (clang-asan) | AddressSanitizer |
| Linux (clang-ubsan) | UndefinedBehaviorSanitizer |
| Linux (clang-tidy) | Static analysis |
| Linux (coverage) | Test coverage (HTML artifact) |
| Windows (MSVC) | Cross-platform build and test |

These jobs have been validated with intentional bug injection (see
[PR #5151](https://github.com/ROCm/rocm-libraries/pull/5151)).

The coverage job currently generates a downloadable HTML report. A
future improvement is to integrate with Codecov for inline PR
annotations (see the TODO in `race-emulator-ci.yml`).

Changes to `shared/race-emulator` also trigger the TheRock blas CI
pipeline, which builds hipBLASLt with race-emulator integration and
runs the full hipBLASLt test suite. rocJenkins runs hipBLASLt with
race-emulator disabled (see `HIPBLASLT_ENABLE_RACE_EMULATOR` in
`projects/hipblaslt/CMakeLists.txt`).

## Building

race-emulator has no dependencies beyond the C++ standard library
(C++20). GoogleTest is required for tests, and is fetched automatically
if not found.

The examples below show typical configurations. Flags like
`-DGPU_TARGETS`, compiler paths, and optional dependency flags will
vary depending on your hardware and environment.

### Standalone (for development and testing)

Build and test the library in isolation, without any other
rocm-libraries components. This is the fastest way to iterate on
race-emulator itself:

```bash
cmake -S shared/race-emulator -B build -G Ninja \
  -DRACE_EMULATOR_BUILD_TESTING=ON
ninja -C build
ctest --test-dir build
```

Run a specific subset of tests:

```bash
ctest --test-dir build -R ParserTest
```

### Enabling race-emulator in hipBLASLt

Race-emulator is auto-detected by hipBLASLt: if `shared/race-emulator`
is present in the source tree (monorepo or sparse checkout), it is
enabled by default. It can be explicitly disabled with:

```
-DHIPBLASLT_ENABLE_RACE_EMULATOR=OFF
```

When disabled, hipBLASLt builds and runs normally without any
race-emulator dependency. If race checking is then requested at runtime
(e.g. via `CheckForRaces: 1` in a YAML file, or via the
`HIPBLASLT_CUSTOM_ASM_DIR` environment variable), a runtime error is
thrown with a message indicating that the build does not support race
checking.

### As part of hipBLASLt

Race-emulator is CPU-only, so `-DGPU_TARGETS` in the commands below is
a hipBLASLt requirement (for device code compilation), not a
race-emulator one. To also build race-emulator's own unit tests within
either hipBLASLt build below, add `-DRACE_EMULATOR_BUILD_TESTING=ON` to
the cmake command.

**Tensilelite client** (for running `Tensile.sh`):

```bash
cmake --preset tensilelite \
  -S projects/hipblaslt -B build/tensilelite -G Ninja \
  -DGPU_TARGETS=gfx942 \
  -DHIPBLASLT_ENABLE_LLVM=1 \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DBoost_ROOT=/path/to/boost
ninja -C build/tensilelite
```

Note: LLVM and Boost are TensileLite dependencies, not race-emulator's.

**hipblaslt-bench**:

```bash
cmake -S projects/hipblaslt -B build/hipblaslt -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx942 \
  -DHIPBLASLT_ENABLE_BLIS=0
ninja -C build/hipblaslt hipblaslt-bench
```

Note: `-DHIPBLASLT_ENABLE_BLIS=0` is only needed if BLIS is not
installed.

### GPU hardware validation tests

The emulator's instruction implementations can be validated against real GPU
hardware. These tests run inline assembly on a GPU and compare the results
against the emulator's Wave API. Set `RACE_EMULATOR_BUILD_HW_TESTING` to the
GPU target (e.g. `gfx942`). This selects the test sources from the
corresponding `tests/gpu_validation/<target>/` directory:

```bash
cmake -S shared/race-emulator -B build -G Ninja \
  -DRACE_EMULATOR_BUILD_TESTING=ON \
  -DRACE_EMULATOR_BUILD_HW_TESTING=gfx942 \
  -DCMAKE_PREFIX_PATH=/opt/rocm
ninja -C build
ctest --test-dir build -R GpuValidation
```

These tests are not part of CI — they are a local development tool for
validating new instruction implementations against hardware.

### As part of the full monorepo superbuild

race-emulator is registered as a supported component. It is built
automatically when `ROCM_LIBS_ENABLE_COMPONENTS` includes
`race-emulator` (or `all`).

## Running with race checking

The instructions below require hipBLASLt to have been built with
race-emulator enabled (the default when building from the monorepo).

### Tensile.sh

Add `CheckForRaces: 1` to the `GlobalParameters` section of a Tensile
YAML file. For example:

```yaml
GlobalParameters:
  NumElementsToValidate: -1
  CheckForRaces: 1
```

Then run as usual:

```bash
./Tensile.sh my_test.yaml outputdir \
  --prebuilt-client=tensilelite/client/tensilelite-client
```

The race emulator will run on each generated kernel. If a race is
detected, the client will report the diagnostic and exit with a non-zero
status.

### hipblaslt-bench (direct assembly)

The `examples/` directory contains `simple_gemm.cpp`, a minimal HIP GEMM
kernel, and `simple_gemm.s`, its compiled gfx942 assembly. To regenerate
the assembly from the source (e.g. for a different target):

```bash
hipcc --cuda-device-only -S --offload-arch=gfx942 -O3 simple_gemm.cpp -o simple_gemm.s
```

Set the environment variables to point at the directory containing the
`.s`
file, then run hipblaslt-bench:

```bash
export HIPBLASLT_CUSTOM_ASM_DIR=/path/to/examples
export HIPBLASLT_ENABLE_DIRECT_ASSEMBLY=1
./hipblaslt-bench -m 128 -n 64 -k 256 -r f32_r --verify --alpha 1 --beta 1
```

On success, the output includes:

```
[DirectAssembly] Running race emulator on: /path/to/examples/simple_gemm.s
[DirectAssembly] Race emulator completed: SUCCESS
```

To demonstrate race detection, edit `simple_gemm.s` and change the
`s_waitcnt vmcnt(0)` on line 73 to `s_waitcnt vmcnt(1)`. This causes the
thread to proceed without waiting for the second global load to
complete.
Re-running hipblaslt-bench will produce a diagnostic like:

```
VGPR race detected on line 74. Conflicting events:

66     |        global_load_dword v7, v[4:5], off
67 --> |        global_load_dword v3, v[8:9], off
68     |        s_add_i32 s2, s2, -1

73     |   s_waitcnt vmcnt(1)
74 --> |        v_fmac_f32_e32 v6, v3, v7
75     |        s_cbranch_scc0 .LBB0_3
```

To disable custom assembly dispatch:

```bash
export HIPBLASLT_ENABLE_DIRECT_ASSEMBLY=0
```
