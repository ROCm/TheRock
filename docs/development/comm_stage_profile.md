# Controlled full COMM-stage link validation

This default-off diagnostic follows the read-only historical
[run 41147801272](https://github.com/ROCm/rockrel/actions/runs/41147801272)
investigation and the isolated rocSHMEM link experiment. It is not a shipping
optimization or a speedup claim. New runs execute only in `ROCm/TheRock`.

The isolated trial found an approximately 18.05-minute native-link reduction with
all four controlled native/image outputs matching. Its raw trial binaries differ
from older installed or published artifacts; installation, postprocessing and
preparation have not been isolated as causes, and an object/header difference has
not been established. It also omitted simultaneous RCCL work and the original
readiness stagger. A fresh common-patch baseline and observed cross-arm source,
object/header and output parity are therefore mandatory. Neither that 18.05-minute
reduction nor the earlier conditional 274.521-second complete-stage replay ceiling
is a measured full-stage or end-to-end gain for this product candidate.

## Exact intervention and controls

Both settings apply the same upstream default-one
`ROCSHMEM_NATIVE_LINK_JOBS` patch through the existing `fetch_sources.py`
`patches/amd-mainline/rocm-systems` mechanism. TheRock passes through
`THEROCK_ROCSHMEM_NATIVE_LINK_JOBS`. Setting 1 retains existing behavior;
setting 2 adds one PRIVATE `--offload-jobs=2` argument to each of
`rocshmem_info` and `rocshmem_functional_tests`. The upstream helper rejects an
unsupported HIP compiler or conflicting HIP-wide jobs above 1. It never modifies
`HIP_CLANG_NUM_PARALLEL_JOBS`, compile options, dependencies, source lists,
architecture selections or the RCCL compilation pipeline.

Both settings must use one reviewed diagnostic SHA, recorded product HEAD
`5b7cb88373362ab80dc0f19605915e534f4158aa` plus recorded diagnostic base
`82d5e8c176ce0cc00b5fd1cf631ce7790be8992b`, rocm-systems pin
`a10dca087065a82f15ea89dc7d6fdd3c6bb2c246`, LLVM pin
`e095f1b3becf0c1bfdc5e7647003d60976a2460b`, inbound SDK from TheRock
run 41189458499, and package version
`10.3.0.dev0+82d5e8c176ce0cc00b5fd1cf631ce7790be8992b`. The common patch series,
patched source HEAD and `.smrev` contents are retained so dirty/revision metadata
cannot silently stand in for a link-only change.

The runner label remains `aws-linux-scale-rocm-prod`; the manylinux container
digest remains
`d666e4613c1ca919bc7d1a31a4c368740b66c34b5c19ef95ba15cd31db7783fb`. Actual runner
identity, visible allocation/affinity/cgroups, quotas/throttle/pressure, resource
use and observer overhead are captured. Visible CPU count is not a dedicated
entitlement claim. CI ccache remains enabled and read-only; release trust domains
are unchanged. Initial/final stats and timestamp/PID cache transaction fields are
retained. No release credentials, OIDC, S3/package publication or notifications
are configured by this diagnostic.

## Fresh full stage

The controller refuses an existing build directory instead of deleting or
adopting it. It fetches the fixed inbound SDK and pinned sources, then performs
the original complete COMM root configuration. No untimed
`rocshmem+configure` or object preparation runs: those would prebuild fmt/gtest
and remove their work from the seven-project baseline.

The measured command is the ordinary
`cmake --build build --target stage-comm-libs therock-artifacts -- -k0`, wrapped
by the existing external profiler. This preserves RCCL, rccl-tests,
rccl-translate, rocSHMEM, fmt, googletest and yaml-cpp configure/build/install
work, artifact split/flatten work, actual dependency-release stagger and
simultaneous RCCL compilation. All 25 RCCL and 31 rocSHMEM full target IDs,
including `gfx1250-strict`, must be retained. No extra build flags or target
filtering are introduced. RCCL's previously measured 97.47% custom-edge share
bypassing compiler-cache launchers is context, not an assumption about a future
run's exact measured percentage; those commands remain untouched.

Setup/fetch/root configure, verification, retention and CTest durations are
separate from the profiled full-stage command. Wall, exact waited-child CPU,
sampled CPU lower bounds and broader cgroup totals remain distinct. The 21
disjoint project-phase CPU records are retained and reconciled against the outer
waited-child total without scaling; the residual stays unallocated. Build CPU
includes generation, cache hits and linking, not just compilation.

## Evidence and correctness gates

Cache setup writes diagnostic logs outside the fresh build directory. Both modes
retain config and compiler-check script fingerprints and safe effective settings;
the controller requires the existing CI dev endpoint, `therock-v4` namespace and
read-only behavior. It preserves the existing compiler/library fingerprint check
and rejects endpoint overrides. Config/auth dumps are not exported. Relevant
timestamped cache transactions, manifest/result keys and hit/miss records remain
available for comparison; a read-only remote can still change externally, so
matching cache outcome and ordinary variation must be assessed before a gain claim.

Every allocation retains root and seven nested Ninja graphs/rules/commands,
expanded compilation databases (including RCCL custom commands), original hidden
Ninja logs, generated compiler metadata, native phase logs, all generated
source/header/object/archive/device-asset content manifests and artifact
file/mode/link manifests. Actual native link inputs/toolchain/config/bitcode
fingerprints and local rocSHMEM object/archive bytes are also retained.

Both rocSHMEM native binaries and the native RCCL shared library are compressed
and retained before independent image inspection. The pinned rocm-kpack parser
records every full target ID and image size/SHA256, including compressed and
concatenated RDC bundles. RCCL can contain multiple images per architecture; its
full coverage set must remain 25. Each rocSHMEM native link retains 31 images.
ELF headers, symbols, sections, dynamic tags and notes remain available for the
two native outputs. The original root packaging CTest inventory and results
must be present and nonempty. These checks do not execute GPU communication
functional/unit tests, and GPU runtime correctness remains unvalidated.

Independent collectors continue after a failed overall build or earlier
collector. Their successes/failures are recorded and the job remains failed.
An incomplete trial cannot become a performance baseline merely because some
useful outputs survived. Artifacts use 14-day retention; native and object bytes
are already compressed, with low-overhead upload compression. GitHub's shared
artifact-storage quota still needs sufficient capacity.

## Repeated controlled runs and comparison

After review, the parent can dispatch four independent allocations in 1/2/2/1
order, all at the same SHA. Set `profile_comm_stage=true` and
`profile_comm_stage_link_jobs` to the appropriate string. This is independent-host
repetition, not same-host ABBA. Prefer completing each allocation before the
next dispatch to avoid adding deliberate shared infrastructure contention.

Before combining timings, require successful full builds/postchecks and use:

```bash
python build_tools/compare_comm_stage.py \
  --baseline /path/to/width1-evidence \
  --candidate /path/to/width2-evidence \
  --output /path/to/parity.json
```

The comparison rejects a different source/SDK/version/control context, another
root configure flag, changed custom/ordinary object commands, changed generated
headers/object/archive/device-asset bytes, changed native/per-image outputs or
artifact contents. The two final argv vectors must differ solely by the new
width-two flag. It does not normalize paths, compiler inputs or binary metadata
to manufacture equivalence. Source/tool/header/object parity with the old
baseline is not assumed. Check actual quotas, cache behavior, hardware activity,
tool versions and source hashes across all four allocations as well; a parity
pass alone is not a speedup result.

Require two concurrent device-link groups per executable in the trace, lower
native/project/full-stage completion beyond observed variation, included final
artifact work, comparable aggregate work, exact artifact checks and passing
CTest. Report the info and functional links separately and retain their overlap;
do not add their wall times. Distinguish any movement of the final limiter to
RCCL translation/kpack/flatten from the link-local reduction. The historical
workflow's unchanged math branch remains a separate whole-CI limit.

The older full stage took approximately 52.5 minutes. Allow 1–2 hours per fresh
allocation including SDK/source preparation and large output inspection;
controller budget 3 hours/job timeout 4 hours are stopping limits, not measured
runtime predictions. Four allocations could consume approximately 4–8 prod
runner-hours; actual cold compilation/cache state, host load and retention cost
can differ. Do not infer a full-stage win before all controlled observations
and correctness checks are available.

## Local validation and scope

```bash
python -m unittest build_tools.tests.profile_comm_stage_test \
  build_tools.tests.rocshmem_native_link_options_test -v
```

The host-generation fixture uses the actual three pinned patched CMake files and
original functional-test source list/bitcode module with explicit external SDK
scaffolding. It configures the same source/build paths at 1 and 2 without
compilation/linking, proves 37 identical compile commands and exactly two changed
final link argv vectors, and rejects invalid/unsupported/conflicting settings.
It also verifies pinned fixture hashes and normal `git am` application. The
controller tests detect changed custom edges and object bytes, exercise the
TheRock/read-only boundary, preserve independent collectors after failures and
verify exact CPU reconciliation. They do not substitute for the real SDK,
fresh-stage, architecture-image or GPU checks.

This is a build/CI + superbuild/patch/reusable-workflow experiment under the
repository's packaging and Python/CI testing guidance. The historical run and
retained investigation are the work-tracking context; no public PR/issue or
external message is created by this preparation.
