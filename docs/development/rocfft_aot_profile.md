# Diagnose rocFFT AOT stalls in the original math-stage context

This is an opt-in diagnostic for the repeated low-CPU intervals in the first
`rocfft_aot_helper` command observed in TheRock run
[41185762902](https://github.com/ROCm/TheRock/actions/runs/41185762902).
The next unmodified phase-CPU reference is
[41190705863](https://github.com/ROCm/TheRock/actions/runs/41190705863).
No optimization or process-selection change is enabled.

The question is: **which measured phase holds the pending rocFFT requests while
the machine has almost no CPU work?** End-only elapsed totals cannot answer it;
one-second active/waiting phase occupancy can.

## Execution and controls

The `profile_rocfft_aot` boolean in `multi_arch_ci.yml` defaults to false for
dispatch and workflow-call events. When true it skips the normal workflow and
runs one Linux job on `aws-linux-scale-rocm-prod`, with the same manylinux image
digest `d666e4613c1ca919bc7d1a31a4c368740b66c34b5c19ef95ba15cd31db7783fb`.
The job and Python controller both require the repository `ROCm/TheRock`.
Never dispatch or rerun rockrel.

The controller fetches the successful compiler-runtime SDK from TheRock run
**41190705863** using `artifact_manager fetch --stage math-libs --bootstrap`.
It pins package version
`10.3.0.dev0+82d5e8c176ce0cc00b5fd1cf631ce7790be8992b` to match that reference.
Product HEAD remains `5b7cb88373362ab80dc0f19605915e534f4158aa`; the diagnostic
branch descends from `82d5e8c176ce0cc00b5fd1cf631ce7790be8992b`.
The source pin remains rocm-libraries
`f4e24d58b3f313f1a31068def1e35376e43c2f49`.

Configure the **entire gfx103X math stage** with the original stage arguments,
testing options, all seven architectures and required projects. Build the same
`stage-math-libs therock-artifacts` root targets as the reference. Do not prebuild
rocFFT dependencies or remove other projects: their initial concurrency may
trigger the failure being investigated. The source instrumentation only changes
rocFFT host-side files. All seven original helper commands and serial dependency
edges remain intact, including their normal temporary-cache behavior.

The controller explicitly applies
`patches/build-profile/rocm-libraries/0001-rocfft-aot-phase-diagnostics.patch` and
copies `build_tools/diagnostics/rocfft_profile.h` to rocFFT's shared directory.
This patch tag is not applied by normal `fetch_sources.py` operation. The source
HEAD and `.smrev` are unchanged; the applied patch and header hashes are recorded
separately. The controller fingerprints every CMake-listed generator checksum
input before and after applying the patch and refuses a change. In particular,
`rtc_compile.cpp` contributes to `generator_sum` and is deliberately untouched.
The existing in-process HIPRTC call is timed as a whole.

Remote ccache is read-only. There are no AWS credentials, publication, release,
deployment or artifact-upload-to-S3 steps. Only GitHub diagnostics are uploaded.
Local compiler/runtime caches retain normal behavior; no forced in-process,
forced subprocess, worker-limit or SQLite cache-policy change is made.

After review and push, dispatch only the diagnostic branch in TheRock:

```bash
gh workflow run multi_arch_ci.yml --repo ROCm/TheRock \
  --ref users/sa-faizal/build-profile-41147801272-rocfft \
  -f profile_rocfft_aot=true
```

The job calls:

```bash
python -u build_tools/profile_rocfft_aot.py \
  --build-dir build --output build/logs/rocfft-aot
```

The controller refuses local execution. Local testing builds only the small
host C++ timing fixture, never ROCm.

## Measurements and interpretation

`ROCFFT_BUILD_PROFILE_DIR` enables a session only in each AOT main process;
RTC subprocesses do not start sessions. Counters reset naturally for each of the
seven independent processes. Each session writes a start record, approximately
one-second snapshots and a final record. A condition-variable stop notification
avoids adding a one-second exit wait.

Snapshots record active phase counts and cumulative completion count, total
duration and maximum duration for queue waits; cache get mutex/query/copy;
generation; subprocess semaphore; in-process lock and compilation; subprocess
compilation; cache store mutex/query; pending-map mutex and future wait; producer
families; worker joining; final copy; cleanup and teardown. Queue depth,
hits/misses, source/code bytes, semaphore capacity and shared/new pending requests
provide context. Snapshot fields are independent atomic reads, not a transaction.

Timers overlap both within a thread and across threads. A future waiter overlaps
the compile thread resolving it; mutex hold overlaps its query/copy; worker join
overlaps remaining work. **Do not sum these totals into elapsed time or savings.**
Active phase occupancy during a near-zero-CPU interval is the causal discriminator.

Subprocess exceptions and fallback counts are recorded, with at most eight
160-character error excerpts. Linux pipe/spawn/wait errors retain the actual numeric
error, and nonzero child exits/signals are counted. Existing exceptions, fallback
selection and return behavior remain unchanged. This can distinguish `EAGAIN`
resource exhaustion from an expensive but successful compiler subprocess.

SQLite's existing busy timeout, connection flags, WAL request and checkpoint
behavior remain unchanged. Read-only pragma queries before worker startup record
actual journal mode, synchronous setting, autocheckpoint, page size and page
count. No busy handler or WAL hook is installed. Full generated-source RTC logging
is rejected. Temporary database, WAL, SHM and output file sizes are sampled.

The external `profile_build.py` records resource/CPU/cgroup telemetry and waited
child CPU separately from sampled process CPU. Initial NPROC/NOFILE limits and
one-second visible `pids.max/current/events` snapshots cover process/thread limits.
An independent observer reads only matching descendant AOT helpers' task wait
channels about every five seconds, with a4096-thread bound and explicit errors.
It reports per-channel counts, not stacks or guaranteed wait causes. Zero wait
channels may be running threads or inaccessible symbols. Hidden host/ancestor
limits remain unobserved.

Every phase snapshot also records Linux `getrusage(RUSAGE_SELF)` and
`getrusage(RUSAGE_CHILDREN)` cumulative user/system CPU and start-to-current
deltas. Each architecture is a separate process, so these OS counters retain
CPU from short-lived waited RTC children missed by sampling. Self includes the
reporter threads. Escaped or unwaited descendants can be omitted. Linux child
`ru_maxrss` is a maximum, never a sum; it is recorded as a high-water mark rather
than a delta. The final counter read precedes its own report-write/exit overhead.
These totals can be checked against the enclosing project's waited-child CPU
ledger without adding overlapping measurements.

The phase sampler's own thread CPU is reported in each trace. External wait
observer CPU, active probe wall time and lifetime wall time are separate artifacts.
Active wall time counts each completed probe body, including JSON write/flush,
once and excludes the one-second wait. Each row reports the cumulative total for
previously completed bodies; the final summary includes every completed body.
The host fixture reports50,000-scope timing-loop cost with and without an enabled
session, and tests a real blocked mutex, exception unwinding and prompt shutdown.
These are overhead observations, not a matched full-build perturbation estimate.
The diagnostic adds one sampler thread in the active AOT process and one external
observer thread; those extra tasks can matter if a PID limit is already nearly
exhausted. Their presence and overhead must be considered when interpreting a
limit-related reproduction.

## Evidence and validation

The `rocfft-aot-<attempt>` GitHub artifact is uploaded even on failure. It includes
source/SDK/compiler identities; source patch/header; generator input hashes;
original helper commands; generated root/rocFFT Ninja commands and graphs; all
available nested Ninja logs and project logs; external resource/wait traces;
one-second phase traces; and per-architecture database inventories.

For each shipped cache the controller runs read-only `quick_check` and inventories
sorted `(kernel_name, arch, hip_version, generator_sum, code-sha256, timestamp)`
rows. It records raw database hashes, logical hashes and entry/code-byte totals.
The helper's recorded generator checksum must match every corresponding cache,
and every architecture must have one complete trace. Inventory I/O happens
**after** the timed build and has separately recorded wall time.
The stage's existing root packaging checks run with
`ctest --test-dir build --output-on-failure` after the timed build as well.

These checks establish diagnostic completeness and retain a semantic correctness
reference. They do not establish equivalence to an unavailable prior cache or
replace existing GPU runtime tests. This run provides causal evidence; it does
not measure an optimization speedup. Any later candidate needs controlled source,
SDK, runner, cache/dataset and correctness comparisons.

Local checks:

```bash
python -m unittest build_tools.tests.profile_rocfft_aot_test -v
```

The host C++ compiler is discovered as `g++` or `clang++`; alternatively set
`ROCFFT_PROFILE_TEST_CXX` to a JSON argument array such as `["zig", "c++"]`.
The workflow requires the real host fixture to compile and pass.

## I-029 private transaction discriminator (default off)

Run41195235975 measured896.133s inside the first helper's17,204 serialized
SQLite stores, versus33.7–34.2s in later helpers. Nearby independent thread
snapshots show193 futex waiters and one journal-commit waiter during almost
zero container CPU. This identifies the serialized store path; it does not
isolate fsync from connection, checkpoint, allocation or competing storage work.

The optional `profile_rocfft_aot_batch_transaction` input applies only inside
`profile_rocfft_aot` diagnostic mode. It passes `--batch-private-aot` to the
controller. Both settings use the same diagnostic source patch and fixed SDK;
the explicit CLI sets `ROCFFT_AOT_BATCH_TRANSACTION` to0 or1 in the measured
child. The recorded mode overrides inherited settings. Normal builds do not
apply these diagnostic patches.

Only the first helper with an absent default temporary DB is eligible. Later
architecture helpers retain the original growing shared private DB and original
autocommit behavior; no database is reset between architectures. A controller
guard rejects a preexisting or symlinked temp DB before the stage starts.
The candidate begins a single `BEGIN IMMEDIATE` transaction before workers
start, commits after all workers finish, and then performs the original output
copy and cleanup. Separate `aot_begin` and `aot_commit` phases include the final
durable commit in helper wall/CPU totals. WAL, synchronousFULL, busy timeout,
checkpoint policy, worker count, generator inputs and compile commands stay
unchanged. The switch refuses nonempty caller-supplied temporary paths; only
the AOT helper's default regenerable intermediate is eligible. The library's
general runtime cache never calls the transaction boundary.

This is a reviewable experiment, not a production optimization or validated
speedup. Risks include larger WAL/transaction working sets, file growth, lock
duration and loss of an unfinished private batch on interruption. SQLite rolls
back an unfinished transaction; the helper must fail on begin/commit errors.
The final output is copied only after successful commit. The default helper
directory must have exclusive ownership, as in the original serial architecture
chain. Never run concurrent AOT helpers in the same temporary-cache directory.

For a controlled discriminator, build the helper once with both source patches
and retain its binary/compiler hashes. In the same TheRock CI allocation, run
balanced autocommit/batch pairs in separate freshly created private directories,
using the original helper argv with an empty temporary-path argument, fixed
RTC-helper path, architecture and environment. Time the complete helper,
including commit/copy/teardown. Use equal initial compiler-cache state; a second
run's warm HIPRTC/COMGR or filesystem cache must not be called an optimization.
Do not compare an all-fresh-seven baseline to the original shared-DB chain.
The first-architecture trial answers the immediate question; keep the full
seven-architecture chain and original math-stage concurrency for final validation,
with batching applied only to the first fresh helper and later helpers unchanged.

Required evidence: store-query and final-commit traces, exact self/waited-child
CPU, resource/pressure/working-set traces,17,204 compiled/store entries per
architecture, unchanged generator checksum, read-only SQLite quick_check, and
equal sorted key/code-blob inventories between controls. Raw database byte
identity is useful but semantic identity is required because page layout may
differ. Do not sum overlapping mutex waits or promise896s of CI savings.
Existing packaging checks and applicable GPU/runtime checks must still pass.
The current prior diagnostic has no retained semantic inventories because its
postprocessor failed, so a new controlled pair must establish that reference.

The metadata regression is corrected separately: compiler identity uses generated
`CMakeCXXCompiler.cmake` when the toolchain variable is absent from CMakeCache.
Independent postchecks all run and retain their status before any failure is
raised. Hidden Ninja logs and install logs are retained. A failed metadata check
continues to fail the job; it no longer discards later semantic output evidence.

The standalone host transaction fixture tests the actual candidate header with
real SQLite: private-path guard, cross-connection visibility, code-blob identity,
FULL policy, an authorizer-injected commit failure and close-time rollback.
For a reproducible host-only check, build SQLite3.51.3 from the same pinned
rocFFT source URL
`https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip`, SHA3-256
`ced02ff9738970f338c9c8e269897b554bcda73f6cf1029d49459e1324dbeaea`.
Compile `sqlite3.c` as C with `SQLITE_OMIT_LOAD_EXTENSION`, then set the paths:

```bash
export ROCFFT_PROFILE_TEST_CXX='["clang++"]'
export ROCFFT_AOT_TEST_SQLITE_DIR=/private/fixture/sqlite-amalgamation-3510300
export ROCFFT_AOT_TEST_SQLITE_LIB=/private/fixture/sqlite3.o
python -m unittest build_tools.tests.profile_rocfft_aot_test \
  build_tools.tests.rocfft_aot_transaction_test -v
```

This fixture is host correctness coverage, not a local ROCm performance run.
The SQLite fixture explicitly skips when its declared build inputs are absent;
the original required host profiler fixture remains required in CI.

### Compiler-cache control and repeated full-stage recipe

The exact LLVM pin's
[`comgr-env.cpp`215–241](https://github.com/ROCm/llvm-project/blob/e095f1b3becf0c1bfdc5e7647003d60976a2460b/amd/comgr/src/comgr-env.cpp#L215)
enables the COMGR command cache by default. `AMD_COMGR_CACHE=0` disables it;
`AMD_COMGR_CACHE_DIR` selects an explicit directory. The default path is the
platform cache directory plus `comgr`.
[`comgr-cache.cpp`139–186](https://github.com/ROCm/llvm-project/blob/e095f1b3becf0c1bfdc5e7647003d60976a2460b/amd/comgr/src/comgr-cache.cpp#L139)
uses default pruning once per hour, up to75% available disk/5GiB.
[`hiprtcInternal.cpp`192–233](https://github.com/ROCm/rocm-systems/blob/a10dca087065a82f15ea89dc7d6fdd3c6bb2c246/projects/clr/hipamd/src/hiprtc/hiprtcInternal.cpp#L192)
and its compile helper route the inspected rocFFT HIPRTC path through COMGR.
The inspected chain does not add a separate HIPRTC disk cache. This does not
claim that all ROCm runtime APIs or OS caches are controlled by that directory.

The controller now creates a new private COMGR directory before preparation,
sets `AMD_COMGR_CACHE=1`, selects it with `AMD_COMGR_CACHE_DIR`, and unsets an
inherited pruning-policy override so all controls use the pinned default. It
refuses reuse rather than deleting any existing directory. Preparation must
leave this directory empty; a recorded nonzero entry count fails before timing.
The final inventory records file names/sizes and filesystem capacity/device
without uploading cached compiler binaries. COMGR remains enabled and can warm
naturally throughout the seven-helper chain. This common control differs from
the earlier diagnostic's unrecorded ambient COMGR cache state: compare A/B
within the new trial set, not new timings directly against the old job.
Compiler append-option overrides and verbose COMGR logging are rejected rather
than silently changing the measured work. Only a specific nonsecret knob
allowlist is recorded.

The following first-only recipe describes the frozen2550 experiment. It is
retained for reproduction; use the owned-chain discriminator below for a new
whole-chain comparison. After review/push of one exact trial SHA, launch four independent full-stage
allocations in A/B/B/A order. A is `profile_rocfft_aot=true` with
`profile_rocfft_aot_batch_transaction=false`; B changes only the latter totrue.
Use `multi_arch_ci.yml`, the same pinned branch/SHA, SDK41190705863, fixed82
package version and `aws-linux-scale-rocm-prod` label. Prefer waiting for each
job to finish before the next dispatch to avoid deliberately adding concurrent
storage load. Record actual start order and allocation identities even if the
scheduler changes launch order. No release, S3 publication or remote ccache
write is part of this diagnostic.

These are independent-host repetitions, not same-host ABBA. Check each run's
compiler/header/source hashes, complete81 project configuration context,
seven helpers, initial private-cache inventories, cpu.max/cpuset/PID limits,
runner identity and resource traces before comparing. Require all seven sorted
cache key/blob inventories to agree across A and B,17,204 entries/misses per
architecture, unchanged generator checksums and passing postchecks. Compare
first-helper store and final-commit time, complete first-helper wall/CPU,
subsequent shared-DB helpers, rocFFT project time and stage time separately.
Two samples per arm describe observed variation; they do not support precise
confidence intervals or universal speedup claims. A slow/failed allocation must
be explained from traces, not silently discarded. GPU correctness remains
unvalidated: the pinned gfx103X test configuration has empty `test-runs-on`.
That does not prove physical runner absence. Do not substitute another GPU
without recording a distinct validation scope.

### Owned growing-chain discriminator (I-029)

The first-only candidate in TheRock run41202568671 retained all seven shipped
DB hashes but moved most of the serialized store delay to the next autocommit
helper. Its first helper improved by880s, while the measured math stage improved
by105s and the complete job was39s longer than run41200240578. One pair is not
a replicated end-to-end speedup. The next discriminator batches each helper's
stores within the same growing, private regenerable intermediate, preserving
all seven architecture outputs and the original helper chain.

`profile_rocfft_aot_cache_mode` defaults to `legacy`; normal CI and the previous
boolean first-only behavior remain unchanged. New choices are
`owned-autocommit`, `owned-first`, and `owned-all`. Combining an owned choice
with the legacy batching boolean is rejected. The controller creates one new
directory below the original `library/src` on the same filesystem, writes a
random nonsecret ownership ID, and records its run ID, marker hash and device.
The private DB is absent initially and reused without resets by all seven
original Ninja commands. No CMake/source pointer, required target, architecture,
kernel generator input, worker count or RTC compile-selection policy changes.

The helper accepts an owned path only with an empty original temp-cache argv,
an absolute canonical parent, a matching regular single-link owner marker, and
no symlink or hardlink substitution of DB/WAL/SHM files. This guards accidental
reuse, not adversarial concurrent modification. `ROCFFT_AOT_BATCH_TRANSACTION=all`
requires ownership; it begins before workers and commits after their join,
before output copy and cleanup. Existing runtime/custom-path cache behavior is
unchanged. WAL, FULL synchronous policy, auto-checkpoint and busy policy remain
unchanged; each durable COMMIT is included in the trace and helper wall time.

After parent review of one exact new SHA, compare independent allocations in
A/B/B/A order, waiting for each to complete. Set `profile_rocfft_aot=true`, keep
`profile_rocfft_aot_batch_transaction=false`, and select
`profile_rocfft_aot_cache_mode=owned-autocommit` for A or `owned-all` for B.
Both arms run the same source/header/binary and use the same SDK41190705863,
package version82, pinned container, prod runner label, original full math-stage
context and initially empty private COMGR directory. Their only policy
difference is the explicit transaction setting. Random owner/path identities
and actual hardware/allocation metadata necessarily differ and are recorded.
These are independent-host repetitions, not a same-host crossover.

Require completed traces for each architecture with17,204 generated/compiled/
stored kernels, unchanged generator checksum, zero subprocess/fallback/store
errors, expected begin/commit counts and unchanged WAL/FULL settings. Allow the
subprocess/in-process split to vary under the unchanged try-lock policy. Require
all seven shipped key/code-blob inventories and raw DB hashes to agree between
arms. The final owned temp DB must pass `quick_check` and contain120,428 rows:
17,204 for each of the seven architectures. Record first-absent/later-present DB
state, marker identity, final file hash/size and ownership. Retain all49 existing
packaging/linkage checks; GPU numerical execution remains a separate unpassed
validation requirement. Compare helper store plus COMMIT, project/stage wall,
exact self/child CPU, I/O and complete job time separately. Never add overlapping
thread timers or infer savings from higher utilization alone.

Local host fixtures cover fresh/reused ownership, mode conflicts, bad markers,
custom-path rejection, real symlink/hardlink DB and sidecar guards, unchanged
legacy defaults, FULL visibility, real COMMIT failure and close-time rollback.
Controller fixtures check unique creation/no deletion, complete worksets and
failure/policy guards. Linux host fixtures exercise actual filesystem guards;
they neither compile ROCm nor execute GPU binaries. This is a default-off
diagnostic candidate for the current investigation, not a proposed release
durability-policy change or an already validated optimization.
