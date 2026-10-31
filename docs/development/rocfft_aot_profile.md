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
