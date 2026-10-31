# Profiler attachment readiness diagnostic

This default-off diagnostic investigates two required profiler fixture failures
in TheRock run 41230700832. It is not an optimization experiment. That baseline
remains ineligible; a passing replay does not replace its failed results.

Use `test_artifacts.yml`, `profile_attachment_diagnostic=true` and the isolated
`users/sa-faizal/build-profile-41147801272-profiler-attachment` branch with the
reviewed SHA as `ref`. Guards require ROCm/TheRock, actor sa-faizal, artifact
run 41230700832, release/CI artifacts, gfx94X-dcgpu/gfx942, comprehensive tests,
`test_labels=["test:rocprofiler-sdk"]` and runner
`linux-gfx942-1gpu-ccs-csp-ossci-rocm`. Other filters/stage exclusions are empty.
The original sanity prerequisite remains. Both component contracts, images and
security options are checked before containers are scheduled. Other callers
default off.

The unchanged profiler test script/`--enable-cdash` argument executes through a
small Python bootstrap. It validates the exact CTest entry and owned loopback URL
before CTest starts. Configured tests, parallelism of 8 and failure policy are unchanged.
Original child status remains separate from diagnostic completeness.

A private `BASH_ENV` hook records only the existing background-thread proc/grep
predicate, preserving grep exit codes 0/1/2 and suppressed stderr. Other grep calls pass
through. The original 30-iteration counter, one-second sleep, test timeouts, ignore
list and security policy remain unchanged. Hook I/O and observation can perturb
timing; a passing instrumented replay does not prove a timeout problem is fixed.

The observer checks descendant PID, creation ticks and one of two exact target
executables before thread/status/wchan and selected provider-map reads. It never
reads process environments. A positive existing ROCTx ready marker proves HIP
allocation/init completed; absence alone does not prove incomplete startup.
Unattributed/reused/unreadable identities remain gaps. A known target's later exit
is recorded as a lifecycle event, not a passing test.

Flat SDK compilers, required providers, launcher, original scripts and two target
sources are fingerprinted before/after. Missing/changed selected inputs fail the
evidence gate. This is not full transitive loader/header coverage. Generated
CTest/CMake/Ninja metadata and original submissions are retained. Visible CPU/pids
limits and pressure are not dedicated CPU entitlement.

Typed and inherited flags suppress every configured Quartz notification,
including sanity. CDash uses a nonce-protected 127.0.0.1 sink and is never forwarded.
Header/body reads have timeouts; budgets are 128MiB/request, 64 requests and 512MiB
total. Partial failures retain an index. No release/S3 publication is added.
GitHub diagnostics upload after failure with 7-day retention.

Observation sleeps 0.5s and records actual timestamps. Bounds include 128MiB JSON,
32MiB poll events, 4096 threads and 128 active identities. Waited-child CPU,
observer/sink CPU and preparation are separate; they do not measure all profiling
perturbation. No speedup/cold-cache/combined-candidate claim follows from this run.

Fixtures use real proc globs, grep outcomes/capture failures, PID attribution,
marker bounds, SDK mutations, notification guards and tiny CTest loopback uploads.
Partial headers/bodies cannot hang cleanup. They run no ROCm build/GPU workload.
