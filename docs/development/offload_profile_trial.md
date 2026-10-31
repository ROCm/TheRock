# Controlled offload compiler trial (I-022)

This opt-in experiment tests whether parallel architecture work inside one long
HIP compilation shortens the gfx103X tail. It uses `--offload-jobs=1` versus
`--offload-jobs=2` on the **original segmented radix-sort pairs object**. There is
no translation-unit split, reduced type list, or reduced architecture coverage.
This is an experimental CI path, not a production build default.

## Execution boundary and inputs

The `profile_offload_trial` boolean in `multi_arch_ci.yml` defaults to false for
both dispatch and reusable calls. When true, normal setup/build/test/packaging
jobs are skipped. The experiment runs only when the repository is `ROCm/TheRock`;
the script also refuses execution outside that repository's Actions environment.
Do not run this experiment through rockrel.

Two Linux jobs use the fixed `aws-linux-scale-rocm-prod` label and the same
manylinux image digest as the baseline stage workflow. Each job keeps its own
runner allocation for both trials: one runs **1 then 2**, the other **2 then 1**.
The jobs have independent allocations and a 360-minute limit. They neither
configure AWS upload credentials nor publish release/build artifacts to S3.
GitHub diagnostic artifacts upload with `always()`.

The script records the checked-out diagnostic SHA and requires source baseline
`5b7cb88373362ab80dc0f19605915e534f4158aa` in its ancestry. It bootstraps the
compiler/runtime from successful **TheRock run 41185762902**, explicitly naming
`ROCm/TheRock` as the artifact source repository. The rocm-libraries source stays
at `f4e24d58b3f313f1a31068def1e35376e43c2f49`, with no source patch. The fixed Linux
package version is
`10.3.0.dev0+5e7281668afc1c2c326f7ae0c73043a117ed8b53`.

Stage CMake arguments come from `configure_stage.py` for math-libs,
`gfx103X-all`, and manylinux. Only `hipCCL_tests+configure` and its dependencies
are built during preparation. Tests and benchmarks remain enabled; config
tuning, tuned-types-only mode, and TU splitting must remain off. The benchmark
therefore retains its original 47 key/value pairs and all **gfx1030–gfx1036**
targets. Other math projects and the full hipCCL test/benchmark suite are not built.

## What each trial does

Before expensive work, a strict tiny HIP compile checks the installed compiler's
actual `--offload-jobs=2` capability. A separate `-###` plan must retain the seven
device targets and forward `--wrapper-jobs=2`. The compiler version/hash and
Ninja version are retained. A rendered plan is acceptance/forwarding evidence;
actual overlapping subprocesses must be established from the later trace.

The script modifies **one explicit FLAGS binding on one generated Ninja object
edge**. Every other edge stays unchanged. The original generated manifest and
rules are retained, and the manifest is restored on exit. Only that generated
object and its `.d` file are removed to force each trial. Paths escaping the
nested build or symlinked outputs are rejected.

Dependency preparation uses read-only ccache. Both timed trials set
`CCACHE_DISABLE=1`. Profiling/option-override environment variables
that could suppress parallel device work are removed from compiler children.
Actual timed commands reject `-v`, `-fproc-stat-report`, driver-only/dry-run modes,
and clang-cl mode. These are important controls: Clang's own process-stat option
can disable its device-cc1 parallel scheduler. External `profile_build.py`
collects wall time, waited-child CPU, sampled process-tree CPU/RSS, cgroups,
affinity, waits/status and sampler overhead instead.

The object is built through **native Ninja**. The original benchmark executable
is then linked through its native target. Object mtime, SHA-256 and the native
Ninja log record must stay unchanged during linking; otherwise the experiment
fails instead of timing an accidental second compilation. Generated rules,
expanded commands, plans, native command hashes, logs and profiler data are kept.

Every trial's executable is retained locally outside the diagnostic upload
directory. A byte-by-byte baseline/candidate comparison runs before the job
finishes. Small executables (at most 128 MiB) are also included in the diagnostic
artifact; larger files retain hashes and inspection results in that artifact.
Installed tool help is checked before offload-image inspection. Unsupported
inspection is explicitly **unverified**, not treated as proof of seven images.

## Reading the result

There are two paired comparisons with balanced order, not enough observations
to establish broad run-to-run variance. Keep each same-allocation pair together;
do not mistake different machines for repeated observations on one machine.
Compiler-cache bypass does not flush filesystem/page caches. First versus
second order, thermal state and shared infrastructure can still matter.

Require device-architecture child jobs to overlap, the predicted phase to shrink,
and measured wall-time improvement beyond variability. Check CPU/RSS and cgroup
limits: the limit is per compiler driver/wrapper, so a flag that helps in isolation
can oversubscribe RAM or CPUs during a whole stage. This first trial does not
enable the flag globally or claim end-to-end CI savings.

Identical object/executable bytes are strong equal-output evidence for this
controlled change. Different bytes require investigation of image/symbol/metadata
differences and focused primitive tests plus benchmark registration/runtime checks
on a compatible gfx103X GPU. The build runner does not provide those GPU checks.
The actual primbench registration interface is bare `--dry` with `--json-out`
and `--csv-out`, not an invented `--list` option; even dry registration needs the
HIP runtime/device initialization. No speedup is validated merely by a green job.

## Local validation

Run the focused tests with Ninja available:

```bash
python -m unittest build_tools.tests.profile_offload_trial_test -v
```

These tests exercise exact edge isolation, rejected coverage/profiling overrides,
cache/environment guards, safe forced rebuilds, workflow routing, and actual
Ninja incremental command hashes/reuse using a tiny Python producer. They do not
perform a local ROCm build or benchmark. The Windows host may skip its symlink
creation check; Linux CI runs that guard with normal symlink support.
