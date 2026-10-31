# Controlled joint rocSOLVER experiment

These are reviewable workflow assets, not executed performance or correctness results. The source checkout is frozen at TheRock `5b7cb88373362ab80dc0f19605915e534f4158aa`, with rocm-libraries `f4e24d58b3f313f1a31068def1e35376e43c2f49`. The prerequisite HIP-path repair is shared by every mode. The combined candidate patch is applied alone and recorded separately.

## Cohort and required outputs

The workflow runs three parallel matrix jobs on `aws-linux-scale-rocm-large`, each in the same pinned manylinux image as v3. Runtime preflight requires 192 host-visible and affinity-allowed logical CPUs, AMD EPYC 9R45, and no visible CPU quota. It saves actual runner identity, cgroups and the whitelisted runtime environment. Hidden ancestor limits and interference still require resource-trace review; the label alone does not prove matching allocation.

Each runner builds the same dependency closure through `rocSOLVER+configure`, including the existing SDK fixture and tests. It then performs one complete baseline/candidate pair. Pairs 1 and 3 use baseline→candidate; pair 2 uses candidate→baseline. Three pairs cannot perfectly balance order. The original source/include input manifest is established before any candidate patch, and those same paths are read before every timed trial. This deliberately primes filesystem inputs without proving residency or dropping the OS cache.

The baseline is the original source. The candidate applies `rocsolver-specialized-shards.patch` alone, enables both `ROCSOLVER_SHARD_GETF2_SMALL` and `ROCSOLVER_SPLIT_GETRI_Z`, and preserves sanitizer, optimization/debug flags, `gfx950:xnack+`, clients, targets, nine background projects and four LLVM link jobs. The preset is `linux-release-asan-debug`; compiler cache is disabled throughout. These are cold **project-output** rebuilds with warm dependencies and deliberately primed original inputs, not a full cold-cache workflow or an enabled-ccache comparison.

Before switching source, the previous nested Ninja graph is cleaned so candidate-only helper objects cannot leak into a later baseline. After source selection, the child's cache/configure stamp is cleared and the outer configure target regenerates it. A second full nested clean clears only this project's `.ninja_log`, `.ninja_deps`, build/stage stamps and local stage/dist directories. Dependencies remain installed. The timed command in both modes is the identical outer `cmake --build build --target rocSOLVER+stage -- -k 0 -d keeprsp`, under the same one-second observer. Exact generated commands, configure caches and separate compiler-launcher record directories are retained.

The installed-stage manifest records every file's hash/size and symlink target. Public ELF checks require identical defined dynamic symbol names/types and SONAME, no missing or extra installed paths, and unchanged symlink targets. Required test and benchmark executables must be installed. File bytes are expected to change; hashes establish provenance rather than byte equality. These checks establish neither numerical correctness nor device registration/ODR correctness. No GPU test is dispatched here, and the supported ASan GPU route remains unresolved.

## Phase probes

After the SDK fixture, every job checks the built Clang and LLD capabilities using `--check-only`, before the expensive remaining math prerequisites. This produces no compilation. Both `-ftime-trace` and device-linker trace forwarding are checked with the captured original command. The root-owned `trace_getf2.py` must support `--expected-source`, `--trace`, `--device-link-trace` and `--check-only`.

Only pair 1 performs actual phase probes, after both timed project trials and their output comparison. It reverses the candidate patch, checks the library tree against its original revision and original specialized-source SHA256 manifest, restores the original top/child configuration, and launches four isolated commands concurrently: original getri_z and getf2_c, each with tracing off/on. Each replay redirects object and dependency outputs to its own private directory. Trace-on includes both Clang and device LLD timing. The entire four-command group has a separate one-second resource observer.

There is only one profiling pair per source, and four compilers co-run. CPU/wall differences are provisional profiling-overhead observations with possible shared-resource interference, not a calibrated overhead distribution. Metadata retains exact argv, environment, timing, object hashes/sizes and trace paths. Private object bytes are excluded from the uploaded archive; traces, dependency files, logs and metadata remain. Do not add probe CPU to overlapping parent CPU totals.

`records/getri_z.json` and `records/getf2_c.json` are exact v3 launcher records, including original cwd and argv; neither uses response files. `records/baseline-build-environment.json` preserves the actual v3 root-Ninja build edge. It runs through Teatime and `cmake -E env`, unsetting `ROCM_PATH`, `ROCM_DIR`, `HIP_PATH` and `HIP_DIR`. Both trial builds use that outer target; probes reproduce those four unsets. Actual new outer build edges and parent whitelisted environment are also captured and checked.

## Publish and run procedure

The parent must review these assets before pushing or dispatching. No workflow was triggered while preparing them. On a unique branch whose parent is the frozen HEAD, publish these exact files:

- Replace the existing `.github/workflows/multi_arch_ci_asan.yml` with this directory's `workflow.yml`; that existing workflow identifier is dispatchable on an experiment ref.
- Include this directory's `run_experiment.sh`, `phase_probes.py` and `records/*.json` at their current `experiments/joint/` paths.
- Include the unchanged v3 `build_tools/perf_probe/{command_trace.py,collect_build_perf.py,bootstrap-codeobj.patch}`.
- Include the reviewed root-owned `experiments/compiler/trace_getf2.py` and joint `experiments/compiler/rocsolver-specialized-shards.patch`.

Resolve and record the instrumentation commit separately from frozen source HEAD. Use the established per-command `sa-faizal` environment in `rock_ci_api.py`, then dispatch the existing workflow identifier on the reviewed experiment ref. Save the run ID and actively monitor all three jobs. Each has a 360-minute timeout; matrix fail-fast is enabled. A failure must be diagnosed from its evidence rather than silently dropped from the comparison.

Artifacts are `joint-solver-perf-pair-1-01a19119`, `...-2-...` and `...-3-...`, each containing a corresponding `joint-solver-perf-N.tar.gz`. They contain `perf-*` and build logs, with private replay objects filtered out. Retention is 14 days; download and hash the artifacts promptly. This dedicated workflow only uploads investigation evidence. It contains no release, deployment, S3 or Quartz actions. This does not restrict authorized run-scoped CI artifact storage in a later full-workflow validation.

Report every trial, pairing/order, project wall, aggregate child CPU, trace bottleneck, memory/resource limits and output sizes. Require both leading tails to shrink and paired elapsed improvement to exceed ordinary variation and observer uncertainty. Public ELF checks are only an intermediate gate. Existing getf2/getri GPU tests, device/ODR checks and the applicable complete HEAD CI workflow must pass before accepting an end-to-end optimization. No gains are validated by these unexecuted assets.
