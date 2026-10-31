# Controlled normal CI acceptance

`multi_arch_ci.yml` has a default-off `profile_normal_validation` mode for a
**selected-configuration end-to-end** comparison. This keeps the normal Linux
job graph: all stages, artifact checks, enabled component tests, Python packages
and tests, native DEB/RPM and actual install/uninstall checks. It selects exactly
gfx94x and gfx103x; Windows, PyTorch and JAX are explicitly outside this selection.
It does not validate the historical complete multi-platform matrix.

No optimization is integrated. `profile_normal_case` labels baseline/candidate
evidence only. The preconfigure extension point records an empty product patch
and option list; it changes neither source nor flags. Future product candidates
require a separate review and must preserve source Git provenance where practical.
Apply reviewed dirty patches after normal source fetch rather than advancing a
large source repository's HEAD just to carry diagnostics. Retain actual patch/file
hashes, existing project history and generated version metadata in both arms.

The controller rejects local execution, other repositories, non-dispatch events,
different families/stage selections, stage reuse and narrowed tests. Both arms
must use these inputs:

```json
{
  "profile_normal_validation": "true",
  "profile_normal_case": "baseline",
  "profile_normal_cache_regime": "ci-readonly",
  "linux_amdgpu_families": "gfx94x,gfx103x",
  "windows_amdgpu_families": "",
  "pr_labels": "test_filter:comprehensive",
  "build_stages": "",
  "build_python_packages": "true",
  "build_pytorch": "false",
  "build_jax": "false",
  "build_native_linux": "true",
  "build_native_windows": "false",
  "prebuilt_stages": "",
  "baseline_run_id": "",
  "stage_reuse_mode": "off",
  "changed_projects": ""
}
```

Change only the case label for a control twin; a later reviewed optimization must
declare its own change separately. This documentation does not dispatch a run.

## Common controls

- Product gitlinks remain those at TheRock
  `5b7cb88373362ab80dc0f19605915e534f4158aa`. Diagnostic ancestry is recorded
  separately. Setup checks all 11 source gitlinks before continuing.
- External configuration is pinned to
  `7624a5e209cec569bbed38dad7071fa7393cb52b`, and a failed checkout is fatal.
  Only Linux large/small/medium build pool selections become the historical AWS
  labels. GPU policy, including disabled gfx103x component runners, is retained.
- Version outputs stay fixed to the observed diagnostic SDK version origin
  `82d5e8c176ce0cc00b5fd1cf631ce7790be8992b` and 2026-10-31 date. This preserves
  that version input in both fresh builds; it does not reuse a prebuilt SDK or
  establish binary/compiler fingerprint equality. Actual fingerprints must match.
- Native package build suffix is the fixed diagnostic value `41147801272`.
  Actual fetch/upload prefixes remain the **new current run ID**. No historical
  object is written. Changelog/repository creation timestamps are still stock
  behavior; exact raw-package equivalence needs the separately reviewed metadata
  controls and complete payload inspection, not arbitrary normalization.
- `ci-readonly` retains the existing CI cache backend and namespace, forbids new
  results, and uses an owned initially empty local path. `private-cold` is a
  distinct optional experiment with remote storage disabled and private local
  writes allowed. Neither clears or writes shared caches. Actual configure/build
  children remove inherited `CCACHE_*` overrides except the explicit config path.
  [ccache's documented read-only setting](https://ccache.dev/manual/4.12.1.html)
  prevents result insertion while retaining lookup/statistics behavior.
- Reuse off controls artifact reuse; it does **not** mean cold compiler cache.
  A CI-readonly run is warm or mixed only if actual critical command keys and
  hit/miss outcomes establish that. Other clients can change remote contents.
- Native install images are immutable references resolved from the existing tags;
  tool/package versions, requested image pins, source HEAD/diff/changed-file
  hashes, SDK executable/device-bitcode hashes and visible allocation are captured.
  A requested pin is not a separately measured Docker image identity. Compare
  actual tool/package inventories and allocation facts across arms.
- Linux configure/build and native/Python package creation use the existing
  external profiler. Project Teatime CPU records are opt-in in this mode. WSL's
  normal bridge build remains present, gains internal configure/build profiles
  and phase CPU, and retains Windows-controller source/tool identity separately.
  Sampling is a CPU lower bound; exact waited phase CPU must not be double-counted.
- Every reachable Quartz notifier is disabled in mode. The CI credential action
  requires successful repository/input/destination guard before its existing
  `always()` credential steps. Product-publication actions, release orchestrators
  and downstream release dispatch are absent from the normal workflow closure.
  Per-run CI artifact handoff remains necessary for downstream stages/install tests.

Default behavior outside this mode stays at product workflow behavior; inherited
unconditional profiling/runner overrides from the diagnostic base are now scoped.
Reusable workflows see the original dispatch event. Expressions access its JSON
object explicitly so a child workflow's separate manual-dispatch schema does not
incorrectly narrow the inherited event during actionlint checks.

## Acceptance and remaining gaps

Runtime input receipts distinguish ambient PATH tools from generated CMake
compiler selections, including the CLR-provided Clang used by rocFFT. They hash
separate LLVM, AMD COMGR, CLR/HIPRTC and HSA runtime candidate trees and query
loader providers for generated tools and relevant runtime helpers/libraries.
Unavailable files, unresolved dependencies and loader-query failures remain
explicit. Equality of an LLVM driver alone is not equality of the RTC compiler.

The common profiler's optional `--capture-rtc-inputs` observes only
`rocfft_aot_helper` and `rocfft_rtc_helper`. It records the reviewed nonsecret
RTC/COMGR/append-option allowlist and loaded-library paths/stat identities,
without reading provider contents during the measured build. Observed provider
files are hashed afterward and checked against their sampled stat identities.
Sampler cost is recorded in its own section. This is positive sampled evidence;
short-lived processes may be missed, and Linux process environment snapshots
may omit a parent's post-exec `setenv`. Child RTC exec inherits those overrides.
No whole environment or unrelated process environment is dumped.

Ambient RTC/COMGR knobs and metadata-only cache inventories are separate from
ccache policy. Generated Ninja command overrides and source-defined AOT cache
arguments are recorded separately. The Linux default COMGR location is a
candidate derived from XDG/HOME and the pinned implementation's `/comgr`
suffix; locality can disable it. The observer does not manufacture a private
fresh cache, change pruning, or infer enabled caching from HOME alone. A
pre-stage inventory is not a guaranteed per-helper initial state if other work
can populate the cache. Future I-029 integration must declare and verify an
identical appropriate runtime-cache contract in both arms, or retain this
normal-regime difference from the targeted diagnostic cohort explicitly.

`build_tools/normal_ci_declared_inputs.json` is initially empty. A future
reviewed product hook must declare every patch and copied source input by
workspace-relative path and exact SHA256, with optional `cases`, `stages` and
`families` selectors. Family names compare case-insensitively while retaining
the complete family ID, so a gfx103X-only header is not required in gfx94X math.
This captures new untracked diagnostic headers without sweeping
generated outputs or secrets. Capture retains a mismatched receipt and fails;
unchanged HEAD/tracked diff cannot hide a changed declared header. No product
patch or copied input is registered by the common-control preparation.

Before combining balanced baseline/candidate pairs, require matching source and
SDK fingerprints, generated commands/architectures, cache regime and **actual
critical-command cache outcomes**, requested/actual runner allocation, tool and
package inventories, and enabled test matrices. Keep read-only remote mutation
and weighted GPU test-pool choices visible; matching input JSON alone is not proof
of equivalence. Do not force stage-reuse compatibility to manufacture a fixed SDK.

Require the predicted bottleneck to shrink in the trace, elapsed improvement to
exceed ordinary variation, and every existing selected correctness/artifact check
to pass. Retain artifacts for independent inspection. Requested gfx103x GPU tests
remain disabled by pinned policy; gfx942 install sanity is not gfx103x numerical
validation. Normal CI and historical nightly release use different artifact/cache
policy and framework paths; do not report this selected result as historical global
gain. Compiler/project time, runner consumption and total selected workflow elapsed
are separate outcomes.

Controls and profiles add overhead, captured outside measured build commands where
possible but included in workflow completion. Tool/package downloads and apt
versions remain external variability; no performance or capacity result is implied
by the local source fixtures. Normal acceptance is still unvalidated until reviewed
CI comparisons succeed. No product candidate should be activated merely because
this common-control patch passes its unit/schema checks.
