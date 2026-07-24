# RFC00XX: Feature Flag Lifecycle and Canary Promotion

- **Author:** Brian Harrison (bharriso)
- **Created:** 2026-06-04
- **Modified:** 2026-06-24
- **Status:** Draft
- **Discussion:** TBD (GitHub Discussion link to be added)

> **In a hurry?** Jump to the [Quick reference](#quick-reference) cheat-sheet, or the [Maintainer Playbooks](#maintainer-playbooks) for step-by-step procedures.

## Overview

TheRock and the ROCm libraries it builds need one disciplined way to introduce risky
changes incrementally: gate them behind a flag, exercise both code paths in CI, soak a
candidate default for a cycle, promote it on a predictable cadence, and back it out quickly
when it misbehaves. Today's mechanism — the `FLAGS.cmake` build-flag registry — is real but
partial: no runtime dimension, no metadata or expiry, no team-owned both-state CI, and no
branch-level promotion.

This RFC proposes a complete feature-flag lifecycle built on two flag kinds and a fixed,
automated canary-to-mainline promotion train, modeled on Google Chrome's release channels and
LLVM's release-manager-gated backports. Canary is a soak-and-staging branch: the team flips
the default of the flag (or flags) being promoted to ON, soaks it for one cadence cycle, then
swaps the default on `main`.

The proposal extends the build-flag registry rather than replacing it, and adds a parallel
runtime-flag mechanism — the preferred path, because one binary serves every flag state and
can be reverted in the field with no rebuild. The runtime mechanism is a generic,
library-agnostic TheRock contract: a JSON-location convention, a discovery idiom, and a
precedence order. To ease adoption, TheRock also publishes a small reference header,
`rocm_feature_flags.h`, that implements the contract — but it is not shipped, linked, or
auto-included; there is no `.so` and no package dependency. A library either copies the header
into its own tree (the most portable path, which this RFC recommends) or reimplements the
contract against its existing environment-flag system. So the contract is the standard; the
header is a reference a project may copy. hipDNN, the first adopter, uses its own config reader
rather than the header.

### Glossary

| Term                                          | Meaning                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Flag**                                      | A named boolean toggle gating a change. Two kinds: runtime and build-time.                                                                                                                                                                                                                                                                                                                          |
| **Flag flip**                                 | Changing a flag's effective default (for example, OFF to ON) for a branch or channel.                                                                                                                                                                                                                                                                                                               |
| **Binary-neutral**                            | A change that does not alter emitted artifacts, ABI, or build topology, and can therefore be gated at runtime.                                                                                                                                                                                                                                                                                      |
| **Canary**                                    | The soak-and-staging branch. The team flips the current promotion batch to ON here, so it soaks for one cadence cycle before the default is swapped on `main`. A flag may be OFF on both `main` and canary; canary is not an "everything on" branch, nor the mechanism for both-state CI coverage.                                                                                                  |
| **Soak**                                      | The period a flipped default spends on canary, with builds and tests green, before promotion.                                                                                                                                                                                                                                                                                                       |
| **Promotion**                                 | Swapping a soaked, green flag default from canary onto mainline by means of the automated train.                                                                                                                                                                                                                                                                                                    |
| **Mainline**                                  | `main`, the trunk; the default state users receive from stable and nightly builds.                                                                                                                                                                                                                                                                                                                  |
| **Kill switch**                               | Reverting a flag to OFF in the field without a rebuild (runtime) or by revert and rebuild (build-time).                                                                                                                                                                                                                                                                                             |
| **Both-state CI**                             | Running CI for a single change in both flag states, ON and OFF, so neither code path goes stale. A team-owned mechanism on a team's own PR or branch, triggered by a `flag:<NAME>:both` label or dispatch input, or by a flip branch. Distinct from canary, which soaks only a candidate default. Costs one build and two test runs for a runtime flag; two builds and tests for a build-time flag. |
| **Standalone build**                          | Building a ROCm library on its own, outside a full TheRock assembly, where there is no shipped `share/therock/feature_flags.json` and no TheRock header on the include path. The reader contract degrades silently: the project carries its own copy of the header (or its own reader), and every flag resolves to its compile-time default unless overridden by `ROCM_FEATURE_<NAME>`.             |
| **Reference header (`rocm_feature_flags.h`)** | An example implementation of the reader contract that TheRock publishes for projects to copy into their own tree or reimplement against their existing environment-flag system. Not shipped, linked, or auto-included; there is no `.so` and no package dependency.                                                                                                                                 |

## Goals

1. **Define a flag taxonomy and a firm decision rule:** runtime by default; build-time only for build-structural, ABI, or artifact-altering changes.
1. **Add a generic runtime flag contract** to TheRock: a JSON-location convention, `dladdr`-relative discovery, and a resolution order checked highest-first (`ROCM_FEATURE_<NAME>` environment variable, then installed JSON, then compile-time default), plus an example `rocm_feature_flags.h` header that implements it. Teams copy the header into their own tree or reimplement the contract against their existing environment-flag system; there is no shipped, linked, or auto-included dependency. hipDNN's `EngineOverrideConfig`-style reader is one instantiation, not the contract itself.
1. **Make flags first-class inventory:** owner, created date, expiry, stage, and tracking issue on every flag, surfaced in the configure report and the shipped manifest.
1. **Establish a canary-to-mainline promotion train** (fixed and automated; period to be determined). Canary is a soak-and-staging branch: the team flips the candidate default(s) to ON, soaks for one cadence cycle, then swaps the default on `main`. The minimum soak signal is canary builds and tests green across the cycle; RFC0011's `latest_good.json` remains the `main` and nightly health signal, with an optional per-branch `latest_good@canary.json` as a possible deepening.
1. **Make both-state CI a first-class, team-owned mechanism, decoupled from canary.** A team developing behind a flag can run CI in both states (ON and OFF) for its PR, triggered by a label (for example, `flag:<NAME>:both`), a `workflow_dispatch` input, or a flip branch whose committed config flips the default. Runtime flags cost one build and two test runs; build-time flags cost two builds.
1. **Specify backout, kill-switch, and failed-promotion policy.** The environment kill switch (`ROCM_FEATURE_<NAME>=0`) is the per-process, per-host, minutes-scale revert; editing the installed JSON sets the next-package channel default (which requires a respin). Build-time backout takes one cycle. A bad flip is dropped from the canary batch, not carried forward.
1. **Define flag hygiene:** expiry enforcement, mandatory retirement after default-on, and a per-cycle flag-debt audit (Fowler).
1. **Provide teams a concrete playbook** with clear ownership boundaries between project teams, TheRock, and Quartz.

## Non-Goals

- Replacing the existing `FLAGS.cmake` registry; this RFC extends it.
- Defining the CI data and notification substrate; that is RFC0011 (Quartz), whose signals this RFC consumes.
- Percentage or telemetry-driven rollout (a Finch-style gradual ramp). Flags here are binary on or off; no telemetry pipeline exists in ROCm today to drive a percentage ramp.
- Artifact-level RC-to-final promotion (`build_tools/packaging/promote_*`); that remains as-is, and this RFC adds branch-level flag promotion alongside it.
- Mandating that every existing flag immediately gain a runtime equivalent.

## Background and Prior Art

### TheRock today (EXISTING)

| Mechanism        | File                                       | Behavior                                                                                                                                                                                                                                                                                                                                               |
| ---------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Flag registry    | `FLAGS.cmake`                              | Declares flags via `therock_declare_flag(NAME, DEFAULT_VALUE, DESCRIPTION, …)`.                                                                                                                                                                                                                                                                        |
| Flag utils       | `cmake/therock_flag_utils.cmake`           | `therock_declare_flag` records metadata only; `therock_finalize_flags()` creates `set(THEROCK_FLAG_<NAME> <default> CACHE BOOL …)`; `therock_report_flags()` prints a `Build flags:` header then one `  * NAME = ON (-DTHEROCK_FLAG_NAME=ON)` line per flag; `therock_override_flag_default(name, value)` rewrites the stored DEFAULT before finalize. |
| Branch overrides | `BRANCH_FLAGS.cmake`                       | `include(... OPTIONAL)` sits between the declare block and `therock_finalize_flags()`. It is gitignored on `main` and committed on integration branches to flip defaults; it logs `Flag <name> default overridden …` at configure time.                                                                                                                |
| As-built state   | `build_tools/generate_therock_manifest.py` | `therock_finalize_flags()` writes `flag_settings.json` into the build directory, which is merged into the shipped `share/therock/therock_manifest.json` under `"flags"`. As-built flag state already ships.                                                                                                                                            |
| Fan-out          | generated `project_init.cmake`             | Flags force CACHE variables or `add_compile_definitions` into a subproject only when ON.                                                                                                                                                                                                                                                               |
| Docs             | `docs/development/flags.md`                | (existing flag documentation)                                                                                                                                                                                                                                                                                                                          |

**Precedence (EXISTING):** CLI `-DTHEROCK_FLAG_<NAME>` takes precedence over a `BRANCH_FLAGS` override, which takes precedence over the declared `DEFAULT_VALUE`.

The current registry holds `KPACK_SPLIT_ARTIFACTS` (ON; artifact slicing, build-structural) and
`HIP_KERNEL_PROVIDER_ENABLE` (OFF; DESCRIPTION "Enable hip-kernel-provider plugin", target
`hipkernelprovider`). A legacy
`THEROCK_FLAG_INCLUDE_PROFILER` uses `cmake_dependent_option` outside the registry
(not reported, not in the manifest).

**Gaps this RFC addresses:** no owner, created, expiry, or stage metadata; no expiry or staleness
enforcement; no runtime dimension; two parallel patterns (registry versus `cmake_dependent_option`).
Note that `BRANCH_FLAGS` is gitignored and therefore not observable, which is precisely why this RFC
flips flags by means of observable, committed default changes (`DEFAULT_VALUE` in `FLAGS.cmake` or the
runtime registry default) rather than a branch override.

### CI, branching, and release (EXISTING; promotion hooks here)

| Fact                                                                                                                                                                                                                                                     | Where                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Trunk-based on `main`; submodules auto-rolled by `bump_submodules.yml` (push trigger `branches:[main]`, cron every 12h); `bump_automation.py` hardcodes `base="main"`. Submodule rolls land on `main` only.                                              | repo root, `build_tools/github_actions/bump_automation.py` |
| Long-lived branches: `long_lived_full_match = ["main"]`, `long_lived_prefix_match = ["release/therock-"]`. Long-lived branches receive presubmit and postsubmit on push; others receive presubmit only.                                                  | `build_tools/github_actions/configure_ci.py`               |
| Branch-prefix routing exists for `release/therock-*` branches (push-triggered postsubmit).                                                                                                                                                               | `build_tools/github_actions/configure_ci.py`               |
| Data-driven matrix: families by platforms by build_variants; tiers presubmit within postsubmit within nightly.                                                                                                                                           | `configure_ci.py`, `amdgpu_family_matrix.py`               |
| CI controls: PR labels (`ci:skip`, `ci:run-all-archs`, `test_filter:<…>`, `test:<proj>`), `workflow_dispatch` inputs including `prebuilt_stages` and `baseline_run_id`; the `build_variant` environment value selects release, asan, or tsan.            | `docs/development/ci_behavior_manipulation.md`             |
| `ci_weekly.yml` is a work-in-progress placeholder: a placeholder cron `0 3 * * 0` is commented out, with a no-op `echo "Skipped"` job. This is the reserved automated-promotion slot; the chosen period and schedule are to be determined (see cadence). | `.github/workflows/ci_weekly.yml`                          |
| `ci_nightly.yml` runs daily at `0 02 * * *` via `schedule` (plus manual `workflow_dispatch`). GitHub `schedule` crons run only on the default branch (`main`), so the scheduled nightly cannot run on canary.                                            | `.github/workflows/ci_nightly.yml`                         |
| Release channels are stable, prerelease, nightly, dev, and dev-builds; promotion today is artifact-level (RC to final), not branch to branch.                                                                                                            | `docs/packaging/versioning.md`                             |
| `latest_good.json` is a single global symlink to the most-recent fully-passing nightly (RFC0011 Quartz). It tracks `main` and nightly health; there is no per-branch canary equivalent today.                                                            | RFC0011                                                    |
| `.github/CODEOWNERS` does not currently cover `FLAGS.cmake`, `configure_ci.py`, or `ci_weekly.yml`.                                                                                                                                                      | `.github/CODEOWNERS`                                       |

**Gaps:** nothing toggles a `THEROCK_FLAG_*` from CI today (the controls govern architecture, test,
and variant, not flags); there is no automated branch-to-branch promotion; the promotion slot is
unimplemented; no per-branch canary health signal exists (the nightly is `main`-only); and the flag
registries and promotion workflow are not CODEOWNERS-gated. The gated behavior lives in submodules
that roll on `main` only, so this RFC keeps canary current by frequently rebasing it onto `main` (the
only difference being the flag defaults), so that canary never goes stale relative to the submodule
pointers on `main`.

### External grounding

**Google Chrome (primary).** Channels form a train: Canary (daily, may break), Dev (one to two
times per week), Beta (approximately weekly), Stable (approximately every four weeks). A change
"begins as an experiment in Canary… updated to Dev, then Beta, with more and more testing… until it
makes its way into Stable." Milestone branches are cut from a Canary build and then stabilized; only
safe fixes are backported, and risky work waits for the next train.

- **Runtime flags:** `base::Feature` (`ENABLED` or `DISABLED_BY_DEFAULT`) plus Finch server-side
  field trials for staged rollout (1%, 10%, 50%, 100%) and a kill switch ("dialed back to
  0% until a fix"; "much faster and cheaper to update a server config" than an emergency binary
  respin). `--enable-features` and `--disable-features` provide a local developer override (highest
  precedence). `chrome://flags` entries carry an expiration milestone.

- **Prefer runtime:** "New code should use `base::Feature` instead of switches," for the same
  capability "without requiring binary divergence." Compile-time mechanisms (`buildflag_header`, GN
  args, `is_official_build`) are used only when configurations are "fundamentally incompatible" or
  structural; their costs are binary fragmentation, no gradual rollout, a rebuild required to change,
  and a combinatorial matrix.

- **Validated kill-switch procedure:** first land a code change that disables the flag (proving
  nothing breaks), and then dial the rollout to 0%. The deeper reason to land the disable change
  first is that Finch cannot reach 100% of clients (enterprise and Finch-off populations), so a code
  default change is required for full coverage; the server dial alone never reaches everyone.

  **ROCm has no Finch-equivalent server-side config delivery.** The "dial to 0%" half is therefore
  not realizable fleet-wide in ROCm: it degrades to a per-host environment override or a default-off
  JSON in the next package; there is no remote fleet flip. This coverage gap is exactly why the
  environment and JSON degradation matters: only a code or JSON default change reaches every install.

**LLVM (secondary).** A time-based release branch with a release manager who gatekeeps backports; the
bar tightens as the release nears (RC1 bugfixes, then final critical-only); automated backport via a
`/cherry-pick <hash>` comment that opens an auto-PR with a milestone label. This RFC mirrors RM-gated
promotion, tightening gates, and automated revert and backport tooling.

**Flag hygiene (Fowler, "Feature Toggles").** Flags are inventory with a carrying cost: every flag
needs an owner, purpose, category, and expiry; release toggles are the shortest-lived ("a release
toggle present 2 months post-launch is debt"); toggle points should be minimized (guard entry points
only); a kill switch is retained approximately one cycle after default-on and then the flag and dead
path are removed; stale flags correlate with defects.

**Mapping asserted by this RFC:** a runtime flag is analogous to `base::Feature` and Finch; a
build-time flag is analogous to `buildflag` and GN args. Runtime is preferred because one binary
serves all states, revert is immediate and remote, and there is no combinatorial build matrix.

## Flag Taxonomy (Runtime vs Build-Time)

|                             | **Runtime flag (preferred)**                                    | **Build-time flag**                                                      |
| --------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Analog                      | Chrome `base::Feature` / Finch                                  | Chrome `buildflag` / GN arg                                              |
| Lives in                    | TheRock runtime registry, shipped as `feature_flags.json` (NEW) | `FLAGS.cmake` registry (EXISTING)                                        |
| Set by                      | environment override, installed JSON, or compile default        | `-DTHEROCK_FLAG_*`, `DEFAULT_VALUE` in `FLAGS.cmake`, or compile default |
| Change in field             | Yes: edit JSON or environment, no rebuild                       | No: revert, rebuild, and re-promote                                      |
| Binary impact               | Binary-neutral (one binary, all states)                         | May change artifacts, ABI, or topology                                   |
| Rollout granularity         | Per-process, per-host, per-channel                              | Per-build only                                                           |
| CI cost to test both states | One build, two test runs (toggle environment between runs)      | Two builds (one OFF, one ON)                                             |

### Decision rule

> **Use a runtime flag. Use a build-time flag only if you must answer "yes" to one of the litmus questions below.**

Litmus questions; any "yes" requires build-time:

1. Does the toggle change the set of emitted artifacts or packages (for example, new `-kernels-gfxNNNN` packages or split wheels)?
1. Does it change the build topology or pipeline shape (extra stages, recombine steps, a new build dependency)?
1. Does it change ABI or public headers such that the two states are not link-compatible?
1. Are the two configurations fundamentally incompatible to coexist in one binary (mutually exclusive toolchains or codegen modes)?
1. Does keeping both paths in one binary impose an unacceptable size or security cost that a shipped-disabled feature cannot mitigate?
1. Is the change purely build-system plumbing with no runtime-observable behavior to gate?

If all answers are "no," the change is binary-neutral and must be a runtime flag. The
canonical "yes" case is Multi-Arch Packaging (`KPACK_SPLIT_ARTIFACTS`; see Worked Examples).

### Where flags are declared

All flags start in TheRock as the single source of truth. TheRock owns the mechanism
(registry, manifest, CI, promotion). Project teams consume the flag in their code and own its
behavior and lifecycle. Teams choose the kind, preferring runtime.

#### Adding a flag does not block development

Declaring a new flag — adding a default-OFF entry to `FLAGS.cmake` or `RUNTIME_FLAGS.cmake` with
its metadata — is low-risk and not gated behind heavyweight review. A default-off flag changes
nothing for users; it only adds the ability to protect an in-development feature. Two steps are
intentionally decoupled:

1. **Landing the flag declaration.** This always lands, and lands quickly. It is a small,
   additive, default-off change that immediately lets a team guard a new feature. It is not gated by
   the release-manager or CODEOWNERS process (the CODEOWNERS gate below applies only to default flips
   and promotion), and it waits on nothing. New-flag PRs are treated as priority reviews, so review
   never blocks a team from starting work behind a flag.
1. **Enabling CI for the flag.** Wiring the flag into both-state CI — the two-build path (build-time)
   or the one-build, two-test-run path (runtime) — is a follow-on step that blocks neither the
   declaration nor the feature. A team can land the flag and guard its code first, then add CI.

So starting a new feature depends on just two things: the flag declaration landing in TheRock, and
the TheRock submodule bump reaching the consuming repository (for example, rocm-libraries; see the
"Starting a new feature" playbook). Neither is a chokepoint: the declaration is reviewed promptly,
and the default-off flag protects the feature the moment the bump lands.

```
TheRock registry (build-time FLAGS.cmake  OR  runtime registry)
        │  declares NAME + metadata + default
        ▼
ships state  ──► therock_manifest.json["flags"]  (build-time, EXISTING)
             └─► feature_flags.json              (runtime, NEW)
        │
        ▼
Project code consumes flag at a guarded entry point and owns the behavior.
```

## Runtime Feature Flags

This is the preferred kind; the entire section is new work. It is a generic, library-agnostic
TheRock contract, so any ROCm library (rocBLAS, MIOpen, rocm-systems, hipDNN, and others)
declares and consumes runtime flags the same way. hipDNN appears afterward as one instantiation,
not the contract itself.

### The generic contract (NEW; TheRock-wide)

**(a) Location contract.** TheRock ships a shared, cross-library default state at
`share/therock/feature_flags.json`, the natural home alongside the existing
`share/therock/therock_manifest.json`. Optional per-component override files
(for example, `share/<component>/feature_flags.json`) layer on top. Files are discovered relative to
the consuming module via the `dladdr` idiom (so they resolve regardless of install prefix), not from
a hardcoded path.

**(b) Documented reader contract and example header.** The contract is the standard: the JSON
location above, `dladdr`-relative discovery, and the precedence order below. TheRock additionally
publishes an example reference header, `rocm_feature_flags.h`, that implements the contract so that
adoption is straightforward. The header is not a shipped, linked, or auto-included component; there
is no `.so` to link and no package dependency. A library either:

- copies the example `rocm_feature_flags.h` into its own source tree (recommended, and the most
  portable option; the project then owns its copy and builds it standalone with no TheRock include
  path), or
- reimplements the contract against its own existing environment-flag system (for example, hipDNN's
  `EngineOverrideConfig` and `validateBeforeAdding` machinery, which does not take the header).

A conforming consumer (the header or a reimplementation) locates the JSON via `dladdr`, parses it
(for example, with `nlohmann::json`), silently falls back to the compile-time default on a missing or
unparseable input (including standalone builds where no `feature_flags.json` exists), and applies the
environment precedence below. hipDNN's `EngineOverrideConfig`-style reader is one implementation and
example of this contract (see Worked Examples), not the generic mechanism.

**Standalone builds.** A consequence of the fallback above: a project that vendors its own copy of
`rocm_feature_flags.h` (see the Glossary) needs no TheRock assembly to build, test, or override a
flag. With no installed `feature_flags.json`, every flag resolves to its compile-time default and
remains overridable via `ROCM_FEATURE_<NAME>`.

**(c) Environment-prefix rule.** The cross-library namespace is `ROCM_FEATURE_<NAME>` (reusing the
`ROCM_`-reserved ROCm-wide namespace); this is the generic kill switch any library honors.
Per-library aliases such as `HIPDNN_FEATURE_<NAME>` are optional and resolve to the same flag.

### Declaration (NEW)

A parallel TheRock-side registry declares runtime flags and generates the shipped default state.
The proposed surface mirrors `therock_declare_flag`:

```cmake
# RUNTIME_FLAGS.cmake (NEW), included from the top-level build like FLAGS.cmake
therock_declare_runtime_flag(
  NAME           SDPA_PAGED_KERNEL_V2
  DEFAULT_VALUE  OFF
  OWNER          attention-team
  CREATED        2026-06-04
  EXPIRES        2026-09-01           # review-by date (hygiene)
  STAGE          canary               # in-development|canary|default-on|deprecated|long-lived
  ISSUE          ALMIOPEN-2002        # required for non-mainline stages
  DESCRIPTION    "Enable v2 paged-attention SDPA kernel variant"
)
```

`therock_finalize_runtime_flags()` (NEW) emits the shared `share/therock/feature_flags.json` into the
install tree, and a `runtime_flags` block into `therock_manifest.json` alongside the existing `flags`
block.

```json
{ "SDPA_PAGED_KERNEL_V2": false, "MY_NEW_BACKEND": false }
```

### Install location: shipped globally, once (no per-library toml wiring)

`feature_flags.json` is a single global file that TheRock ships once, at
`share/therock/feature_flags.json`, directly alongside the existing
`share/therock/therock_manifest.json`, and generated and installed by the same `base/aux-overlay`
step (`base/aux-overlay/CMakeLists.txt`). This requires no per-library `.toml` changes and no new
`include` anywhere: the aux-overlay artifact already claims everything it installs via a catch-all
glob:

```toml
# base/artifact.toml  (EXISTING)
[components.lib."base/aux-overlay/stage"]
default_patterns = false
include = [ "**/*" ]
```

A file installed to `share/therock/` from aux-overlay (exactly how the manifest ships today) is
therefore packaged automatically. `therock_finalize_runtime_flags()` writes `feature_flags.json` into
that step, and it ships without additional configuration.

> Note: the catch-all is specific to aux-overlay. Every other component sorts files by glob, and the
> default `lib` globs are shared-library-only (`**/*.so`, `**/*.dll`, and similar;
> `build_tools/_therock_utils/artifact_builder.py`, `docs/development/artifacts.md`), so a `.json` is
> not auto-claimed there. That matters only for the optional, opt-in case where a library ships its
> own per-component override file under its own tree (for example, `lib/<lib>/feature_flags.json`):
> that library then adds one `include` line to its component toml. The global file needs none.

### Consumption (library-agnostic)

A library reads a flag at a single guarded entry point, using either a vendored copy of the example
`rocm_feature_flags.h` or its own small reader implementing the contract. The reader:

- **locates the JSON relative to the calling module:** it resolves the directory of the module's own
  shared object (the `dladdr` idiom on Linux, or the equivalent on Windows), then a known relative
  path to `share/therock/feature_flags.json`. This is install-prefix-independent, with no hardcoded
  `/opt/rocm`.
- **resolves each flag by the order below,** returning the first source that defines it and otherwise
  the compile-time default.
- **treats a missing or unparseable file as not present:** a silent fallback to the default, never an
  error.
- **re-reads the environment on each query** (no internal static cache), so that an operator's
  `ROCM_FEATURE_*` change takes effect on the next process start without a rebuild; a caller may
  cache the resolved value if it chooses.
- **reads the environment through a small `getEnv`-style wrapper,** never raw `getenv`.

hipDNN is one such consumer (see Worked Examples): its existing config reader and plugin machinery
implement this contract; they are not the contract.

### Resolution order (checked top to bottom; the first source that defines the flag wins)

| Precedence     | Source                                                                                                                                        | Role                                                                      |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **1, highest** | `ROCM_FEATURE_<NAME>=0\|1` environment variable (per-library aliases `HIPDNN_FEATURE_<NAME>=0\|1`, list form `HIPDNN_ENABLED_FEATURES=a,b,c`) | Per-flag operator or developer override; the kill switch.                 |
| 2              | `ROCM_FEATURE_FLAGS_PATH` environment variable pointing to an alternate JSON file (otherwise falls through to the installed file)             | Redirect to a different flags file (parallels `HIPDNN_HEUR_CONFIG_PATH`). |
| 3              | Installed `share/therock/feature_flags.json` (plus optional per-component overrides)                                                          | Vendor, distro, or canary-channel state; the no-rebuild channel default.  |
| **4, lowest**  | Built-in compile-time default                                                                                                                 | Final fallback when the flag is defined nowhere above.                    |

`ROCM_FEATURE_*` is the cross-library generic kill switch (in the `ROCM_`-reserved namespace);
`HIPDNN_FEATURE_*` is an optional per-library alias resolving to the same flag. Editing the installed
JSON sets the channel-level default for the next package; the environment override is the live
operator kill switch and developer override.

### Inspecting active flags (discoverability)

Users and testers can discover the active flag surface two ways: through the installed JSON paths
(`cat <prefix>/share/therock/feature_flags.json`, plus any per-component override files), and through
the manifest `runtime_flags` block in `<prefix>/share/therock/therock_manifest.json`. This RFC
proposes a helper, `rocm-feature-flags --list`, that prints each flag's resolved effective value and
its source (compile default, installed JSON, redirected JSON, or environment), so that an operator can confirm what is actually
active before and after setting `ROCM_FEATURE_<NAME>`.

**Authority:** the installed `feature_flags.json` is the runtime source of truth; the manifest
`runtime_flags` block is the as-built record (what the package shipped with). They can drift if a
distro patch edits one without the other, so tooling should treat the JSON as authoritative and
report any mismatch with the manifest.

**Security and tamper considerations.** The installed JSON lives in the install prefix, which is
root-owned in system installs; the environment override is per-process and unprivileged by design,
which is precisely the point of a developer and operator kill switch. There is a multi-tenant
consideration: a writable install tree (or a permissive `ROCM_FEATURE_*` in a shared environment)
lets an unprivileged user enable an experimental code path. Operators in multi-tenant settings should
keep the install prefix root-owned and treat `ROCM_FEATURE_*` as an explicitly trusted developer and
operator surface.

## Build-Time Feature Flags

This is the existing `FLAGS.cmake` system, extended with metadata. It is used only when a litmus
question requires it.

### Metadata extension (NEW)

Extend `therock_declare_flag` with `OWNER`, `CREATED`, `EXPIRES`, and `STAGE`, keeping `ISSUE`
(required for non-mainline stages) and `DESCRIPTION`:

```cmake
therock_declare_flag(
  NAME          KPACK_SPLIT_ARTIFACTS
  DEFAULT_VALUE ON
  OWNER         packaging-team
  CREATED       2025-11-20
  EXPIRES       2027-01-01
  STAGE         default-on
  DESCRIPTION   "Split target-specific artifacts into generic and arch-specific components"
)
```

`therock_report_flags()` is extended to print owner, stage, and expiry; the manifest `flags` block is
extended to carry the same metadata. `therock_finalize_flags()` (new behavior) warns when a flag is
past `EXPIRES` and errors on `main` when a `STAGE` other than `default-on`, `deprecated`, or
`long-lived` ships without an `ISSUE`.

### Fold legacy flags into the registry (NEW cleanup)

`THEROCK_FLAG_INCLUDE_PROFILER` (currently `cmake_dependent_option`, not reported, not in the
manifest) and any similar out-of-registry options are migrated into `FLAGS.cmake` so that there is one
pattern. This is tracked as a phase-1 task.

### Setting build-time flag state

| Mechanism                        | Reviewable?                     | Use                                                                          |
| -------------------------------- | ------------------------------- | ---------------------------------------------------------------------------- |
| `-DTHEROCK_FLAG_<NAME>` (CLI)    | Not applicable (per-invocation) | Developer or one-off CI build.                                               |
| `DEFAULT_VALUE` in `FLAGS.cmake` | Yes (reviewed diff)             | The reviewed default; changed on `canary` to soak, and on `main` to promote. |

## Flag Lifecycle and Hygiene

Every flag is inventory with a carrying cost (Fowler). Mandatory metadata: NAME, OWNER, CREATED,
EXPIRES, STAGE, ISSUE (non-mainline), DESCRIPTION.

### Stages

| Stage            | Default                                     | Branch         | Notes                                                                                                                                      |
| ---------------- | ------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `in-development` | OFF                                         | feature branch | Immature; not yet promoting.                                                                                                               |
| `canary`         | ON on canary (this batch only), OFF on main | canary         | Soaking one cycle before the default is swapped on main. Only flags in the current promotion batch are ON on canary.                       |
| `default-on`     | ON                                          | main           | Promoted; the new default.                                                                                                                 |
| `deprecated`     | ON, removal scheduled                       | main           | The post-`default-on` kill-switch retention window (approximately one cycle, per Hygiene rule 3); then the flag and dead path are removed. |
| `long-lived`     | ON (or as configured)                       | main           | The justified exception: a permanent operational kill switch or permissioning toggle, exempt from expiry (see Hygiene rule 6).             |

### Hygiene rules (adopting Fowler and Chrome)

1. **Owner and expiry are mandatory.** No flag merges without both. `EXPIRES` is a review-by date.
1. **Minimize toggle points:** guard entry points only (for example, one `validateBeforeAdding` check), and never sprinkle conditionals.
1. **The kill switch is retained approximately one cycle** after `default-on`, then the flag and dead code path are retired via a normal PR; dependents are already removed, so it is a pure dead-code deletion (team-owned).
1. **Test both states while live** (team-owned both-state CI; see CI Integration), and collapse to one when removed.
1. **A per-cycle flag-debt audit** runs at each promotion cycle (the automated promotion job; see cadence): it lists flags past `EXPIRES`, flags `default-on` for more than one cycle and still present, and stale flags, and files removal issues.
1. **Long-lived flags are the justified exception:** a permanent operational kill switch or permissioning toggle may live indefinitely but must be explicitly marked `STAGE: long-lived` and exempt from expiry.

## Branching and Canary-to-Mainline Promotion

### Branch model

| Branch      | Role                                                                                                                                               | CI                                                                                                     |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `feature/*` | Immature flags (`in-development`), OFF everywhere.                                                                                                 | Presubmit only (EXISTING).                                                                             |
| `canary`    | Soak the to-be-promoted flag default or defaults for one cycle. Only the current promotion batch is flipped ON here; everything else matches main. | CI configured to run on `canary` (add to `long_lived_full_match` in `configure_ci.py` for postsubmit). |
| `main`      | Mainline and trunk; the promotion target.                                                                                                          | Presubmit and postsubmit (EXISTING).                                                                   |

**Canary branch.** Canary is a plain `canary` branch. CI is configured to run on it by adding
`canary` to `long_lived_full_match` in `configure_ci.py`, giving presubmit and postsubmit coverage.
Canary is soak-only: it is not an "everything on" branch, and it is not where a team obtains
both-state CI coverage (that is the separate, team-owned mechanism in CI Integration).

### How a flip lands on canary

A flip is a reviewed PR into the `canary` branch that changes the flag's default value and
bumps `STAGE` to `canary`; a normal, observable, reviewable diff:

- **Runtime flag:** the PR changes the flag's default in the runtime registry or shipped
  `feature_flags.json` (flag to true) and bumps `STAGE` to `canary`. This is a fully observable diff.
- **Build-time flag:** the PR changes the flag's `DEFAULT_VALUE` in `FLAGS.cmake` (OFF to ON) and
  bumps `STAGE` to `canary`. This is a fully observable diff.

Because a flip is a committed default change, it is fully reviewable on both `canary` and `main`;
there is no hidden or gitignored override.

### Canary currency (frequent rebase onto main)

`canary` is automatically rebased onto `main` on a frequent schedule; the only difference between
`canary` and `main` is the flag-default flips currently soaking. This keeps canary current by
construction and removes any cross-repo split-brain:

- **No staleness or split-brain.** Because canary is always `main` plus the soaking flag-default
  diffs, it always carries the current submodule pointers from `main`. There is no separate
  SHA-assertion machinery to maintain; the guarantee is automatic because canary equals main plus
  defaults.
- **The ordering invariant is preserved.** The gated code lands on `main` first (via the normal
  submodule bump) and is therefore already present on canary after the next rebase, before its
  default is flipped to soak. Gated code is always on `main` first.
- **Promotion does not affect currency.** Promotion lands the soaked default change on `main`; after
  the next rebase that flip is in `main` and is no longer a canary diff. No manual reset is needed.

### One reviewed batch per cycle

A single shared canary branch yields one green-or-red soak signal for the whole branch; it cannot
attribute a failure to a specific flip or certify the others independently. The default is therefore
one small, explicitly reviewed batch per soak cycle (matching the Chrome framing of "begins as an
experiment"). Per-flip independent certification (per-flip sub-branches each independently signalled,
or Quartz green-per-flag attribution) is new machinery and is not assumed. Until it exists, the claim
that independent flips can proceed past a bad one is not made.

### Promotion cadence (fixed and automated; period to be determined)

Promotion runs on a fixed, automated schedule; this is the only settled cadence decision. The period
(weekly, bi-weekly, or monthly) is open for reviewers, as is the exact slot. The job lives in the
reserved `ci_weekly.yml` slot (which today carries a commented-out placeholder cron `0 3 * * 0`; the
chosen period is to be determined). On each scheduled fire, the job opens the promotion PR; a release
manager (not the flip author) reviews and merges it; the frequent rebase then keeps canary current,
and the next batch of flip PRs is merged in to soak, with no manual reset.

| Phase                                   | Action                                                                                                                                                                                                                                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Scheduled fire: promotion PR opened** | If the soak signal was green across the whole soak cycle (minimum: canary builds and tests green) and the gates pass, the job opens the promotion PR landing the soaked default change or changes (`FLAGS.cmake` `DEFAULT_VALUE` or runtime registry) on `main`, and emits the flag-debt audit.  |
| **RM merge**                            | A release manager (not the flip author) reviews and merges the promotion PR to `main`. After the next frequent rebase, the promoted flip is in `main` and is no longer a canary diff, so no manual reset is required. Automation merges the next approved batch of flip PRs into canary to soak. |
| **Soak cycle**                          | Canary soaks the current batch; each push runs CI, accumulating the soak-cycle signal. The `main`-only nightly continues to feed `latest_good.json` for `main` health.                                                                                                                           |

**No manual canary reset.** Promotion merges the soaked default onto `main`; the frequent rebase of
`canary` onto `main` then keeps canary current automatically (the promoted flip is no longer a
divergence once it is in `main`), and automation merges the next approved batch of flip PRs into
`canary` after the cycle.

Train discipline (Chrome): a flip that misses the cut waits for the next cycle; risky work is
never rushed onto the promoting line.

The job that automates this train — its soak-signal gate, the release-manager-merged promotion PR,
and the flag-debt audit — is specified in CI Integration → Promotion job (implemented in the reserved
slot).

## CI Integration: Team-Owned Both-State Testing

**Principle.** A team developing behind a flag must be able to run CI in both states, flag ON and
flag OFF, so that neither path becomes stale. This is a first-class, team-owned mechanism, independent
of canary: canary soaks a to-be-promoted default, whereas both-state CI is how a team validates its
own change under both settings, on its own PR or branch, whenever it chooses. Nothing toggles a flag
from CI today; wiring the trigger through `configure_ci.py` is new work.

### Trigger: a label, or a designated flip branch

A team requests both-state coverage two ways (either, or both):

- **PR label or dispatch input.** A new PR label (for example, `flag:<NAME>:both`) or a
  `workflow_dispatch` input tells CI to exercise that flag in both states for the PR.
- **Designated flip branch.** A branch whose committed default change flips the flag:
  `DEFAULT_VALUE` in `FLAGS.cmake` (build-time) or the runtime registry or branch `feature_flags.json`
  (runtime), so that the branch's CI exercises the flipped state directly. This is the same observable
  committed-default mechanism canary uses, here serving the team's own validation rather than
  promotion.

### Cost model by kind

- **Runtime flag: one build, two test runs.** Build once; run the affected tests twice, toggling
  `ROCM_FEATURE_<NAME>=0` then `=1` (or via the JSON). The flag is binary-neutral, so no rebuild is
  needed, and this can be a single CI job that invokes the test target twice. This is the primary
  efficiency reason to prefer runtime flags.
- **Build-time flag: two builds.** Build once flag-OFF and once flag-ON, and test each. This is
  expensive; amortize it with `prebuilt_stages` and `baseline_run_id` (existing controls) and scope
  it to the flag-sensitive tests or one architecture (via the label) to bound the cost.

### Implementation hooks

- **Build-time:** add a build dimension mirroring the existing `build_variant` (release, asan, tsan)
  pattern in `amdgpu_family_matrix.py` and `configure_ci.py`, gated by the label so that it is opt-in
  per PR and never the default across all families.
- **Runtime:** a test-invocation parameter that runs the test target twice with `ROCM_FEATURE_<NAME>`
  set to each value. No extra build is required.
- **Wiring:** route the label or `workflow_dispatch` input through `configure_ci.py` (new work;
  nothing toggles a flag from CI today). Build-time selects the second build dimension; runtime
  selects the second test invocation.

### Promotion job (implemented in the reserved slot)

Implement the automated promotion job in `ci_weekly.yml` (replacing the no-op; cron and period to be
determined per cadence) so that, on each scheduled fire, it:

1. Reads the canary soak signal for the soak cycle (minimum: canary builds and tests green across the
   cycle; optionally the new per-branch `latest_good@canary.json` if built). It does not read the
   `main`-only nightly `latest_good.json`.
1. Checks the gates (below). The flag's gated code is already on `main` (guaranteed by the rebase
   model: canary equals main plus defaults). If any gate fails, it does not promote, notifies owners,
   and exits.
1. Opens the promotion PR (the mainline-default diff). It is not auto-merged: a release manager other
   than the flip author merges it (CODEOWNERS-gated; see below).
1. Emits the flag-debt audit report.

**RBAC and CODEOWNERS (new; not present today).** `.github/CODEOWNERS` does not currently cover
`FLAGS.cmake`, `configure_ci.py`, or `ci_weekly.yml`. P1 adds CODEOWNERS entries naming a
release-manager group (for example, `@ROCm/TheRock-release-managers`) for the flag registries
(`FLAGS.cmake`, `RUNTIME_FLAGS.cmake`), the promotion workflow (`.github/workflows/ci_weekly.yml`),
and `configure_ci.py`. The promotion PR must be approved by someone other than the flip author.

**Scope of the gate: flips, not new flags.** The CODEOWNERS gate exists to protect default flips on
`main` (promotion) and the CI and promotion machinery, which are the high-blast-radius changes. It is
not intended to make adding a new default-off flag a chokepoint (see "Adding a flag does not block
development"). Where CODEOWNERS coverage of `FLAGS.cmake` or `RUNTIME_FLAGS.cmake` would otherwise
force release-manager review on every new-flag declaration, such additive default-off PRs are treated
as priority approvals so that the protective flag can land promptly; the full promotion bar is
reserved for PRs that change a `DEFAULT_VALUE` or registry default on `main`.

### Promotion gates

- The canary soak signal is green for the entire soak cycle (minimum: canary builds and tests green;
  optionally `latest_good@canary.json`).
- The flag's gated code is already on `main` (guaranteed by the rebase model: canary equals main plus
  defaults).
- There are no open regressions tied to the soaked flip or flips.
- Each promoting flag has a non-expired `OWNER` and `EXPIRES`, and (if a non-mainline stage) an
  `ISSUE`.
- The flag was exercised in both states (team-owned both-state CI: label or flip branch) and is green
  in both.
- The promotion PR is approved by a release manager other than the flip author (CODEOWNERS).

## Backout and Error Handling

| Case                                          | Action                                                                                                                                                                                                                                                                                                | Cost                                                                        |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Runtime flag bad: environment kill switch** | `ROCM_FEATURE_<NAME>=0` (alias `HIPDNN_FEATURE_<NAME>=0`): a true per-process and per-host live control, with no package change. This is the minutes-scale revert, but it is per-host (ROCm has no fleet config push, so it does not reach the fleet remotely).                                       | Minutes, per host. Primary argument for preferring runtime.                 |
| **Runtime flag bad: channel default**         | Edit the installed `feature_flags.json` to default the flag OFF. This is the channel default for the next package only; already-installed users need a package respin and reinstall. Per Chrome's procedure, land the disable change first (proving nothing breaks), then change the shipped default. | Not minutes; a package respin and reinstall for existing installs.          |
| **Build-time flag bad**                       | Revert the flip, rebuild, and re-promote next cycle.                                                                                                                                                                                                                                                  | One cycle plus a full rebuild. Argument against overusing build-time flags. |
| **Failed promotion (post-merge regression)**  | LLVM RM-gated revert of the offending mainline-default diff on `main`; re-promote next cycle once fixed.                                                                                                                                                                                              | One cycle; isolates blame to the promoted flip.                             |
| **Bad flip caught on canary**                 | Drop the offending default-flip commit from `canary` (or exclude it from the next batch). The frequent-rebase and batch model means the bad flip is simply not carried forward; no surgical in-place revert is needed.                                                                                | Bounded; the bad flip simply misses the cut.                                |
| **Canary red mid-cycle**                      | Promotion does not happen; the flip or flips soak another cycle. There is no mainline impact (the mainline default is OFF).                                                                                                                                                                           | Zero mainline impact.                                                       |

Note the fleet-coverage gap: there is no Finch equivalent in ROCm, so the environment kill switch is
per-host and the JSON default reaches only the next package. A full-coverage revert requires the
shipped default change, which is exactly why landing the disable change first matters.

The release-manager role (per LLVM) comprises the promotion job and the human owner who approves the
automated promotion PR (someone other than the flip author); the bar tightens as flips approach
`default-on`.

## Team Workflow

Ownership: the team owns the flag, the gated behavior, and retirement; TheRock owns the registry,
manifest, CI mechanism, and promotion train; Quartz (RFC0011) owns the green signal.

1. **Create the flag in TheRock.** Choose the kind via the decision rule (prefer runtime); declare it in the runtime registry (`therock_declare_runtime_flag`) or `FLAGS.cmake` with full metadata (owner, created, expires, stage `in-development`, issue).
1. **Guard the code in your project** at one entry point (runtime: a single check via the documented reader contract, that is, a vendored copy of the example `rocm_feature_flags.h` header or your own small consumer, for example hipDNN's `validateBeforeAdding`; build-time: a `#define` or CMake-gated branch).
1. **Land behind the flag, default-off, on trunk** (and land the gated submodule code OFF on `main` first). Mainline stays OFF; nothing changes for users yet.
1. **Validate both states in CI** (team-owned both-state CI): apply the `flag:<NAME>:both` label or dispatch input, or develop on a flip branch. Runtime takes one build and two test runs; build-time takes two builds.
1. **Open a reviewed PR into `canary` changing the flag's default** (`STAGE` to `canary`); a fully observable diff (the runtime registry or `feature_flags.json` default for runtime; `DEFAULT_VALUE` in `FLAGS.cmake` for build-time). The default is one reviewed batch per cycle. The flip soaks one cadence cycle, accumulating the canary soak signal.
1. **Automated promotion** lands the soaked default change on `main` (a reviewed `FLAGS.cmake` `DEFAULT_VALUE` or runtime registry diff), once the soak signal is green and the gated code is already on `main` (guaranteed by the rebase model); it is merged by a release manager other than the flip author. Stage becomes `default-on`.
1. **After approximately one cycle,** retire the flag in dependency order: first drop both-state CI for it; then remove the now-dead gated code from the owning library's trunk (for example, a PR against rocm-libraries `develop`), deleting the OFF path and keeping the former ON path; then remove the flag declaration from TheRock `main` (`FLAGS.cmake` or the runtime registry). Removing the consumer before the declaration is uniformly safe for both flag kinds: once the gated code is gone nothing references the flag, so the declaration removal can neither revert the live build nor break a consumer. The dead-code step is a pure deletion. Permanent kill switches (`STAGE: long-lived`) are the marked exception.

## Maintainer Playbooks

These goal-oriented checklists describe what to do to accomplish a given task. Each is self-contained;
you should not need to read the rest of this RFC to follow one. Conventions: `<NAME>` is your flag,
`ROCM_FEATURE_<NAME>` is its environment override, and the flag is default-off until promotion.

### Playbook A: starting a new feature behind a flag

**Goal:** protect new, in-development code so that it ships disabled and changes nothing for users.

1. **Pick the kind** with the decision rule (Flag Taxonomy). Default to runtime; pick build-time only
   if a litmus question is "yes" (an artifact, topology, or ABI change).
1. **Declare the flag in TheRock, default OFF,** with metadata: `OWNER`, `CREATED`, `EXPIRES`,
   `STAGE: in-development`, `ISSUE`:
   - runtime: `therock_declare_runtime_flag(NAME <NAME> DEFAULT_VALUE OFF …)` in `RUNTIME_FLAGS.cmake`
   - build-time: `therock_declare_flag(NAME <NAME> DEFAULT_VALUE OFF …)` in `FLAGS.cmake`
   - This PR can land immediately; it is a priority review and is not release-manager-gated (see
     "Adding a flag does not block development").
1. **Wait for the TheRock submodule bump** to carry the new flag into your consuming repository (for
   example, rocm-libraries). The default-off flag now exists for you to guard against. (These two
   steps, the declaration landing and the bump, are the only prerequisites to starting, and neither
   is a review chokepoint.)
1. **Guard your code at exactly one entry point.** Runtime: a single check via a vendored copy of the
   example `rocm_feature_flags.h` (or your own reader or existing environment-flag system), for
   example hipDNN's `validateBeforeAdding`. Build-time: one `#define` or CMake-gated branch. Do not
   sprinkle conditionals.
1. **Land your feature default-OFF on trunk.** Mainline stays OFF; nothing changes for users. You can
   now iterate behind the flag.

**Completion criteria:** the flag exists in TheRock (default OFF), the bump has reached your
repository, your new code is guarded at one entry point, and trunk is green with the feature OFF.

### Playbook B: testing a new feature in both states

**Goal:** prove both the ON and OFF paths work, so that neither becomes stale and promotion later is
safe.

1. **Choose a trigger** (either or both):
   - **Label or dispatch:** add the `flag:<NAME>:both` PR label (or set the `workflow_dispatch`
     input).
   - **Flip branch:** develop on a branch whose committed default flips `<NAME>` to ON.
1. **Run both states:**
   - **Runtime flag: one build, two test runs.** Build once; run the affected tests twice, toggling
     `ROCM_FEATURE_<NAME>=0` then `ROCM_FEATURE_<NAME>=1` (or via the JSON). No rebuild between runs.
   - **Build-time flag: two builds.** Build OFF and test, then build ON and test. Amortize with
     `prebuilt_stages` and `baseline_run_id`; scope to the flag-sensitive tests or one architecture
     via the label.
1. **Confirm what is active** with `rocm-feature-flags --list` (resolved value and source) before each
   run.
1. **Locally or standalone:** you do not need a TheRock assembly. With no `feature_flags.json`, the
   flag sits at its compile-time default; flip it with `ROCM_FEATURE_<NAME>=1` for an ad-hoc ON run.

**Completion criteria:** the affected tests are green in both the ON and OFF states, on CI,
attributable to this PR.

### Playbook C: promoting a feature default to `main`

**Goal:** make ON the new default, safely, on the predictable train.

1. **Pre-check the gates** (all must hold): a non-expired `OWNER` and `EXPIRES` (plus `ISSUE` if a
   non-mainline stage); the gated code is already on `main` (guaranteed by the canary-rebase model);
   the flag was green in both states (Playbook B); there are no open regressions tied to it.
1. **Open a reviewed PR into `canary`** that flips the default to ON and bumps `STAGE: canary`:
   - runtime: flip the default in the runtime registry or `feature_flags.json` (to `true`)
   - build-time: change `DEFAULT_VALUE` from OFF to ON in `FLAGS.cmake`
   - This is a fully observable, reviewable diff. The default cap is one small reviewed batch per
     cycle.
1. **Soak one cycle on `canary`.** Each push runs CI; the soak signal accumulates (minimum: canary
   builds and tests green for the whole cycle). If canary goes red, your flip simply soaks another
   cycle, with zero mainline impact.
1. **Automated promotion opens the mainline PR** on the scheduled fire (the `ci_weekly.yml` slot) once
   the soak is green. A release manager other than you reviews and merges it. `STAGE` becomes
   `default-on`. You do not manually reset canary; the frequent rebase removes the promoted flip as a
   divergence.
1. **If a regression appears post-merge:** the release manager reverts the mainline-default diff; you
   re-promote next cycle once it is fixed. Field-level mitigation in the meantime is
   `ROCM_FEATURE_<NAME>=0` (per-host, minutes).

**Completion criteria:** the default-ON diff is merged on `main` by a release manager, `STAGE` is
`default-on`, and the post-merge nightly is green.

### Playbook D: retiring a flag (after it has been default-on for approximately one cycle)

**Goal:** pay down the flag debt by removing the toggle and the now-dead OFF path.

The steps are listed in the order to perform them: remove things in dependency order, the consumer
before the declaration it depends on. This sequence is uniformly safe for runtime and build-time
flags alike — once the gated code is gone, nothing references the flag, so removing the declaration
can neither revert the live build nor break a consumer.

1. **Confirm it has been `default-on` for approximately one cycle** and that nothing still depends on
   the OFF behavior. Keep a permanent kill switch only if it is explicitly `STAGE: long-lived` (the
   marked exception).
1. **Collapse both-state CI to one** for that flag (drop the label or flip-branch handling). Do this
   first: it depends on nothing and stops CI from exercising a flag that is about to disappear.
1. **Remove the dead gated code** from the owning library's trunk (for example, a PR against
   rocm-libraries `develop`): delete the OFF branch, drop the flag check, and keep only the former ON
   path unconditionally. After this, nothing references the flag. This is a pure dead-code deletion.
1. **Remove the flag declaration** from TheRock `main` (`FLAGS.cmake` or `RUNTIME_FLAGS.cmake`). Safe
   for both flag kinds, because no consumer references the flag any longer.

**Completion criteria:** the flag no longer appears in any registry, the manifest, or the code, and
CI runs a single state.

## Worked Examples

### Build-time exemplar: Multi-Arch Packaging and `KPACK_SPLIT_ARTIFACTS` (RFC0008)

This is the canonical case where a build-time flag is required. Multi-arch packaging ships kernels for
all architectures as fat binaries; it re-architects the build into a sharded pipeline (a generic-once
stage, a per-architecture parallel split, and a recombine), changes the produced artifact set and
packages (`artifact_generic` plus `artifact_gfxNNNN`, `-kernels-gfxNNNN` packages, split wheels), and
adds build-time tooling (split and recombine, `clang-offload-bundler`, `kpack`) plus a build
dependency (`base/rocm-kpack`). Litmus answers: Q1 yes, Q2 yes, Q5 yes. The toggle changes build
topology and emitted artifacts, so it cannot be a runtime branch, even though its runtime loader is
deliberately binary-neutral. The flag is the existing `KPACK_SPLIT_ARTIFACTS` (ON) in `FLAGS.cmake`.

### Runtime exemplar 1: a new hipDNN backend or provider (the hipDNN instantiation)

This example shows hipDNN instantiating the generic contract: hipDNN's
`EnginePluginManager::validateBeforeAdding` reader is one implementation of the documented reader
contract (b); hipDNN implements its own consumer rather than taking the header. The litmus answers
are all "no" (a dlopen plugin, gated additively), so a runtime flag is used.

```cmake
# RUNTIME_FLAGS.cmake (TheRock, NEW)
therock_declare_runtime_flag(NAME MY_NEW_BACKEND DEFAULT_VALUE OFF
  OWNER my-team CREATED 2026-06-04 EXPIRES 2026-09-01 STAGE canary ISSUE ALMIOPEN-XXXX
  DESCRIPTION "Enable the experimental my_new_backend engine plugin")
```

```json
// shipped share/therock/feature_flags.json (ships automatically via base/aux-overlay alongside the manifest; no toml change; see Install location)
{ "MY_NEW_BACKEND": false }
```

```cpp
// backend/src/plugin/EnginePluginManager.hpp - validateBeforeAdding (EXISTING hook)
// Note: adding a flag check here broadens this hook's purpose; it adds a second, distinct
// rejection reason (flag-disabled) alongside the existing engine-API-major mismatch check.
if (plugin.engineName() == "my_new_backend" && !FeatureFlags::get().enabled("MY_NEW_BACKEND"))
    return reject(plugin, "disabled by feature flag");
```

`validateBeforeAdding` (`backend/src/plugin/EnginePluginManager.hpp`) already rejects plugins on an
engine-API-major mismatch; the flag check is additive to the dlopen plugin registry
(`PluginManagerBase::loadPlugins`, `backend/src/plugin/PluginCore.hpp`), not structural, at the cost
of giving the hook a second responsibility. The field kill switch, with no rebuild, is
`ROCM_FEATURE_MY_NEW_BACKEND=0` (or the hipDNN alias `HIPDNN_FEATURE_MY_NEW_BACKEND=0`).

### Runtime exemplar 2: a new SDPA kernel variant

```cmake
therock_declare_runtime_flag(NAME SDPA_PAGED_KERNEL_V2 DEFAULT_VALUE OFF
  OWNER attention-team CREATED 2026-06-04 EXPIRES 2026-09-01 STAGE canary ISSUE ALMIOPEN-2002
  DESCRIPTION "Enable v2 paged-attention SDPA kernel variant")
```

The consuming check at the kernel-selection entry point reads the same shared
`share/therock/feature_flags.json` via the `dladdr`-based discovery idiom (hipDNN's
`getCurrentModuleDirectory()` is one implementation). Canary ships
`{ "SDPA_PAGED_KERNEL_V2": true }`; `main` ships `false` until promotion. The field kill switch is
`ROCM_FEATURE_SDPA_PAGED_KERNEL_V2=0` (alias `HIPDNN_FEATURE_SDPA_PAGED_KERNEL_V2=0`).

## Alternatives Considered

### Flag kind default

| Option                       | Pros                                                                                                       | Cons                                                                                      | Verdict                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------- |
| **Runtime-default (chosen)** | One binary for all states; immediate field revert; no combinatorial build matrix; matches Chrome guidance. | Disabled code still ships in the binary (size); needs a new runtime mechanism.            | Chosen.                          |
| Build-time-default           | Reuses the existing `FLAGS.cmake` only.                                                                    | Rebuild required to change; combinatorial CI; no field kill switch; binary fragmentation. | Rejected; only for forced cases. |

### Promotion mechanism

| Option                                                                   | Pros                                                                                           | Cons                                                                                                                   | Verdict   |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------- |
| **Fixed, automated canary soak train (chosen; period to be determined)** | Predictable; one soak cycle before swap; uses the canary soak signal; matches Chrome and LLVM. | New canary branch, scheduled job, and a frequent rebase of canary onto main (only difference being the flag defaults). | Chosen.   |
| Promote on every green nightly                                           | Faster.                                                                                        | No soak window; high-churn; no human release-manager gate; the nightly is `main`-only (no canary nightly).             | Rejected. |
| Manual ad-hoc promotion                                                  | No new infrastructure.                                                                         | Unpredictable; no soak guarantee; debt accrues.                                                                        | Rejected. |

### Where flags live

| Option                                      | Pros                                                             | Cons                                                         | Verdict   |
| ------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------ | --------- |
| **TheRock single source of truth (chosen)** | One registry; uniform metadata, manifest, and CI; teams consume. | Cross-repo declare-then-consume indirection.                 | Chosen.   |
| Per-project registries                      | Local autonomy.                                                  | No global inventory; duplicated mechanism; no central audit. | Rejected. |

### Both-state CI (team-owned; decoupled from canary)

| Option                                                                                               | Pros                                                             | Cons                                                                                                        | Verdict                         |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------- |
| **Label or flip-branch both-state, runtime equals one build and two test runs (chosen for runtime)** | No rebuild (binary-neutral); inexpensive; team-owned, on any PR. | Needs the label or dispatch wired through `configure_ci.py` (new work).                                     | Chosen.                         |
| Label or flip-branch both-state, build-time equals two builds                                        | Necessary for build-time flags.                                  | Two builds; amortize with `prebuilt_stages` and `baseline_run_id`, and scope via the label.                 | Chosen (scoped) for build-time. |
| Rely on canary for both-state coverage                                                               | No new mechanism.                                                | The wrong model: canary only soaks the to-be-promoted default; it does not give a team ON-and-OFF coverage. | Rejected.                       |

### Canary flip carrier

| Option                                                                        | Pros                                                                                            | Cons                                                                                               | Verdict   |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------- |
| **Committed `DEFAULT_VALUE` or registry-default change on `canary` (chosen)** | Observable and reviewable on both `canary` and `main`; the promotion diff is the same artifact. | None significant.                                                                                  | Chosen.   |
| TheRock's existing `BRANCH_FLAGS.cmake` branch override                       | Reuses an existing mechanism.                                                                   | Gitignored and not observable on `main`; a flip would be invisible except as a configure-log line. | Rejected. |

## Implementation Phases

| Phase                                                                | Deliverables                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P1: metadata, hygiene, and RBAC**                                  | Extend `therock_declare_flag` with OWNER, CREATED, EXPIRES, and STAGE; surface them in `therock_report_flags()` and the manifest; add an expiry warning and non-mainline `ISSUE` enforcement; fold `THEROCK_FLAG_INCLUDE_PROFILER` into the registry; add `.github/CODEOWNERS` entries (release-manager group) for `FLAGS.cmake`, `RUNTIME_FLAGS.cmake`, `ci_weekly.yml`, and `configure_ci.py`; update `docs/development/flags.md`.                                                                                                                                                                                                                                                                       |
| **P2: generic runtime contract**                                     | `RUNTIME_FLAGS.cmake` plus `therock_declare_runtime_flag` plus `therock_finalize_runtime_flags()`; emit the shared `share/therock/feature_flags.json` from the `base/aux-overlay` step alongside the manifest (it ships automatically via aux-overlay's existing `**/*` catch-all, with no toml change); add `runtime_flags` to the manifest; document the reader contract (location, `dladdr` discovery, `ROCM_FEATURE_*` precedence) and publish the example reference `rocm_feature_flags.h` (copied-in, not shipped or linked) with a standalone-build fallback note; the `rocm-feature-flags --list` helper; wire the first instantiation (hipDNN at `validateBeforeAdding`, using its own consumer). |
| **P3: canary branch, team-owned both-state CI, and canary currency** | Create `canary` (soak-only) and add it to `long_lived_full_match` in `configure_ci.py` so that CI runs on it; document the soak convention (the current batch flipped ON, else matching main); wire the `flag:<NAME>:both` label or dispatch input through `configure_ci.py` (runtime equals one build and two test runs, build-time equals two builds, scoped); add the scheduled frequent rebase of `canary` onto `main` (only difference being the flag defaults).                                                                                                                                                                                                                                      |
| **P4: automated promotion job**                                      | Implement `ci_weekly.yml` (replacing the no-op; cron and period to be determined): the canary soak-signal gate plus the promotion PR (release-manager-merged, not auto-merged) plus the flag-debt audit (the flag's gated code is already on `main` by the rebase model).                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **P5: canary soak signal (optional deepening)**                      | A new per-branch canary validation job (`workflow_dispatch` or matrix checking out canary) plus a Quartz-emitted per-branch `latest_good@canary.json`, if the builds-and-tests-green minimum is judged insufficient.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **P6: build-time both-state build dimension (optional and scoped)**  | A build dimension in `amdgpu_family_matrix.py` and `configure_ci.py` for build-time flags plus `prebuilt_stages` amortization, beyond the label wiring in P3.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **P7: adopt and retire**                                             | Migrate the first real flags (SDPA v2, the new backend); run the first full train; remove the first `default-on` flag to validate retirement.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

## Decisions and Open Questions

### Resolved decisions

1. **Canary branch is a plain `canary` branch,** with CI configured to run on it via `long_lived_full_match` in `configure_ci.py` (presubmit and postsubmit). It is soak-only, not "everything on."
1. **There is no shared runtime library; the header is an example, not a component.** TheRock provides a documented reader contract (location, `dladdr` discovery, `ROCM_FEATURE_<NAME>` precedence) plus an example reference `rocm_feature_flags.h` that implements it. Libraries copy the example header into their own tree (recommended, most portable) or reimplement the contract against their existing environment-flag system; there is no shipped, linked, or auto-included dependency. Standalone builds vendor their own copy and fall back to compile-time defaults.
1. **One reviewed batch per cycle:** small and explicitly reviewed, so that the result remains attributable; no split-attribution without the per-flip machinery.
1. **The global `share/therock/feature_flags.json` ships automatically** via `base/aux-overlay`'s existing `**/*` catch-all (the same path as the manifest), with no `.toml` change. Only optional, opt-in per-library override files add their own `include`.
1. **Promotion cadence is fixed and automated:** the job opens the PR, a release manager merges it, and the frequent rebase keeps canary current with no manual reset.
1. **Both-state CI is team-owned and decoupled from canary,** triggered via the `flag:<NAME>:both` label or dispatch and/or a flip branch: runtime equals one build and two test runs; build-time equals two builds.

### Open for reviewer input

1. **Canary soak test scope:** which tests run on canary, whether and what additional testing is stacked on top, and the soak-cycle length. The minimum is canary builds and tests green; this can optionally be deepened with the per-branch `latest_good@canary.json` (P5). The exact test set, any additional stacked testing, and the cycle length are open.
1. **Promotion cadence period:** weekly, bi-weekly, or monthly (and the exact `ci_weekly.yml` slot). Only "fixed and automated" is settled; the period and slot are open.
1. **When to build the per-PR both-state label mechanism:** wire `flag:<NAME>:both` through `configure_ci.py` now, or rely initially on flip branches and add the label later. Flip branches need no new CI wiring; the label is the more ergonomic per-PR path.

## Summary

This RFC turns feature flags into a disciplined, auditable lifecycle in TheRock and its
libraries.

**What it adds.** It extends the existing `FLAGS.cmake` build-flag registry with mandatory
owner, expiry, and stage metadata, and adds a generic runtime-flag contract: a shipped
`share/therock/feature_flags.json`, a documented reader (`dladdr` discovery,
`ROCM_FEATURE_<NAME>` precedence), and an example `rocm_feature_flags.h` that implements it.
Libraries copy the header or reimplement the contract — there is no shipped, linked, or
auto-included dependency, and standalone builds fall back to compile-time defaults. hipDNN is
the first adopter, using its own `EngineOverrideConfig` and `validateBeforeAdding` path.

**How a flag is promoted.** A fixed, automated canary-to-mainline train (period to be
determined) runs in the reserved `ci_weekly.yml` slot. Canary is a soak-and-staging branch: the
team flips the candidate default(s) to ON and soaks for one cycle (minimum signal: canary builds
and tests green; RFC0011's `latest_good.json` remains the `main` and nightly signal). On a green
soak the automated job opens a promotion PR that a release manager — not the flip author —
merges. Frequently rebasing canary onto `main` keeps it current: the gated code is always on
`main` first, the only divergence is the soaking defaults, and a promoted flip stops being a
divergence with no manual reset. The default is one reviewed batch per cycle, and a bad flip is
simply dropped.

**The decision rule.** Runtime by default; build-time only when the change alters artifacts,
topology, or ABI (Multi-Arch Packaging is the exemplar). Both states are exercised by a
team-owned both-state CI mechanism, decoupled from canary and triggered by a `flag:<NAME>:both`
label, a dispatch input, or a flip branch — one build and two test runs for runtime flags, two
builds for build-time (the main reason to prefer runtime).

**Backout.** `ROCM_FEATURE_<NAME>=0` is the minutes-scale, per-host kill switch (ROCm has no
fleet push); editing the installed JSON sets the next-package default (a respin for existing
installs); build-time backout takes one cycle.

### Quick reference

Full detail in Maintainer Playbooks.

| To do this                         | Do the following                                                                                                                                                                                                       | Success criterion                                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Start a new feature**            | Pick runtime (default) or build-time; add a default-OFF flag in TheRock (lands quickly); after the bump reaches your repository, guard your code at one check; ship it OFF on trunk.                                   | The flag exists (OFF), the code is guarded at one point, and trunk is green with the feature OFF. |
| **Test it both ways**              | Add the `flag:<NAME>:both` label (or use a flip branch). Runtime: one build, tests run twice with `ROCM_FEATURE_<NAME>=0` then `=1`. Build-time: build OFF and ON, testing each.                                       | The affected tests are green in both states on CI.                                                |
| **Turn it on for everyone**        | Open a PR flipping the default ON into `canary` (`STAGE: canary`); let it soak one cycle green; the automated job opens the `main` PR; a release manager (not you) merges it.                                          | The default-ON change is merged on `main` by a release manager, and the nightly is green.         |
| **Turn it off quickly (it broke)** | Set `ROCM_FEATURE_<NAME>=0` on the affected host (minutes, no rebuild). For everyone: land a PR setting the default back to OFF; existing installs need a respin.                                                      | The bad path no longer runs.                                                                      |
| **Clean up afterward**             | After ~one cycle default-on, drop both-state CI for the flag, delete the dead OFF code from your library, then remove the flag declaration from TheRock. (A `long-lived` flag is the exception — it is never retired.) | The flag is gone from the registry, the manifest, and the code.                                   |

The core sequence: add a runtime flag, default off; guard one entry point; soak on canary;
promote via the release-manager-merged PR; revert with `ROCM_FEATURE_<NAME>=0`.

## References

- RFC0008: Multi-Architecture Packaging with Kpack (`docs/rfcs/RFC0008-Multi-Arch-Packaging.md`); build-time exemplar.
- RFC0011: Quartz, Central CI/CD Data Hub (`latest_good.json` green signal).
- TheRock: `FLAGS.cmake`, `cmake/therock_flag_utils.cmake`, `BRANCH_FLAGS.cmake`, `build_tools/generate_therock_manifest.py`, `docs/development/flags.md`.
- TheRock CI and cross-repo: `build_tools/github_actions/configure_ci.py`, `build_tools/github_actions/amdgpu_family_matrix.py`, `.github/workflows/{ci_weekly,ci_nightly,bump_submodules}.yml`, `build_tools/github_actions/bump_automation.py`, `.github/CODEOWNERS`, `docs/development/ci_behavior_manipulation.md`, `docs/packaging/versioning.md`.
- TheRock artifacts: `base/artifact.toml` (the `base/aux-overlay` component's `**/*` catch-all that ships `share/therock/**` automatically), `base/aux-overlay/CMakeLists.txt` (where the manifest, and the proposed `feature_flags.json`, is generated and installed to `share/therock`), `build_tools/_therock_utils/artifact_builder.py` (the default `lib` component is `.so`-only, relevant only to opt-in per-library override files), `docs/development/artifacts.md`, `ml-libs/artifact-hipdnn.toml`.
- hipDNN runtime (one instantiation): `backend/src/heuristics/config/EngineOverrideConfig.hpp`, `backend/src/plugin/{PluginCore,EnginePluginManager}.hpp`, `backend/src/PlatformUtils.linux.cpp`, `projects/hipdnn/data_sdk/include/hipdnn_data_sdk/utilities/PlatformUtils.linux.hpp`.
- Google Chrome: release channels (Canary, Dev, Beta, Stable); `base::Feature` plus Finch field trials plus kill switch; `chrome://flags` expiration; "prefer `base::Feature` over switches."
  - Configuration and `base::Feature`: <https://chromium.googlesource.com/chromium/src/+/main/docs/configuration.md>
  - Flag expiry: <https://chromium.googlesource.com/chromium/src/+/main/docs/flag_expiry.md>
- LLVM: release branch plus release-manager-gated backports; `/cherry-pick` automated backport.
  - Release process: <https://llvm.org/docs/HowToReleaseLLVM.html>
  - Backport and `/cherry-pick`: <https://llvm.org/docs/GitHub.html#backporting-fixes-to-the-release-branches>
- Martin Fowler, "Feature Toggles (aka Feature Flags)": <https://martinfowler.com/articles/feature-toggles.html>

## Revision History

- **2026-06-04**: Initial draft.
- **2026-06-15**: Current revision. Defines the generic runtime-flag contract with an example reference `rocm_feature_flags.h` header (copied in or reimplemented per project; no shipped, linked, or auto-included dependency) and a standalone-build fallback; specifies the canary soak-and-promote train with a fixed, automated cadence (period to be determined); specifies team-owned both-state CI decoupled from canary; adds flag metadata, hygiene, and retirement; clarifies that adding a flag does not block development and that the release-manager gate applies to promotion rather than to flag declaration; and adds goal-based maintainer playbooks and a quick-start summary.
- **2026-06-24**: Editorial revision. Consolidated the separate Summary and Quick-Start Summary into a single Summary (moved before References); tightened the Overview, Glossary, Goals, and other sections for concision and readability; added the `long-lived` value to the `STAGE` enum and Stages table for consistency, and exempted `long-lived` from the `therock_finalize_flags()` `ISSUE`-on-`main` enforcement so a permanent kill switch need not carry a perpetual tracking issue. Reordered the flag-retirement steps (Playbook D, Team Workflow step 7, Summary "Clean up" row) into dependency order — CI off, then dead code, then declaration — which removes the consumer before the declaration it depends on and is uniformly safe for runtime and build-time flags (once the gated code is gone, removing the declaration can neither revert the live build nor break a consumer).
