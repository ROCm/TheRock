# RFC00XX: Feature Flag Lifecycle and Canary Promotion

- **Authors:** Brian Harrison (bharriso), Tony Davis (tony-davis)
- **Created:** 2026-06-04
- **Modified:** 2026-08-04
- **Status:** Draft
- **Discussion:** TBD (GitHub Discussion link to be added)

> **In a hurry?** Jump to the [Quick reference](#quick-reference) cheat-sheet, or the [Maintainer Playbooks](#maintainer-playbooks) for step-by-step procedures.

## Overview

TheRock and the ROCm libraries it builds need one disciplined way to introduce risky
changes incrementally: gate them behind a flag, exercise both code paths in CI, soak a
candidate default for a cycle, promote it on a predictable cadence, and back it out quickly
when it misbehaves. Today's mechanism, the `FLAGS.cmake` build-flag registry, is real but
partial: no runtime dimension, no metadata or expiry, no team-owned both-state CI, and no
branch-level promotion.

This RFC proposes a complete feature-flag lifecycle built on two flag kinds and a fixed,
automated canary-to-mainline promotion train, modeled on Google Chrome's release channels and
LLVM's release-manager-gated backports. Canary is a soak-and-staging branch: the team flips
the default of the flag (or flags) being promoted to ON, soaks it for one cadence cycle, then
swaps the default on `main`.

The proposal extends the build-flag registry rather than replacing it, and adds a parallel
runtime-flag mechanism, the preferred path, because one binary serves every flag state and
can be reverted in the field with no rebuild. The runtime mechanism is a generic,
library-agnostic TheRock contract: a JSON-location convention, a discovery idiom, and a
precedence order. To ease adoption, TheRock also publishes a small reference header,
`rocm_feature_flags.h`, that implements the contract, but it is not shipped, linked, or
auto-included; there is no `.so` and no package dependency. A library either copies the header
into its own tree (the most portable path, which this RFC recommends) or reimplements the
contract against its existing environment-flag system. So the contract is the standard; the
header is a reference a project may copy. hipDNN, the first adopter, uses its own config reader
rather than the header.

**Flags are ephemeral by design.** A feature flag in this lifecycle is temporary scaffolding for
landing a risky change incrementally. It is not a supported product configuration knob and not a
permanent tuning surface for users. Every flag carries an owner and an expiry; once its default has
been promoted to ON and has settled for roughly one cycle, the flag and its now-dead OFF path are
removed, along with the declaration. The single exception is an explicitly marked `long-lived`
operational kill switch. This is why the lifecycle below treats retirement as a mandatory stage rather
than an afterthought: a flag that outlives its purpose is debt, not a feature.

### Glossary

| Term                                          | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Flag**                                      | A named, typed control gating an in-development change. `BOOL` is the default and the normal case for lifecycle purposes; build-time flags may also be `INTEGER` with a `VALID_VALUES` set (PR #6984). Temporary by design: it carries an owner and an expiry and is retired once its default is promoted and settles, so it is not a permanent or supported configuration knob (the `long-lived` kill switch is the sole exception). Two kinds: runtime and build-time.                   |
| **Flag flip**                                 | Changing a flag's effective default (for example, OFF to ON) for a branch or channel.                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Binary-neutral**                            | A change that does not alter emitted artifacts, ABI, or build topology, and can therefore be gated at runtime.                                                                                                                                                                                                                                                                                                                                                                             |
| **Canary**                                    | The soak-and-staging branch. The team flips the current promotion batch to ON here, so it soaks for one cadence cycle before the default is swapped on `main`. A flag may be OFF on both `main` and canary; canary is not an "everything on" branch, nor the mechanism for both-state CI coverage.                                                                                                                                                                                         |
| **Soak**                                      | The period a flipped default spends on canary, with builds and tests green, before promotion.                                                                                                                                                                                                                                                                                                                                                                                              |
| **Promotion**                                 | Swapping a soaked, green flag default from canary onto mainline by means of the automated train.                                                                                                                                                                                                                                                                                                                                                                                           |
| **Mainline**                                  | `main`, the trunk; the default state users receive from stable and nightly builds.                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Kill switch**                               | Reverting a flag to OFF in the field without a rebuild (runtime) or by revert and rebuild (build-time).                                                                                                                                                                                                                                                                                                                                                                                    |
| **Both-state CI**                             | Running CI for a single change in both flag states, ON and OFF, so neither code path goes stale. A team-owned mechanism on a team's own PR or branch, triggered by the `ci:flag-both-state` label (with the single flag named in the PR description) or a dispatch input, or by a flip branch. Distinct from canary, which soaks only a candidate default. Costs one build and two test runs for a runtime flag; two builds and tests for a build-time flag.                               |
| **Standalone build**                          | Building a ROCm library on its own, outside a full TheRock assembly, where there is no build-flag provider state, no shipped `share/therock/feature_flags.json`, and no TheRock header on the include path. Build-time: `rocm_resolve_build_flag()` creates the project's own cache variable and uses the project default. Runtime: the project carries its own copy of the reader header, and every flag resolves to its compile-time default unless overridden by `ROCM_FEATURE_<NAME>`. |
| **Build-flag provider protocol**              | The versioned, typed contract by which TheRock hands resolved build-time flag values to a project being built inside it (PR #6984): TheRock emits `rocm_build_flags_state.cmake`, and the project resolves each flag through `rocm_resolve_build_flag()` from its super-repository's verbatim copy of `ROCMBuildFlags.cmake`. Fail-closed and build-time only; never installed.                                                                                                            |
| **Reference header (`rocm_feature_flags.h`)** | An example implementation of the *runtime* reader contract that TheRock publishes for projects to copy into their own tree or reimplement against their existing environment-flag system. Not shipped, linked, or auto-included; there is no `.so` and no package dependency. The runtime analogue of `ROCMBuildFlags.cmake`, and copied in for the same reason.                                                                                                                           |

## Goals

1. **Define a flag taxonomy and a firm decision rule:** runtime by default; build-time only for build-structural, ABI, or artifact-altering changes.
1. **Add a generic runtime flag contract** to TheRock: a JSON-location convention, `dladdr`-relative discovery, and a resolution order checked highest-first (`ROCM_FEATURE_<NAME>` environment variable, then installed JSON, then compile-time default), plus an example `rocm_feature_flags.h` header that implements it. Teams copy the header into their own tree or reimplement the contract against their existing environment-flag system; there is no shipped, linked, or auto-included dependency. hipDNN's `EngineOverrideConfig`-style reader is one instantiation, not the contract itself.
1. **Make flags first-class inventory:** owner, created date, expiry, stage, and tracking issue on every flag, surfaced in the configure report and the shipped manifest.
1. **Establish a canary-to-mainline promotion train** (fixed and automated; period to be determined). Canary is a soak-and-staging branch: the team flips the candidate default(s) to ON, soaks for one cadence cycle, then swaps the default on `main`. The minimum soak signal is canary builds and tests green across the cycle; RFC0011's `latest_good.json` remains the `main` and nightly health signal, with an optional per-branch `latest_good@canary.json` as a possible deepening.
1. **Make both-state CI a first-class, team-owned mechanism, decoupled from canary, and uniform across every consumer CI.** A team developing behind a flag can run CI in both states (ON and OFF) for its PR, triggered by a label (`ci:flag-both-state`, with the single flag named in the PR description), a `workflow_dispatch` input, or a flip branch whose committed config flips the default. The trigger grammar and the OFF/ON leg semantics are a single repo-agnostic contract with one shared implementation in TheRock, which TheRock, rocm-libraries, and rocm-systems all adopt, each at its own call site (see Multi-repo adoption). Runtime flags cost one build and two test runs; build-time flags cost two builds.
1. **Specify backout, kill-switch, and failed-promotion policy.** The environment kill switch (`ROCM_FEATURE_<NAME>=0`) is the per-process, per-host, minutes-scale revert; editing the installed JSON sets the next-package channel default (which requires a respin). Build-time backout takes one cycle. A bad flip is dropped from the canary batch, not carried forward.
1. **Define flag hygiene:** expiry enforcement, mandatory retirement after default-on, and a per-cycle flag-debt audit.
1. **Provide teams a concrete playbook** with clear ownership boundaries between project teams, TheRock, and Quartz.

## Non-Goals

- Replacing the existing `FLAGS.cmake` registry; this RFC extends it.
- Defining the build-time declaration or consumption mechanism. [PR #6984](https://github.com/ROCm/TheRock/pull/6984) does that (typed `BOOL`/`INTEGER` flags, provider state, `rocm_resolve_build_flag()`, and the fail-closed `ROCM_BUILD_FLAG(name)` accessor). This RFC adopts it and adds the lifecycle layer above it.
- Defining the CI data and notification substrate; that is RFC0011 (Quartz), whose signals this RFC consumes.
- Percentage or telemetry-driven rollout (a gradual percentage ramp). A flag here resolves to one discrete value for a whole build or process (usually on or off), not to a per-user or per-request sample; no telemetry pipeline exists in ROCm today to drive a percentage ramp.
- Artifact-level RC-to-final promotion (`build_tools/packaging/promote_*`); that remains as-is, and this RFC adds branch-level flag promotion alongside it.
- Mandating that every existing flag immediately gain a runtime equivalent.
- Treating flags as a permanent or user-facing configuration surface. A flag is temporary scaffolding with a mandatory expiry and retirement, not a supported long-term tuning knob; the explicitly marked `long-lived` kill switch is the sole exception.

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

| Fact                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Where                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Trunk-based on `main`; submodules auto-rolled by `bump_submodules.yml` (push trigger `branches:[main]`, cron every 12h); `bump_automation.py` hardcodes `base="main"`. Submodule rolls land on `main` only.                                                                                                                                                                                                                                                                                                                                                                                                                              | repo root, `build_tools/github_actions/bump_automation.py`                                     |
| Long-lived-branch coverage: pushes to `main`, `multi_arch/**`, and `release/therock-*` (the `on.push.branches` list in `.github/workflows/multi_arch_ci.yml`) run the broader presubmit-plus-postsubmit tier; pull requests run presubmit only. Trigger-type tier selection lives in `configure_multi_arch_ci.py`.                                                                                                                                                                                                                                                                                                                       | `.github/workflows/multi_arch_ci.yml`, `build_tools/github_actions/configure_multi_arch_ci.py` |
| Branch-prefix routing exists for `release/therock-*` branches (push-triggered postsubmit), declared in the workflow `on.push.branches` list.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `.github/workflows/multi_arch_ci.yml`                                                          |
| Data-driven matrix: families by platforms by build_variants; tiers presubmit within postsubmit within nightly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `configure_multi_arch_ci.py`, `amdgpu_family_matrix.py`                                        |
| CI controls: PR labels (`ci:skip`, `ci:run-all-archs`, `test_filter:<…>`, `test:<proj>`), `workflow_dispatch` inputs including `prebuilt_stages` and `baseline_run_id`; the `build_variant` environment value selects release, asan, or tsan.                                                                                                                                                                                                                                                                                                                                                                                            | `docs/development/ci_behavior_manipulation.md`                                                 |
| `configure_multi_arch_ci.py` already contains `schedule`-event routing to a full nightly tier, but no workflow currently triggers it on a schedule. The only scheduled workflows in the tree today are `bump_submodules.yml` (every 12h) and `gitleaks_main.yml` (weekly); there is no scheduled multi-arch run and no separate promotion workflow. This RFC proposes adding a `schedule` trigger to `multi_arch_ci.yml` (routed to the nightly tier the orchestrator already supports) and attaching the promotion job to that scheduled run rather than creating a standalone workflow; the cadence is to be determined (see cadence). | `configure_multi_arch_ci.py`, `.github/workflows/multi_arch_ci.yml`                            |
| There is no dedicated `ci_nightly.yml` in the tree today (removed in the multi-arch migration). The design constraint still holds: GitHub `schedule` crons run only on the default branch (`main`), so a scheduled nightly cannot run on canary.                                                                                                                                                                                                                                                                                                                                                                                         | `.github/workflows/ci_nightly.yml`                                                             |
| Release channels are stable, prerelease, nightly, dev, and dev-builds; promotion today is artifact-level (RC to final), not branch to branch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `docs/packaging/versioning.md`                                                                 |
| `latest_good.json` is a single global symlink to the most-recent fully-passing nightly (RFC0011 Quartz). It tracks `main` and nightly health; there is no per-branch canary equivalent today.                                                                                                                                                                                                                                                                                                                                                                                                                                            | RFC0011                                                                                        |
| `.github/CODEOWNERS` does not currently cover `FLAGS.cmake` or `configure_multi_arch_ci.py`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `.github/CODEOWNERS`                                                                           |

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
| Lives in                    | TheRock runtime registry, shipped as `feature_flags.json` (NEW) | `FLAGS.cmake` registry (EXISTING)                                        |
| Value type                  | `BOOL` (`0`/`1`)                                                | `BOOL` or `INTEGER` with `VALID_VALUES` (PR #6984)                       |
| Consumed via                | vendored `rocm_feature_flags.h` or an equivalent reader (NEW)   | `rocm_resolve_build_flag()` from `ROCMBuildFlags.cmake` (PR #6984)       |
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

TheRock owns the *mechanism*: the declaration surface, the typed provider protocol, the manifest,
CI, and promotion. It does not have to be the origination point for every flag. A flag may be
declared either in TheRock's registry or in the consuming super-repository (rocm-libraries,
rocm-systems), whichever is closer to the team doing the work. In an integrated build TheRock's
value is authoritative regardless of where the flag was declared; in a standalone build the
project's own default applies. Project teams consume the flag in their code and own its behavior
and lifecycle. Teams choose the kind, preferring runtime.

> **Why origination is not restricted to TheRock.** Requiring every flag to start in TheRock reads
> cleanly, but it interlocks the two repositories at exactly the wrong moment: a team cannot begin
> guarding code until a declaration lands in TheRock *and* that TheRock version reaches their
> branch. The typed provider protocol (PR #6984) removes the need for that interlock, because a
> project resolves a flag through `rocm_resolve_build_flag()` with its own default and defers to the
> provider when one is present. TheRock remains the single source of truth for the *global
> inventory* (every flag, wherever declared, is aggregated into the manifest and the hygiene audit);
> it is not the mandatory birthplace.

#### Adding a flag does not block development

Declaring a new flag (adding a default-OFF entry to `FLAGS.cmake` or `RUNTIME_FLAGS.cmake` with
its metadata) is low-risk and not gated behind heavyweight review. A default-off flag changes
nothing for users; it only adds the ability to protect an in-development feature. Two steps are
intentionally decoupled:

1. **Landing the flag declaration.** This always lands, and lands quickly. It is a small,
   additive, default-off change that immediately lets a team guard a new feature. It is not gated by
   the release-manager or CODEOWNERS process (the CODEOWNERS gate below applies only to default flips
   and promotion), and it waits on nothing. New-flag PRs are treated as priority reviews, so review
   never blocks a team from starting work behind a flag.
1. **Enabling CI for the flag.** Wiring the flag into both-state CI (the two-build path for
   build-time, or the one-build, two-test-run path for runtime) is a follow-on step that blocks
   neither the declaration nor the feature. A team can land the flag and guard its code first, then
   add CI.

So starting a new feature depends on one thing: a default-off declaration landing somewhere the team
can reach. If the flag is declared in the consuming super-repository, that is a single PR in the
repository the team already works in, and there is no cross-repo wait at all. If it is declared in
TheRock, the team advances the TheRock version its branch resolves against (in rocm-libraries, by
resyncing with `develop`, which moves the merge-base that
[PR #9602](https://github.com/ROCm/rocm-libraries/pull/9602) pins the TheRock ref to) rather than
waiting on the 12-hour submodule bump cron. Either way it is self-service and neither path is a
review chokepoint.

```
Declared in TheRock (FLAGS.cmake / runtime registry)
   OR in the consuming super-repo (rocm-libraries, rocm-systems)
        │  NAME + TYPE + metadata + default
        ▼
TheRock aggregates the global inventory and emits provider state
        │
        ├─► rocm_build_flags_state.cmake     (build-time provider state, PR #6984)
        ├─► therock_manifest.json["flags"]   (build-time as-built, EXISTING)
        └─► feature_flags.json               (runtime, NEW)
        │
        ▼
Project code resolves the flag (rocm_resolve_build_flag / runtime reader):
provider value wins in an integrated build, the project default in a standalone
build. The project guards one entry point and owns the behavior.
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

### Distribution boundary (what ships, and what must not)

The build-flag provider protocol (PR #6984) draws a hard line: `ROCMBuildFlags.cmake`, the provider
state file, and any generated private flag header are **build-time inputs only**. They are never
installed, never exported as a target, and never appear in an installed package config, so an
application calling `find_package()` on an installed ROCm library acquires no build-system
dependency. This RFC adopts that boundary unchanged.

The boundary constrains what a build **consumes**, not what a package **records**. Three artifacts
sit in three different places, and keeping them distinct is what makes the rule workable:

| Artifact                                                                                                         | Read when                                    | Installed?    | Why                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Provider state and helper (`rocm_build_flags_state.cmake`, `ROCMBuildFlags.cmake`, the generated private header) | Configuring and compiling the library        | No, by design | A resolved value is baked into the binary. Shipping the input would export build plumbing to `find_package()` consumers for no benefit.                                                |
| As-built record: the manifest `flags` block (EXISTING)                                                           | After install, by people and tooling         | Yes           | Provenance. Which flags a package was built with is essential for triage, support, and the flag-debt audit. It is a record, never an input: no build reads it to configure or compile. |
| Runtime `feature_flags.json` (NEW)                                                                               | Every process start of the installed library | Yes           | A value that cannot be read after install is not a runtime flag. The installed JSON *is* the no-rebuild channel default, and removing it removes the kill switch.                      |

**The rule is about direction, not secrecy.** Nothing that participates in producing a binary ships
as an input to somebody else's build. What does ship is the state that must be readable after the
binary exists (the runtime defaults) and the record of how the binary was produced (the manifest).
Build-time flag state is therefore fully observable on an installed system even though the provider
state file is not installed. The runtime side keeps the rest of the boundary intact:
`rocm_feature_flags.h` is copied in rather than shipped, there is no `.so` to link, no exported
target, and no `find_dependency()`.

The corollary from PR #6984 still binds: a flag that must affect a public header or ABI is outside
both contracts and needs a separate, project-owned installed configuration contract. Such a change
is a build-time litmus "yes" (question 3) and is not a runtime flag.

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
  error. This is deliberate, and it is the one place the runtime contract diverges from the
  build-time protocol's fail-closed posture: a runtime reader must work in a standalone build where
  no JSON exists, so a missing file cannot be an error. The divergence is scoped to the file, not to
  the flag name (next bullet).
- **fails closed on the flag *name*, not on the file.** A misspelled or unregistered flag name must
  be detectable rather than resolving silently to a default forever. Consumers should name flags
  through a generated, per-project accessor rather than a bare string or a plain `#if`, so that an
  unknown name is a compile error, exactly as the build-time protocol's function-like
  `ROCM_BUILD_FLAG(name)` macro achieves (PR #6984). `rocm-feature-flags --list` reports any name
  present in an installed or redirected JSON that is not in the manifest's `runtime_flags` block.
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
question requires it. The declaration surface and the mechanism by which a project consumes a value
are **not** proposed here: PR #6984 defines them, and this RFC layers lifecycle metadata on top
without changing them.

### Mechanism (PR #6984; this RFC does not redefine it)

| Piece                            | What it does                                                                                                                                                                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TYPE` on `therock_declare_flag` | `BOOL` (default) or `INTEGER` with an optional `VALID_VALUES` set. BOOL normalizes to `0`/`1`; INTEGER requires canonical signed base-10.                                                                                                   |
| `rocm_build_flags_state.cmake`   | Versioned provider state emitted by `therock_finalize_flags()` into the build tree, written as plain `set()` calls so a CMake 3.7 project needs no JSON parser. Ends with a completion marker.                                              |
| `ROCMBuildFlags.cmake`           | The consumer API. TheRock's copy is canonical; each consuming super-repository carries a verbatim copy. No external dependencies.                                                                                                           |
| `rocm_resolve_build_flag()`      | Resolves one flag. Provider value is authoritative in an integrated build (and defining the project's own cache variable is then an error, so nothing silently shadows TheRock); the project's cache variable and default apply standalone. |
| `ROCM_BUILD_FLAG(name)`          | The generated function-like C/C++ accessor. An unknown name is a syntax error in both normal code and `#if`, instead of the preprocessor's silent undefined-to-zero.                                                                        |
| `base/aux-overlay`               | The canonical worked example, with permanent C and C++ conformance canaries compiled unconditionally.                                                                                                                                       |

Two properties of that design carry consequences for this RFC and are adopted as given: it is
**fail-closed** (a missing, incomplete, duplicate, malformed, or type-mismatched value is a
configure error, not a default), and it is **build-time only** (never installed; see Distribution
boundary above).

### Metadata extension (NEW)

Extend `therock_declare_flag` with `OWNER`, `CREATED`, `EXPIRES`, and `STAGE`, keeping `TYPE`,
`VALID_VALUES`, `ISSUE` (required for non-mainline stages), and `DESCRIPTION`. The lifecycle
metadata is orthogonal to the type: an `INTEGER` flag has an owner, an expiry, and a stage on the
same terms as a `BOOL` one.

```cmake
therock_declare_flag(
  NAME          KPACK_SPLIT_ARTIFACTS
  TYPE          BOOL
  DEFAULT_VALUE ON
  OWNER         packaging-team
  CREATED       2025-11-20
  EXPIRES       2027-01-01
  STAGE         default-on
  DESCRIPTION   "Split target-specific artifacts into generic and arch-specific components"
)
```

> **Note on non-BOOL flags and the promotion train.** The canary train promotes a *default*, which
> is well-defined for any type: a soaking `INTEGER` flag has a candidate default value rather than a
> candidate ON. Both-state CI, however, is defined for two states. For an `INTEGER` flag it exercises
> the current default and the candidate value, not the full `VALID_VALUES` cross-product; exhaustive
> coverage of a multi-valued flag is the owning team's responsibility and is out of scope here.

`therock_report_flags()` is extended to print owner, stage, and expiry; the manifest `flags` block is
extended to carry the same metadata. `therock_finalize_flags()` (new behavior) warns when a flag is
past `EXPIRES` and errors on `main` when a `STAGE` other than `default-on`, `deprecated`, or
`long-lived` ships without an `ISSUE`.

### Fold legacy flags into the registry (NEW cleanup)

`THEROCK_FLAG_INCLUDE_PROFILER` (currently `cmake_dependent_option`, not reported, not in the
manifest) and any similar out-of-registry options are migrated into `FLAGS.cmake` so that there is one
pattern. This is tracked as a phase-1 task.

### Setting build-time flag state

| Mechanism                          | Reviewable?                     | Use                                                                                                                                    |
| ---------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `-DTHEROCK_FLAG_<NAME>` (CLI)      | Not applicable (per-invocation) | Developer or one-off CI build.                                                                                                         |
| `DEFAULT_VALUE` in `FLAGS.cmake`   | Yes (reviewed diff)             | The reviewed default; changed on `canary` to soak, and on `main` to promote.                                                           |
| The project's own `CACHE_VARIABLE` | Yes (reviewed diff)             | Standalone builds only. Setting it in an integrated build is a configure error by design, so it can never shadow the promoted default. |

## Flag Lifecycle and Hygiene

Every flag is inventory with a carrying cost. Mandatory metadata: NAME, OWNER, CREATED,
EXPIRES, STAGE, ISSUE (non-mainline), DESCRIPTION.

### Stages

| Stage            | Default                                     | Branch         | Notes                                                                                                                                      |
| ---------------- | ------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `in-development` | OFF                                         | feature branch | Immature; not yet promoting.                                                                                                               |
| `canary`         | ON on canary (this batch only), OFF on main | canary         | Soaking one cycle before the default is swapped on main. Only flags in the current promotion batch are ON on canary.                       |
| `default-on`     | ON                                          | main           | Promoted; the new default.                                                                                                                 |
| `deprecated`     | ON, removal scheduled                       | main           | The post-`default-on` kill-switch retention window (approximately one cycle, per Hygiene rule 3); then the flag and dead path are removed. |
| `long-lived`     | ON (or as configured)                       | main           | The justified exception: a permanent operational kill switch or permissioning toggle, exempt from expiry (see Hygiene rule 6).             |

### Hygiene rules

1. **Owner and expiry are mandatory.** No flag merges without both. `EXPIRES` is a review-by date.
1. **Minimize toggle points:** guard entry points only (for example, one `validateBeforeAdding` check), and never sprinkle conditionals.
1. **The kill switch is retained approximately one cycle** after `default-on`, then the flag and dead code path are retired via a normal PR; dependents are already removed, so it is a pure dead-code deletion (team-owned).
1. **Test both states while live** (team-owned both-state CI; see CI Integration), and collapse to one when removed.
1. **A per-cycle flag-debt audit** runs at each promotion cycle (the automated promotion job; see cadence): it lists flags past `EXPIRES`, flags `default-on` for more than one cycle and still present, and stale flags, and files removal issues.
1. **Long-lived flags are the justified exception:** a permanent operational kill switch or permissioning toggle may live indefinitely but must be explicitly marked `STAGE: long-lived` and exempt from expiry.

## Branching and Canary-to-Mainline Promotion

### Branch model

| Branch      | Role                                                                                                                                               | CI                                                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `feature/*` | Immature flags (`in-development`), OFF everywhere.                                                                                                 | Presubmit only (EXISTING).                                                                                                     |
| `canary`    | Soak the to-be-promoted flag default or defaults for one cycle. Only the current promotion batch is flipped ON here; everything else matches main. | CI configured to run on `canary` (add `canary` to `on.push.branches` in `.github/workflows/multi_arch_ci.yml` for postsubmit). |
| `main`      | Mainline and trunk; the promotion target.                                                                                                          | Presubmit and postsubmit (EXISTING).                                                                                           |

**Canary branch.** Canary is a plain `canary` branch. CI is configured to run on it by adding
`canary` to the `on.push.branches` list in `.github/workflows/multi_arch_ci.yml` (alongside `main`,
`multi_arch/**`, and `release/therock-*`), giving presubmit and postsubmit coverage.
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
one small, explicitly reviewed batch per soak cycle. Per-flip independent certification (per-flip sub-branches each independently signalled,
or Quartz green-per-flag attribution) is new machinery and is not assumed. Until it exists, the claim
that independent flips can proceed past a bad one is not made.

### Promotion cadence (fixed and automated; period to be determined)

Promotion runs on a fixed, automated schedule; this is the only settled cadence decision. The period
(weekly, bi-weekly, or monthly) is open for reviewers, as is the exact schedule. The job runs as a
scheduled step on the multi-arch CI. `configure_multi_arch_ci.py` already routes the `schedule` event
to a nightly tier, but a `schedule` trigger must first be added to `multi_arch_ci.yml`, since no
workflow fires that orchestrator on a schedule today; the promotion job then attaches to that run
rather than a new workflow, and the chosen period is to be determined. On each scheduled fire, the job
opens the promotion PR, a release manager (not the flip author) reviews and merges it, the frequent
rebase then keeps canary current, and the next batch of flip PRs is merged in to soak, with no manual
reset.

| Phase                                   | Action                                                                                                                                                                                                                                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Scheduled fire: promotion PR opened** | If the soak signal was green across the whole soak cycle (minimum: canary builds and tests green) and the gates pass, the job opens the promotion PR landing the soaked default change or changes (`FLAGS.cmake` `DEFAULT_VALUE` or runtime registry) on `main`, and emits the flag-debt audit.  |
| **RM merge**                            | A release manager (not the flip author) reviews and merges the promotion PR to `main`. After the next frequent rebase, the promoted flip is in `main` and is no longer a canary diff, so no manual reset is required. Automation merges the next approved batch of flip PRs into canary to soak. |
| **Soak cycle**                          | Canary soaks the current batch; each push runs CI, accumulating the soak-cycle signal. The `main`-only nightly continues to feed `latest_good.json` for `main` health.                                                                                                                           |

**No manual canary reset.** Promotion merges the soaked default onto `main`; the frequent rebase of
`canary` onto `main` then keeps canary current automatically (the promoted flip is no longer a
divergence once it is in `main`), and automation merges the next approved batch of flip PRs into
`canary` after the cycle.

Train discipline: a flip that misses the cut waits for the next cycle; risky work is
never rushed onto the promoting line.

The job that automates this train, including its soak-signal gate, the release-manager-merged
promotion PR, and the flag-debt audit, is specified in CI Integration, Promotion job (a scheduled step
on the multi-arch CI).

## CI Integration: Team-Owned Both-State Testing

**Principle.** A team developing behind a flag must be able to run CI in both states, flag ON and
flag OFF, so that neither path becomes stale. This is a first-class, team-owned mechanism, independent
of canary: canary soaks a to-be-promoted default, whereas both-state CI is how a team validates its
own change under both settings, on its own PR or branch, whenever it chooses. Nothing toggles a flag
from CI today; wiring the trigger through `configure_multi_arch_ci.py` (and the consumer orchestrators) is new work.

### Trigger: a label, or a designated flip branch

A team requests both-state coverage two ways (either, or both):

- **PR label plus a PR-body flag declaration, or dispatch input.** Apply the generic
  `ci:flag-both-state` label and name the single flag to exercise on its own line in the PR
  description, `Flag: <NAME>`. The label needs no per-flag maintenance (it exists once), and because
  applying a label already requires triage/write access, it doubles as the permission gate; the body
  just carries the parameter. CI validates `<NAME>` against the flag registry and fails loud on an
  unknown name. A `workflow_dispatch` input carries the same flag name. Exactly one flag per PR is
  supported by design: multiple flags multiply build and test cost and muddy the signal, and
  single-state pinning is what a flip branch is for.
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
  it to the flag-sensitive tests or one architecture (via the existing scoping labels) to bound the cost.

### Implementation hooks

The trigger and leg-expansion logic is a single shared, CI-framework-agnostic helper in TheRock,
`build_tools/github_actions/flag_both_state.py`, so every orchestrator behaves identically:

- **Shared helper (single source of truth).** `parse_both_state_request(pr_labels, pr_body, known_flags)` detects the `ci:flag-both-state` label, reads the single `Flag: <NAME>` line from the
  PR body, and validates it against the registry; `expand_flag_both_state(legs, flag_name, runtime_flags)` then returns the expanded legs. Build-time: the ON leg appends
  `-DTHEROCK_FLAG_<NAME>=ON` (two builds). Runtime: one build, with two test invocations toggling
  `ROCM_FEATURE_<NAME>` (one build, two test runs). Both are pure and unit-testable, in the style of
  TheRock's existing `tests/configure_multi_arch_ci_test.py` (and the consumer repos'
  `therock_configure_ci_test.py`).
- **Build-time:** the helper's two-leg expansion mirrors the existing `build_variant` (release, asan,
  tsan) pattern in `amdgpu_family_matrix.py`, gated by the label so that it is opt-in per PR and
  never the default across all families.
- **Runtime:** the helper attaches a second test-env entry; the test workflow runs the target once
  per entry. No extra build is required.
- **Wiring (all three orchestrators call the same helper).** TheRock's `configure_multi_arch_ci.py`,
  rocm-libraries' `.github/scripts/therock_configure_ci.py`, and rocm-systems'
  `.github/scripts/therock_configure_ci.py` each call `expand_flag_both_state` on the legs they
  build. Both consumer repos already check out TheRock and invoke its `build_tools/` scripts in CI
  setup, so the helper is reachable without vendoring. Nothing toggles a flag from CI today, so the
  helper plus these three call sites are the new work (see Multi-repo adoption).

### Multi-repo adoption (consumer CI orchestrators)

Both-state CI must behave identically wherever a flag-guarded change is developed. There are three
independent CI orchestrators, and a change in a consumer repo is steered only by that repo's
orchestrator, never by TheRock's:

| Repo           | Orchestrator                                            | Matrix/legs                                                                                     |
| -------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| TheRock        | `build_tools/github_actions/configure_multi_arch_ci.py` | `amdgpu_family_matrix.py`                                                                       |
| rocm-libraries | `.github/scripts/therock_configure_ci.py`               | `.github/scripts/therock_matrix.py` (`collect_projects_to_run`)                                 |
| rocm-systems   | `.github/scripts/therock_configure_ci.py`               | `.github/scripts/therock_matrix.py` project map; legs built inline in `therock_configure_ci.py` |

The shared helper `build_tools/github_actions/flag_both_state.py` is the single source of truth for
the trigger parsing and leg expansion; each orchestrator adapts its own leg shape to and from the helper's
normalized leg dict. The two consumer orchestrators are not structurally identical (see the call-site
note below), but both already have the two building blocks the adoption needs:

- Both already parse PR steering labels in `therock_configure_ci.py` (rocm-libraries handles `test:*`
  and `test_type:*` plus the `skip-therockci` label; rocm-systems handles `ci:skip`), so adding one
  more label parse is routine.
- rocm-libraries already injects `-DTHEROCK_FLAG_*=ON` into a build (the rocKE entry in
  `therock_matrix.py` sets `-DTHEROCK_FLAG_HIPKERNELPROVIDER_ENABLE_ROCKE=ON`), which establishes the
  precedent; rocm-systems has no `THEROCK_FLAG_*` injection today but uses the same
  `-DTHEROCK_ENABLE_*` cmake-option convention, so the mechanics carry over.
- Both repos' CI setup already checks out TheRock and runs its `build_tools/` scripts, so the helper
  is on hand without vendoring.

So a consumer adoption PR is small: import the helper and call it on the legs before the matrix is
emitted (the exact call site differs per repo, as noted below). This keeps TheRock, rocm-libraries,
and rocm-systems consistent on the contract, and prevents the per-repo drift that separate
implementations would create.

The helper is a pure function over a normalized leg representation, so each orchestrator only adapts
its own leg shape to and from the normalized dict:

```python
# build_tools/github_actions/flag_both_state.py  (NEW; TheRock, single source of truth)
#
# A "leg" is one build+test job. Normalized shape:
#   {
#     "cmake_options": ["-DTHEROCK_ENABLE_BLAS=ON", ...],  # list, pre-join
#     "projects_to_test": ["rocblas", ...],
#     "test_env": [ {} ],          # list of env dicts; one test invocation per entry
#     "flag_state_label": "",      # cosmetic, for the job name (e.g. "MYFLAG=ON")
#   }
#
# Trigger (parsed here):
#   Label  ci:flag-both-state   -> the switch: exercise the PR's declared flag in
#                                  both OFF and ON. Applying a label needs
#                                  triage/write access, so it is the permission gate.
#   PR body line  Flag: <NAME>   -> which single flag to exercise.
# Exactly one flag per PR (multiple multiplies cost and muddies the signal).
# Single-state pinning is served by a flip branch, not by this label.

import copy
import re

_FLAG_BOTH_STATE_LABEL = "ci:flag-both-state"

# The flag name is declared on its own line in the PR body and normalized to
# upper-case, because CMake and env flag names are upper-case
# (THEROCK_FLAG_<NAME>, ROCM_FEATURE_<NAME>).
_BODY_FLAG_RE = re.compile(
    r"^\s*Flag:\s*(?P<name>[A-Za-z0-9_]+)\s*$", re.IGNORECASE | re.MULTILINE
)


def parse_both_state_request(pr_labels, pr_body, known_flags):
    """Return the single flag NAME to exercise in both states, or None.

    The generic ci:flag-both-state label is the switch; the PR body names the one
    flag (Flag: <NAME>). The name is validated against the flag registry so a typo
    fails loud instead of silently doing nothing.
    """
    labels = {label.strip().lower() for label in pr_labels}
    if _FLAG_BOTH_STATE_LABEL not in labels:
        return None
    names = [m.group("name").upper() for m in _BODY_FLAG_RE.finditer(pr_body or "")]
    if len(names) != 1:
        raise ValueError(
            f"{_FLAG_BOTH_STATE_LABEL} needs exactly one 'Flag: <NAME>' line in "
            f"the PR body; found {len(names)}."
        )
    name = names[0]
    if name not in known_flags:
        raise ValueError(f"Unknown flag '{name}'; not in the flag registry.")
    return name


def _cmake_without_flag(cmake_options, name):
    """Drop any pre-existing -DTHEROCK_FLAG_<NAME>=... so a leg starts clean."""
    prefix = f"-DTHEROCK_FLAG_{name}="
    return [opt for opt in cmake_options if not opt.startswith(prefix)]


def expand_flag_both_state(legs, flag_name, runtime_flags=frozenset()):
    """Expand legs to exercise flag_name in both OFF and ON. No-op if None.

    Runtime flag (flag_name in runtime_flags): keep ONE build and run the test
      target twice, overlaying ROCM_FEATURE_<NAME>=0 then =1 onto each existing
      test_env entry (an explicit 0 forces OFF even when the runtime default is
      ON, the canary case). Cost = 1 build, 2 test runs.
    Build-time flag: duplicate each leg; the OFF copy strips any pre-existing
      -DTHEROCK_FLAG_<NAME>=..., the ON copy sets it ON, so OFF stays genuinely
      OFF even if the base leg already set the flag. Cost = 2 builds.

    Pure: never mutates `legs`; always returns a new, deep-copied list.
    """
    result = [copy.deepcopy(leg) for leg in legs]
    if not flag_name:
        return result
    name = flag_name.upper()

    if name in runtime_flags:
        for leg in result:
            base_envs = leg.get("test_env") or [{}]
            leg["test_env"] = [
                {**env, f"ROCM_FEATURE_{name}": value}
                for value in ("0", "1")
                for env in base_envs
            ]
        return result

    # Build-time flag: expand each leg into an OFF and an ON leg, each clean.
    expanded = []
    for leg in result:
        off_leg = copy.deepcopy(leg)
        on_leg = copy.deepcopy(leg)
        off_leg["cmake_options"] = _cmake_without_flag(off_leg["cmake_options"], name)
        on_leg["cmake_options"] = _cmake_without_flag(on_leg["cmake_options"], name) + [
            f"-DTHEROCK_FLAG_{name}=ON"
        ]
        off_leg["flag_state_label"] = f"{name}=OFF"
        on_leg["flag_state_label"] = f"{name}=ON"
        expanded += [off_leg, on_leg]
    return expanded
```

Notes on the sketch:

- The helper is deliberately CI-framework-agnostic (no GitHub Actions imports). It is unit-testable
  in isolation, matching the existing `therock_configure_ci_test.py` pattern in both consumer repos.
- `known_flags` is the set of declared flags from the registry (`FLAGS.cmake` plus
  `RUNTIME_FLAGS.cmake` / the shipped `feature_flags.json`); the parser validates the PR-body
  `Flag: <NAME>` against it so a typo fails loud. `runtime_flags` is the runtime subset, so the
  helper knows which flags are one-build/two-test versus two-build. Until the runtime contract lands
  (P2), `runtime_flags` is empty and everything is the build-time path.
- Scoping (one architecture, flag-sensitive tests only) stays the caller's job via the existing
  scoping labels, exactly as the RFC already says for build-time cost control.

A consumer call site is a single added call where legs are built today (for example, rocm-libraries'
`collect_projects_to_run` inside `retrieve_projects`):

```python
# rocm-libraries: .github/scripts/therock_configure_ci.py
# (rocm-systems adapts the same helper call to its own inline leg-building; see below.)
from flag_both_state import (  # reachable via the TheRock checkout
    expand_flag_both_state,
    parse_both_state_request,
)


def retrieve_projects(args):
    ...
    project_to_run = collect_projects_to_run(subtrees)  # EXISTING
    # NEW: pr_labels already parsed above for test:* handling; pr_body is the PR
    # description from the event payload; known/runtime flags come from the registry.
    flag_name = parse_both_state_request(pr_labels, pr_body, load_known_flags())
    project_to_run = expand_flag_both_state(
        project_to_run, flag_name, runtime_flags=load_runtime_flags()
    )
    return project_to_run, test_type
```

The call site differs by repo, because the two consumer orchestrators are not structurally identical:

- **rocm-libraries** builds its legs in `therock_matrix.collect_projects_to_run`, which joins
  `cmake_options` into a space-separated string and `projects_to_test` into a comma-separated string
  at the end. Do the expansion before that join (operate on the lists), then join. This is the call
  site shown above.
- **rocm-systems** has no `collect_projects_to_run`; it builds its legs inline in `retrieve_projects`
  (in `therock_configure_ci.py`), merging `cmake_options` and `projects_to_test` from its own
  `project_map` and joining them there. The adoption is the same one helper call over the same
  normalized leg dict, but placed at that inline site rather than in the matrix module, and it has no
  rocKE/hip-kernel-provider entry to model on.

In both cases the runtime `test_env` entries flow into the matrix so the test workflow runs the target
once per entry. TheRock's own `configure_multi_arch_ci.py` likewise calls the same helper, adapting
its leg shape to and from the normalized dict.

### Promotion job (a scheduled step on the multi-arch CI)

Add the automated promotion job to the multi-arch CI's scheduled run. `configure_multi_arch_ci.py`
already routes the `schedule` event to a nightly tier, so the remaining wiring is a `schedule` trigger
on `multi_arch_ci.yml` (none fires it today) plus the promotion job on that run; cron and period are
to be determined per cadence. On each scheduled fire, it:

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
`FLAGS.cmake` or `configure_multi_arch_ci.py`. P1 adds CODEOWNERS entries naming a
release-manager group (for example, `@ROCm/TheRock-release-managers`) for the flag registries
(`FLAGS.cmake`, `RUNTIME_FLAGS.cmake`) and the multi-arch CI that hosts the scheduled promotion step
(`configure_multi_arch_ci.py`, `.github/workflows/multi_arch_ci.yml`). The promotion PR must be approved by someone other than the flip author.

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

| Case                                          | Action                                                                                                                                                                                                                                                                        | Cost                                                                        |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Runtime flag bad: environment kill switch** | `ROCM_FEATURE_<NAME>=0` (alias `HIPDNN_FEATURE_<NAME>=0`): a true per-process and per-host live control, with no package change. This is the minutes-scale revert, but it is per-host (ROCm has no fleet config push, so it does not reach the fleet remotely).               | Minutes, per host. Primary argument for preferring runtime.                 |
| **Runtime flag bad: channel default**         | Edit the installed `feature_flags.json` to default the flag OFF. This is the channel default for the next package only; already-installed users need a package respin and reinstall. Land the disable change first (proving nothing breaks), then change the shipped default. | Not minutes; a package respin and reinstall for existing installs.          |
| **Build-time flag bad**                       | Revert the flip, rebuild, and re-promote next cycle.                                                                                                                                                                                                                          | One cycle plus a full rebuild. Argument against overusing build-time flags. |
| **Failed promotion (post-merge regression)**  | Release-manager-gated revert of the offending mainline-default diff on `main`; re-promote next cycle once fixed.                                                                                                                                                              | One cycle; isolates blame to the promoted flip.                             |
| **Bad flip caught on canary**                 | Drop the offending default-flip commit from `canary` (or exclude it from the next batch). The frequent-rebase and batch model means the bad flip is simply not carried forward; no surgical in-place revert is needed.                                                        | Bounded; the bad flip simply misses the cut.                                |
| **Canary red mid-cycle**                      | Promotion does not happen; the flip or flips soak another cycle. There is no mainline impact (the mainline default is OFF).                                                                                                                                                   | Zero mainline impact.                                                       |

Note the fleet-coverage gap: ROCm has no server-side config push, so the environment kill switch is
per-host and the JSON default reaches only the next package. A full-coverage revert requires the
shipped default change, which is exactly why landing the disable change first matters.

The release-manager role comprises the promotion job and the human owner who approves the
automated promotion PR (someone other than the flip author); the bar tightens as flips approach
`default-on`.

## Team Workflow

Ownership: the team owns the flag, the gated behavior, and retirement; TheRock owns the registry,
manifest, CI mechanism, and promotion train; Quartz (RFC0011) owns the green signal.

1. **Create the flag in TheRock.** Choose the kind via the decision rule (prefer runtime); declare it in the runtime registry (`therock_declare_runtime_flag`) or `FLAGS.cmake` with full metadata (owner, created, expires, stage `in-development`, issue).
1. **Guard the code in your project** at one entry point (runtime: a single check via the documented reader contract, that is, a vendored copy of the example `rocm_feature_flags.h` header or your own small consumer, for example hipDNN's `validateBeforeAdding`; build-time: a `#define` or CMake-gated branch).
1. **Land behind the flag, default-off, on trunk** (and land the gated submodule code OFF on `main` first). Mainline stays OFF; nothing changes for users yet.
1. **Validate both states in CI** (team-owned both-state CI): apply the `ci:flag-both-state` label and add a `Flag: <NAME>` line to the PR description (or use a dispatch input), or develop on a flip branch. Runtime takes one build and two test runs; build-time takes two builds.
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
1. **Declare the flag, default OFF,** with metadata: `OWNER`, `CREATED`, `EXPIRES`,
   `STAGE: in-development`, `ISSUE`. Declare it wherever is closer to your work (see "Where flags are
   declared"); both are valid and TheRock aggregates the inventory either way:
   - runtime: `therock_declare_runtime_flag(NAME <NAME> DEFAULT_VALUE OFF …)` in `RUNTIME_FLAGS.cmake`
   - build-time: `therock_declare_flag(NAME <NAME> TYPE BOOL DEFAULT_VALUE OFF …)` in `FLAGS.cmake`
   - This PR can land immediately; it is a priority review and is not release-manager-gated (see
     "Adding a flag does not block development").
1. **If you declared in TheRock, advance the TheRock version your branch resolves against** so the
   new flag is visible to your repository. In rocm-libraries this is a resync with `develop`, which
   moves the merge-base the TheRock ref is pinned to
   ([PR #9602](https://github.com/ROCm/rocm-libraries/pull/9602)); you do not wait on the 12-hour
   bump cron. If you declared in your own super-repository, skip this step entirely.
1. **Guard your code at exactly one entry point.** Runtime: a single check via a vendored copy of the
   example `rocm_feature_flags.h` (or your own reader or existing environment-flag system), for
   example hipDNN's `validateBeforeAdding`. Build-time: resolve the flag once with
   `rocm_resolve_build_flag()` (from your super-repository's copy of `ROCMBuildFlags.cmake`),
   configure it into a private generated header, and branch on `ROCM_BUILD_FLAG(<NAME>)`; see
   `base/aux-overlay` for the worked example. Do not sprinkle conditionals.
1. **Land your feature default-OFF on trunk.** Mainline stays OFF; nothing changes for users. You can
   now iterate behind the flag.

**Completion criteria:** the flag exists (default OFF) and is visible to your repository, your new
code is guarded at one entry point, and trunk is green with the feature OFF.

### Playbook B: testing a new feature in both states

**Goal:** prove both the ON and OFF paths work, so that neither becomes stale and promotion later is
safe.

1. **Choose a trigger** (either or both):
   - **Label plus PR-body flag, or dispatch:** add the `ci:flag-both-state` PR label and a
     `Flag: <NAME>` line to the PR description (or set the `workflow_dispatch` input).
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
1. **Automated promotion opens the mainline PR** on the scheduled fire (the scheduled multi-arch run) once
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
flags alike: once the gated code is gone, nothing references the flag, so removing the declaration
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

### Adoption exemplar: enabling both-state CI in rocm-libraries

A team guards a new build-time flag `MY_FEATURE` (declared default-OFF in TheRock `FLAGS.cmake`) and
wants both states exercised on its rocm-libraries PR. Two things happen, both small:

1. The developer adds the `ci:flag-both-state` label and a `Flag: MY_FEATURE` line to the PR
   description.

1. rocm-libraries' `therock_configure_ci.py` already parses PR labels and already builds legs via
   `collect_projects_to_run`. The one-time adoption change is a single call to the shared helper
   before the matrix is emitted:

   ```python
   project_to_run = collect_projects_to_run(subtrees)  # existing
   flag_name = parse_both_state_request(pr_labels, pr_body, load_known_flags())  # new
   project_to_run = expand_flag_both_state(
       project_to_run, flag_name, runtime_flags=load_runtime_flags()  # new: one call
   )
   ```

For `MY_FEATURE` (build-time), the affected project's job fans out into an OFF leg and an ON leg
(`-DTHEROCK_FLAG_MY_FEATURE=ON`), each built and tested. If `MY_FEATURE` were a runtime flag, the job
would build once and run the affected tests twice with `ROCM_FEATURE_MY_FEATURE=0` then `=1`.
rocm-systems consumes the same helper for the same behavior, though its orchestrator differs
structurally (it builds legs inline in `retrieve_projects` rather than in a shared
`collect_projects_to_run`), so its call site is a different line while the contract and result are
identical. No per-repo reimplementation, no drift.

## Alternatives Considered

### Flag kind default

| Option                       | Pros                                                                              | Cons                                                                                      | Verdict                          |
| ---------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------- |
| **Runtime-default (chosen)** | One binary for all states; immediate field revert; no combinatorial build matrix. | Disabled code still ships in the binary (size); needs a new runtime mechanism.            | Chosen.                          |
| Build-time-default           | Reuses the existing `FLAGS.cmake` only.                                           | Rebuild required to change; combinatorial CI; no field kill switch; binary fragmentation. | Rejected; only for forced cases. |

### Promotion mechanism

| Option                                                                   | Pros                                                                  | Cons                                                                                                                   | Verdict   |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------- |
| **Fixed, automated canary soak train (chosen; period to be determined)** | Predictable; one soak cycle before swap; uses the canary soak signal. | New canary branch, scheduled job, and a frequent rebase of canary onto main (only difference being the flag defaults). | Chosen.   |
| Promote on every green nightly                                           | Faster.                                                               | No soak window; high-churn; no human release-manager gate; the nightly is `main`-only (no canary nightly).             | Rejected. |
| Manual ad-hoc promotion                                                  | No new infrastructure.                                                | Unpredictable; no soak guarantee; debt accrues.                                                                        | Rejected. |

### Where flags live

| Option                                                                       | Pros                                                                                                                                             | Cons                                                                                                                     | Verdict   |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | --------- |
| **TheRock owns the mechanism; a flag may originate in either repo (chosen)** | One mechanism, one aggregated inventory, uniform metadata and CI, and no cross-repo wait to start work. Matches PR #6984's dual-mode resolution. | Inventory aggregation must reach into the consuming super-repos, not just read `FLAGS.cmake`.                            | Chosen.   |
| TheRock is the mandatory origination point                                   | Simplest possible inventory; one file to read.                                                                                                   | Interlocks the repos at the worst moment: a team cannot guard code until a declaration lands *and* reaches their branch. | Rejected. |
| Per-project registries with no central mechanism                             | Local autonomy.                                                                                                                                  | No global inventory; duplicated mechanism; no central audit.                                                             | Rejected. |

### Both-state CI (team-owned; decoupled from canary)

| Option                                                                                               | Pros                                                             | Cons                                                                                                              | Verdict                         |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| **Label or flip-branch both-state, runtime equals one build and two test runs (chosen for runtime)** | No rebuild (binary-neutral); inexpensive; team-owned, on any PR. | Needs the label or dispatch wired through `configure_multi_arch_ci.py` and the consumer orchestrators (new work). | Chosen.                         |
| Label or flip-branch both-state, build-time equals two builds                                        | Necessary for build-time flags.                                  | Two builds; amortize with `prebuilt_stages` and `baseline_run_id`, and scope via the existing labels.             | Chosen (scoped) for build-time. |
| Rely on canary for both-state coverage                                                               | No new mechanism.                                                | The wrong model: canary only soaks the to-be-promoted default; it does not give a team ON-and-OFF coverage.       | Rejected.                       |

### Canary flip carrier

| Option                                                                        | Pros                                                                                            | Cons                                                                                               | Verdict   |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------- |
| **Committed `DEFAULT_VALUE` or registry-default change on `canary` (chosen)** | Observable and reviewable on both `canary` and `main`; the promotion diff is the same artifact. | None significant.                                                                                  | Chosen.   |
| TheRock's existing `BRANCH_FLAGS.cmake` branch override                       | Reuses an existing mechanism.                                                                   | Gitignored and not observable on `main`; a flip would be invisible except as a configure-log line. | Rejected. |

## Implementation Phases

| Phase                                                                | Deliverables                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **P0: land the typed provider protocol**                             | PR #6984 (typed `BOOL`/`INTEGER` declarations, `rocm_build_flags_state.cmake`, `ROCMBuildFlags.cmake` plus `rocm_resolve_build_flag()`, the `ROCM_BUILD_FLAG(name)` accessor, and the `base/aux-overlay` conformance canaries). Not owned by this RFC; everything below assumes it. Follow-on within P0: allow a flag to originate in a consuming super-repository as well as in TheRock.                                                                                                                                                                                                                                                                                                                                                                                                              |
| **P1: metadata, hygiene, and RBAC**                                  | Extend `therock_declare_flag` with OWNER, CREATED, EXPIRES, and STAGE (alongside the `TYPE` and `VALID_VALUES` from P0); surface them in `therock_report_flags()` and the manifest; add an expiry warning and non-mainline `ISSUE` enforcement; aggregate flags declared in consuming super-repositories into the same inventory, report, and manifest so origination location does not create a blind spot; fold `THEROCK_FLAG_INCLUDE_PROFILER` into the registry; add `.github/CODEOWNERS` entries (release-manager group) for `FLAGS.cmake`, `RUNTIME_FLAGS.cmake`, and the multi-arch CI (`configure_multi_arch_ci.py`, `.github/workflows/multi_arch_ci.yml`); update `docs/development/flags.md`.                                                                                               |
| **P2: generic runtime contract**                                     | `RUNTIME_FLAGS.cmake` plus `therock_declare_runtime_flag` plus `therock_finalize_runtime_flags()`; emit the shared `share/therock/feature_flags.json` from the `base/aux-overlay` step alongside the manifest (it ships automatically via aux-overlay's existing `**/*` catch-all, with no toml change); add `runtime_flags` to the manifest; document the reader contract (location, `dladdr` discovery, `ROCM_FEATURE_*` precedence) and publish the example reference `rocm_feature_flags.h` (copied-in, not shipped or linked) with a standalone-build fallback note; the `rocm-feature-flags --list` helper; wire the first instantiation (hipDNN at `validateBeforeAdding`, using its own consumer).                                                                                             |
| **P3: canary branch, team-owned both-state CI, and canary currency** | Create `canary` (soak-only) and add it to the `on.push.branches` list in `.github/workflows/multi_arch_ci.yml` so that CI runs on it; document the soak convention (the current batch flipped ON, else matching main); add the shared helper `build_tools/github_actions/flag_both_state.py` (generic-label plus PR-body flag parsing, registry-validated, plus OFF/ON leg expansion; runtime equals one build and two test runs, build-time equals two builds, scoped); wire it into TheRock's `configure_multi_arch_ci.py`; land thin call-site adoption PRs in rocm-libraries and rocm-systems (`.github/scripts/therock_configure_ci.py` plus `therock_matrix.py`) that call the same helper; add the scheduled frequent rebase of `canary` onto `main` (only difference being the flag defaults). |
| **P4: automated promotion job**                                      | Add a `schedule` trigger to `multi_arch_ci.yml` (routed to the nightly tier `configure_multi_arch_ci.py` already supports), then attach the promotion job to that scheduled run (cron and period to be determined): the canary soak-signal gate plus the promotion PR (release-manager-merged, not auto-merged) plus the flag-debt audit (the flag's gated code is already on `main` by the rebase model).                                                                                                                                                                                                                                                                                                                                                                                             |
| **P5: canary soak signal (optional deepening)**                      | A new per-branch canary validation job (`workflow_dispatch` or matrix checking out canary) plus a Quartz-emitted per-branch `latest_good@canary.json`, if the builds-and-tests-green minimum is judged insufficient.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **P6: build-time both-state build dimension (optional and scoped)**  | A build dimension in `amdgpu_family_matrix.py` and `configure_multi_arch_ci.py` for build-time flags plus `prebuilt_stages` amortization, beyond the label wiring in P3.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **P7: adopt and retire**                                             | Migrate the first real flags (SDPA v2, the new backend); run the first full train; remove the first `default-on` flag to validate retirement.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

## Decisions and Open Questions

### Resolved decisions

1. **Canary branch is a plain `canary` branch,** with CI configured to run on it via the `on.push.branches` list in `.github/workflows/multi_arch_ci.yml` (presubmit and postsubmit). It is soak-only, not "everything on."
1. **There is no shared runtime library; the header is an example, not a component.** TheRock provides a documented reader contract (location, `dladdr` discovery, `ROCM_FEATURE_<NAME>` precedence) plus an example reference `rocm_feature_flags.h` that implements it. Libraries copy the example header into their own tree (recommended, most portable) or reimplement the contract against their existing environment-flag system; there is no shipped, linked, or auto-included dependency. Standalone builds vendor their own copy and fall back to compile-time defaults.
1. **One reviewed batch per cycle:** small and explicitly reviewed, so that the result remains attributable; no split-attribution without the per-flip machinery.
1. **The global `share/therock/feature_flags.json` ships automatically** via `base/aux-overlay`'s existing `**/*` catch-all (the same path as the manifest), with no `.toml` change. Only optional, opt-in per-library override files add their own `include`.
1. **Promotion cadence is fixed and automated:** the job opens the PR, a release manager merges it, and the frequent rebase keeps canary current with no manual reset.
1. **Both-state CI is team-owned and decoupled from canary,** triggered via the `ci:flag-both-state` label (the flag named in the PR description) or dispatch and/or a flip branch: runtime equals one build and two test runs; build-time equals two builds.
1. **The build-time mechanism is PR #6984's typed provider protocol, adopted as-is.** This RFC does not define a competing declaration or consumption surface. It adds lifecycle metadata, hygiene, canary, promotion, and both-state CI on top, and adopts the protocol's fail-closed posture and distribution boundary.
1. **A flag may originate in TheRock or in a consuming super-repository.** TheRock owns the mechanism and the aggregated inventory, not the mandatory birthplace. Restricting origination to TheRock would interlock the repositories at the point where a team is trying to start work.
1. **Runtime flags start `BOOL`-only and may gain `INTEGER` later.** Build-time flags are typed `BOOL` or `INTEGER` (PR #6984). Starting the runtime contract at `BOOL` keeps `ROCM_FEATURE_<NAME>=0|1` and the shipped JSON simple, and no runtime use case yet needs a non-BOOL value. Nothing precludes extending it when one does: neither the JSON, the manifest `runtime_flags` block, nor the precedence order depends on the value being boolean.

### Open for reviewer input

1. **Canary soak test scope:** which tests run on canary, whether and what additional testing is stacked on top, and the soak-cycle length. The minimum is canary builds and tests green; this can optionally be deepened with the per-branch `latest_good@canary.json` (P5). The exact test set, any additional stacked testing, and the cycle length are open.
1. **Promotion cadence period:** weekly, bi-weekly, or monthly (and the exact schedule on the multi-arch CI). Only "fixed and automated" is settled; the period and schedule are open.
1. **How the inventory aggregates flags declared outside TheRock.** Because origination is allowed in either place, the hygiene audit and the manifest need to see super-repo-declared flags. Options: TheRock scans the checked-out consuming super-repositories at configure time; each super-repo emits its own declarations into the provider handshake; or the audit runs per-repo and reports into a shared place. Preference not settled; PR #6984's follow-on work will constrain the answer.
1. **When and where to build the per-PR both-state label mechanism:** wire the `ci:flag-both-state` label (the flag named in the PR body) via the shared `flag_both_state.py` helper now, or rely initially on flip branches and add the label later. Flip branches need no new CI wiring; the label is the more ergonomic per-PR path. Because the helper is repo-agnostic and both consumer repos already parse steering labels (rocm-libraries `test:*`, rocm-systems `ci:skip`) and rocm-libraries already injects `-DTHEROCK_FLAG_*=ON` (the rocKE precedent), the build-time label path is adoptable in both today, before the runtime contract (P2) lands. Open: whether to land all three call sites in P3 together, or TheRock first with consumers fast-following.

## Summary

This RFC turns feature flags into a disciplined, auditable lifecycle in TheRock and its
libraries.

**What it adds.** It extends the existing `FLAGS.cmake` build-flag registry with mandatory
owner, expiry, and stage metadata, and adds a generic runtime-flag contract: a shipped
`share/therock/feature_flags.json`, a documented reader (`dladdr` discovery,
`ROCM_FEATURE_<NAME>` precedence), and an example `rocm_feature_flags.h` that implements it.
Libraries copy the header or reimplement the contract; there is no shipped, linked, or
auto-included dependency, and standalone builds fall back to compile-time defaults. hipDNN is
the first adopter, using its own `EngineOverrideConfig` and `validateBeforeAdding` path.

**What it does not add.** The build-time declaration and consumption mechanism is PR #6984's typed
provider protocol (`TYPE BOOL|INTEGER`, `rocm_build_flags_state.cmake`, `ROCMBuildFlags.cmake` plus
`rocm_resolve_build_flag()`, and the fail-closed `ROCM_BUILD_FLAG(name)` accessor), adopted as-is.
This RFC is the lifecycle layer on top of it, and it inherits that protocol's two governing
properties: fail closed, and never install a build-time input. A flag may originate in TheRock or in
a consuming super-repository; TheRock owns the mechanism and the aggregated inventory, not the
birthplace.

**How a flag is promoted.** A fixed, automated canary-to-mainline train (period to be
determined) runs as a scheduled step on the multi-arch CI, on a `schedule` trigger added to
`multi_arch_ci.yml` and routed to the nightly tier the orchestrator already supports. Canary is a
soak-and-staging branch: the team flips the candidate default(s) to ON and soaks for one cycle
(minimum signal: canary builds and tests green; RFC0011's `latest_good.json` remains the `main` and
nightly signal). On a green soak the automated job opens a promotion PR that a release manager, not
the flip author, merges. Frequently rebasing canary onto `main` keeps it current: the gated code is always on
`main` first, the only divergence is the soaking defaults, and a promoted flip stops being a
divergence with no manual reset. The default is one reviewed batch per cycle, and a bad flip is
simply dropped.

**The decision rule.** Runtime by default; build-time only when the change alters artifacts,
topology, or ABI (Multi-Arch Packaging is the exemplar). Both states are exercised by a
team-owned both-state CI mechanism, decoupled from canary and triggered by the `ci:flag-both-state`
label (the flag named in the PR description), a dispatch input, or a flip branch: one build and two test runs for runtime flags, two
builds for build-time (the main reason to prefer runtime).

**Backout.** `ROCM_FEATURE_<NAME>=0` is the minutes-scale, per-host kill switch (ROCm has no
fleet push); editing the installed JSON sets the next-package default (a respin for existing
installs); build-time backout takes one cycle.

### Quick reference

Full detail in Maintainer Playbooks.

| To do this                         | Do the following                                                                                                                                                                                                                   | Success criterion                                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Start a new feature**            | Pick runtime (default) or build-time; add a default-OFF flag in TheRock or in your own super-repository (lands quickly); guard your code at one check; ship it OFF on trunk.                                                       | The flag exists (OFF), the code is guarded at one point, and trunk is green with the feature OFF. |
| **Test it both ways**              | Add the `ci:flag-both-state` label and a `Flag: <NAME>` line to the PR description (or use a flip branch). Runtime: one build, tests run twice with `ROCM_FEATURE_<NAME>=0` then `=1`. Build-time: build OFF and ON, testing each. | The affected tests are green in both states on CI.                                                |
| **Turn it on for everyone**        | Open a PR flipping the default ON into `canary` (`STAGE: canary`); let it soak one cycle green; the automated job opens the `main` PR; a release manager (not you) merges it.                                                      | The default-ON change is merged on `main` by a release manager, and the nightly is green.         |
| **Turn it off quickly (it broke)** | Set `ROCM_FEATURE_<NAME>=0` on the affected host (minutes, no rebuild). For everyone: land a PR setting the default back to OFF; existing installs need a respin.                                                                  | The bad path no longer runs.                                                                      |
| **Clean up afterward**             | After ~one cycle default-on, drop both-state CI for the flag, delete the dead OFF code from your library, then remove the flag declaration from TheRock. (A `long-lived` flag is the exception; it is never retired.)              | The flag is gone from the registry, the manifest, and the code.                                   |

The core sequence: add a runtime flag, default off; guard one entry point; soak on canary;
promote via the release-manager-merged PR; revert with `ROCM_FEATURE_<NAME>=0`.

## References

- RFC0008: Multi-Architecture Packaging with Kpack (`docs/rfcs/RFC0008-Multi-Arch-Packaging.md`); build-time exemplar.
- RFC0011: Quartz, Central CI/CD Data Hub (`latest_good.json` green signal).
- TheRock: `FLAGS.cmake`, `cmake/therock_flag_utils.cmake`, `BRANCH_FLAGS.cmake`, `build_tools/generate_therock_manifest.py`, `docs/development/flags.md`.
- [PR #6984](https://github.com/ROCm/TheRock/pull/6984), "add typed ROCm build flag provider protocol"; the build-time mechanism this RFC layers on top of: `cmake/ROCMBuildFlags.cmake`, `rocm_resolve_build_flag()`, the generated `rocm_build_flags_state.cmake`, the `ROCM_BUILD_FLAG(name)` accessor, `base/aux-overlay` conformance canaries, and the distribution boundary.
- [rocm-libraries PR #9602](https://github.com/ROCm/rocm-libraries/pull/9602), merge-base-time TheRock ref resolution (`.github/scripts/resolve_therock_ref.py`); why advancing the TheRock version a branch resolves against is self-service rather than a wait on the bump cron.
- TheRock CI and cross-repo: `build_tools/github_actions/configure_multi_arch_ci.py`, `build_tools/github_actions/configure_ci_path_filters.py`, `build_tools/github_actions/amdgpu_family_matrix.py`, the proposed `build_tools/github_actions/flag_both_state.py` (shared both-state helper), `.github/workflows/multi_arch_ci.yml`, `.github/workflows/multi_arch_ci_asan.yml`, `.github/workflows/bump_submodules.yml`, `build_tools/github_actions/bump_automation.py`, `.github/CODEOWNERS`, `docs/development/ci_behavior_manipulation.md`, `docs/packaging/versioning.md`.
- Consumer CI orchestrators (both-state adoption call sites): `rocm-libraries/.github/scripts/therock_configure_ci.py` and `therock_matrix.py`; `rocm-systems/.github/scripts/therock_configure_ci.py` and `therock_matrix.py`.
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
- **2026-06-24**: Editorial revision. Consolidated the separate Summary and Quick-Start Summary into a single Summary (moved before References); tightened the Overview, Glossary, Goals, and other sections for concision and readability; added the `long-lived` value to the `STAGE` enum and Stages table for consistency, and exempted `long-lived` from the `therock_finalize_flags()` `ISSUE`-on-`main` enforcement so a permanent kill switch need not carry a perpetual tracking issue. Reordered the flag-retirement steps (Playbook D, Team Workflow step 7, Summary "Clean up" row) into dependency order (CI off, then dead code, then declaration), which removes the consumer before the declaration it depends on and is uniformly safe for runtime and build-time flags (once the gated code is gone, removing the declaration can neither revert the live build nor break a consumer).
- **2026-07-27**: Both-state CI multi-repo amendment (Tony Davis). Makes team-owned both-state CI a repo-agnostic contract with one shared TheRock helper (`build_tools/github_actions/flag_both_state.py`) that TheRock, rocm-libraries, and rocm-systems all call, so the mechanism reaches the consumer repos where flag-guarded work actually lives rather than TheRock's CI alone (Goal 5, Implementation hooks, the new Multi-repo adoption subsection with the helper sketch and consumer call site, a rocm-libraries adoption worked example, P3, and Open Question 3). Also reconciles stale CI filenames: `configure_ci.py` was deleted in favor of `multi_arch_ci.yml` (PR #5794), so references now point at `configure_multi_arch_ci.py`, and the canary/long-lived-branch coverage is described by the `on.push.branches` list in `.github/workflows/multi_arch_ci.yml` (the `long_lived_full_match` symbol no longer exists in the tree).
- **2026-07-27**: Correctness and editorial pass on the multi-repo amendment. (1) Scheduled-CI grounding: no workflow fires `configure_multi_arch_ci.py` on a schedule today (the only scheduled workflows are `bump_submodules.yml` and `gitleaks_main.yml`; `multi_arch_ci_asan.yml` carries no cron), so the promotion job is now framed as requiring a new `schedule` trigger on `multi_arch_ci.yml` routed to the nightly tier the orchestrator already supports, rather than attaching to an already-scheduled run (fact table, cadence, Promotion job, P4, Summary). (2) Consumer-orchestrator symmetry: corrected the overstated claim that rocm-systems mirrors rocm-libraries; rocm-systems builds legs inline in `retrieve_projects` with no `collect_projects_to_run` and no rocKE/`THEROCK_FLAG_*` precedent, so its call site differs while consuming the same helper (Multi-repo adoption bullets and table, the call-site note, the adoption exemplar, Open Question 3). Fixed the steering-label vocabulary (rocm-libraries `test:*`/`test_type:*` plus `skip-therockci`; rocm-systems `ci:skip`) and pointed the helper's unit-test analogue at TheRock's `tests/configure_multi_arch_ci_test.py`. (3) Clarified that flags are ephemeral by design and retired after promotion, not a permanent or supported configuration surface, with the `long-lived` kill switch as the sole exception (Overview callout, Glossary, a new Non-Goal). (4) Editorial: removed em-dashes throughout for a more conventional register; no structural changes.
- **2026-07-28**: Both-state trigger ergonomics (Tony Davis). Replaces the per-flag `flag:<NAME>:both` label with a single generic `ci:flag-both-state` label as the switch, plus a `Flag: <NAME>` line in the PR description naming the one flag, validated against the flag registry. This removes per-flag label maintenance (GitHub can only apply labels that already exist, so a brand-new flag's label would not exist until someone created it per repo), keeps the permission gate on the label (applying a label needs triage/write access while a PR body does not), and matches the repo `ci:*` label convention. The shared helper gains `parse_both_state_request(pr_labels, pr_body, known_flags)`, and `expand_flag_both_state` now takes a single validated `flag_name`; exactly one flag per PR is supported by design, with single-state pinning left to flip branches (Glossary, Goal 5, Trigger, Implementation hooks and helper sketch, consumer call site, adoption exemplar, Playbooks A and D, P3, Open Question 3, Summary, and Quick reference).
- **2026-08-04**: Align with the typed build flag provider protocol, [PR #6984](https://github.com/ROCm/TheRock/pull/6984) (Tony Davis). PR #6984 lands the build-time declaration and consumption mechanism this RFC had only sketched, so the RFC now defers to it rather than proposing a parallel surface, and is positioned as the lifecycle layer above it. Four substantive changes. (1) **Origination is no longer TheRock-only.** "All flags start in TheRock as the single source of truth" is replaced by "TheRock owns the mechanism and the aggregated inventory; a flag may originate in TheRock or in a consuming super-repository," following the author's stated intent on #6984 to let a flag originate in either place "so we aren't so strictly interlocked." This removes the declare-then-wait round trip that blocked a team from guarding code until a TheRock declaration both landed and reached their branch; where a TheRock declaration is still involved, advancing the pinned TheRock ref is self-service via a resync with `develop` ([rocm-libraries PR #9602](https://github.com/ROCm/rocm-libraries/pull/9602)) rather than a wait on the 12-hour bump cron (Where flags are declared, "Adding a flag does not block development", the declaration diagram, Playbook A steps 2 and 3, the Where-flags-live alternatives table, P1 inventory aggregation, a new resolved decision, a new open question, Summary, Quick reference). (2) **Flags are typed, not boolean.** The Glossary, taxonomy table, and `therock_declare_flag` example now carry `TYPE` (`BOOL` default, `INTEGER` with `VALID_VALUES`), with a note on what a multi-valued flag means for the promotion train and for both-state CI (the current default versus the candidate value, not the full cross-product). Runtime flags stay BOOL-only for now, deliberately without precluding typing later (new open question). (3) **The distribution boundary is adopted, and the runtime exception is justified explicitly.** #6984 forbids installing build-time inputs; this RFC ships an installed `feature_flags.json`. A new "Distribution boundary" subsection states the rule for build-time flags unchanged and explains why runtime state is on the other side of the line by construction (a value unreadable after install is not a runtime flag), so the two read as consistent rather than contradictory. (4) **Fail-closed on flag names.** The runtime reader keeps its silent fallback on a missing or unparseable file (a standalone build has none), but a misspelled or unregistered flag *name* must now be detectable, mirroring the intent of #6984's function-like `ROCM_BUILD_FLAG(name)` accessor; `rocm-feature-flags --list` reports names absent from the manifest. Also adds a P0 phase for #6984 itself, two resolved decisions, and reference entries for #6984 and #9602.
- **2026-08-05**: Review feedback on the #6984 alignment (Tony Davis). (1) The Distribution boundary subsection separated build-flag *inputs* from the build-flag *record*, which the earlier two-column framing blurred: the provider state file and helper are never installed, but the manifest `flags` block continues to ship the as-built flag state for provenance, triage, and the flag-debt audit, and is a record rather than an input. The table is now keyed by artifact, with the manifest as its own row. (2) The typed-runtime-flags question moved from open to resolved: runtime flags start `BOOL`-only and may gain `INTEGER` later, with nothing in the JSON, the manifest `runtime_flags` block, or the precedence order precluding it.
