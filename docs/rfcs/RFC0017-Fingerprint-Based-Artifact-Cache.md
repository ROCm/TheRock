---
author: Ameya Mallya (ammallya), Erman Gurses (erman-gurses)
created: 2026-10-26
modified: 2026-11-23
status: draft
discussion: https://github.com/ROCm/TheRock/issues/8651
---

# Fingerprint-Based CI Artifact Cache

## Problem

Today reuse walks recent workflow runs to select a single baseline run, then decides
per build stage using change-impact analysis plus commit-compatibility, job-health,
and artifact-availability checks (`baseline_runs.py`, `stage_reuse_decision.py`).
Fingerprints are never consulted. Reuse is therefore limited to whole stages from one
run, cannot combine compatible artifacts from different runs, and costs GitHub API
queries for runs and jobs plus artifact listings on every selection. Release
branches disable reuse
([#8454](https://github.com/ROCm/TheRock/issues/8454)).

## Proposed solution

Select reusable artifacts by the fingerprints computed during build configuration.
CI queries the artifact cache index, verifies and restores usable matches, and
builds artifacts without usable matches. Compatible artifacts may come from
different producer runs.

Trusted producer runs upload archives, checksums, and fingerprints, then publish
entries to the remote cache database. The database records artifact locations;
archives remain in the existing S3 buckets.

## Goals

- Preserve reuse when unrelated sources or repository commits change.
- Rebuild artifacts when relevant sources, configuration, selected files, or
  dependency fingerprints change.
- Support partial reuse within stages and across the build graph.
- Once all projects in a source set are enabled, a PR that changes only ROCm
  Libraries hits the cache for every ROCm Systems project.
- Fall back to building when cached artifacts cannot be restored safely.
- Keep test selection independent of reuse: a hit does not suppress tests selected
  by [RFC0013](./RFC0013-Consumer-Based-Test-Selection.md).

## Non-goals

- Replacing `ccache` or `sccache`, which cache individual compilation operations.
- Introducing new artifact object storage.
- Adding automatic remote-cache access to ordinary CMake invocations.
- Requiring byte-identical outputs for the explicitly accepted metadata
  differences described below.

## Overall architecture

The build system defines compatibility, the remote cache database finds candidate
artifacts, and CI verifies and restores them.

![Fingerprint-based artifact cache flow](../assets/RFC0017-cache-flow.svg)

### 1. Build system: artifact fingerprints

TheRock computes fingerprints during CMake configuration. Input changes require
reconfiguration before building or querying the cache.

| Term                   | Meaning                                                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Source identity        | Identity of a selected source directory, using its applicable `.smrev` marker or Git tree. Additional source inputs are declared explicitly. |
| Subproject fingerprint | Combines source identity, selected configuration values and file hashes, and dependency fingerprints.                                        |
| Artifact fingerprint   | Accounts for the artifact's contributing subprojects and is used for artifact lookup.                                                        |
| Archive checksum       | Verifies the bytes of a particular archive; it does not establish build compatibility.                                                       |

Equal artifact fingerprints should identify interchangeable artifacts, ideally
with identical bytes. Validation must detect both false hits (changed relevant
inputs retain a fingerprint) and false misses (equivalent inputs produce different
fingerprints). Invalid fingerprints must not authorize reuse, although local
source builds may continue.

A **restore group** contains the components required to restore one artifact name
and target. For example, restoring the complete generic `amd-llvm` bundle requires
its `dbg`, `dev`, `doc`, `lib`, and `run` components, containing outputs from
`amd-llvm`, `amd-comgr`, and `hipcc`. For generic reuse, those components must share
one fingerprint and be restored together. This differs from `artifact_group` in
`BUILD_TOPOLOGY.toml`, which organizes artifacts for stage planning.

Dependencies follow declared inputs, including inputs across repository
boundaries. LLVM's additional inputs include the SPIR-V translator and, on
applicable platforms, HSA headers. See
[#9907](https://github.com/ROCm/TheRock/pull/9907) for source scoping and additional
inputs.

#### Coverage and stability

Fingerprint correctness, stability, and robustness are tracked in
[#9525](https://github.com/ROCm/TheRock/issues/9525).

Generic artifacts provide the initial implementation path. Device-specific and
split artifacts need fingerprint generation, target identification, and validated
restoration rules before reuse is enabled. This includes outputs described in
[RFC0008](./RFC0008-Multi-Arch-Packaging.md).

#### Accepted metadata differences

Non-release CI may reuse unchanged HIPCC inputs with the producer's original
changelog timestamp. Relevant HIPCC source changes must invalidate the applicable
fingerprints. Making changelog bytes reproducible across fresh builds is optional
and non-blocking under this policy; see
[#10011](https://github.com/ROCm/TheRock/issues/10011). Other metadata differences
require explicit justification. Release builds do not consume cached artifacts.

### 2. CI system: cache consumption and publication

#### Read and write policy

Non-release CI may read artifacts enabled for reuse. Only trusted producer runs
may publish to the shared cache. CI identity and permissions must prevent
untrusted jobs from modifying either the database or trusted artifact objects.
An unrelated test failure does not by itself make a successfully produced artifact
unusable; complete matching outputs and trusted provenance are required.

Release builds build from source. Artifact reuse does not change which tests run.
The cache does not need per-entry trust or test-validation states: only complete
entries from authorized producers become visible to readers.

#### Consumer flow

The stage job that uses restored outputs also computes fingerprints and builds
misses. It must use the same stage configuration procedure as producer builds,
such as `configure_stage.py`, with equivalent fingerprint-relevant settings.
Different configuration modes can produce different fingerprints from the same
sources.

1. Configure the prepared sources and save the expected artifact fingerprints.
1. Query the artifact cache index in a batch and produce a hit/miss plan.
1. Populate the current run's artifact location, checking required components,
   fingerprints, and archive checksums.
1. Restore verified artifacts through the prebuilt-stage mechanism.
1. Configure again so CMake recognizes the prebuilt stages. Require selected
   fingerprints to match their values before restoration.
1. Build uncached projects and run the selected downstream work and tests.

A lookup hit is only a candidate. Missing objects, transfer errors, invalid
checksums, restoration failures, or changed post-restoration fingerprints must fall back to a source build for the affected restore group: remove its restored state and prebuilt markers, then reconfigure. Failed reuse must not leave prebuilt state that suppresses that build.

Population copies objects into the current run's artifact location so downstream
jobs can continue fetching from that run. The database retains pointers to the
original objects. The transfer and storage costs are part of reuse measurements.

#### Producer flow

1. Build and package the artifact.
1. Upload its archives, checksums, and matching fingerprints.
1. Publish an entry only after all required uploads succeed.

Publish newly built outputs. Restored artifacts retain their original producer
information; copying them does not make the current run their producer.

#### Rollout and local use

Use one `stage_reuse_mode` control during migration, with exactly one reuse
implementation owning each restore group. Migrated restore groups use fingerprint
reuse; others may continue baseline-run reuse. The two paths must not restore the
same outputs or let baseline reuse override a fingerprint rejection.

For automatic reuse, `off` disables restoration, `dry-run` reports decisions, and
`reuse-stage` permits restoration. Preserve explicit manual reuse through
`prebuilt_stages` and `baseline_run_id` in `dry-run`, as described in the
[stage reuse documentation](../development/stage_reuse.md). Release policy excludes
cache consumption, including manual reuse. Migrated restore groups do not require
a baseline run ID. Detailed mode precedence and workflow wiring belong in the CI
implementation plan.

Local use remains an explicit script-driven lookup and bootstrap operation.
Ordinary CMake invocations do not automatically access the remote cache.

### 3. Remote cache database: storage and lookup

The **artifact cache index** is the lookup interface, implemented first by a
static JSON file and later by the remote cache database. The static prototype
([#10026](https://github.com/ROCm/TheRock/pull/10026)) is easy to inspect and edit
for experiments; it is not the production design.

The remote cache database must support batched lookup, automated trusted
publication, and historical queries relating artifacts and fingerprints to
producer runs. Replacing the static backend should not require rewriting
restoration logic.

#### Lookup identity and records

Conceptually, lookup uses:

```text
(epoch, compatibility scope, artifact name, component, target,
 artifact fingerprint)
```

The artifact fingerprint comes from the build system; the client must not
independently rehash its inputs. Compatibility scope retains required distinctions
such as platform until they are demonstrably covered by the fingerprint.
Manually supplied build-variant labels should not duplicate configuration already
represented. Target identity identifies the requested output even when its
fingerprint is target-specific.

Each record contains object locations, archive names, checksums, and original
producer information (repository and run). This producer information is
**provenance**, not an additional build input. A global epoch permits invalidation
when the fingerprint contract or compatibility assumptions change.

#### Publication and recovery

Publish complete records after uploads finish. Readers must handle stale records
whose objects have expired by falling back to building. The database must be
recoverable from artifact storage or producer history while preserving trusted
provenance and validating required metadata. A `.fprint` file alone is not enough
to establish a reusable entry.

Database technology, schema, retry/concurrency behavior, retention, recovery, and
authorization need an implementation plan before production enablement.

## Alternatives considered

- **Continue baseline-run reuse:** retains workflow-history and stage-level
  selection instead of direct artifact-input matching.
- **Use a Git-maintained artifact cache index in production:** useful for
  experiments, but continuous publication would create excessive repository
  updates.
- **Store only a fingerprint-to-URL mapping:** insufficient for required-component
  completeness, checksums, and producer traceability.
- **Rely only on compiler caches:** helps compilation but does not reuse complete
  installed and packaged artifacts.

## Implementation Plan

Start with AMD LLVM using the static artifact cache index. Connect the remote
cache database and revalidate AMD LLVM before expanding outward through the build
graph.

### Enablement checklist

For **each milestone from 2 through 7**:

1. First validate fingerprint correctness and stability under
   [#9525](https://github.com/ROCm/TheRock/issues/9525): equivalent inputs match,
   relevant changes invalidate affected artifacts, and unaffected controls remain
   stable.
1. Verify complete producer metadata and lookup/population in an independent
   consumer run.
1. Restore into fresh build directories, configure again, and require unchanged
   expected fingerprints, skipped compilation of restored projects, and successful
   downstream builds and selected tests.
1. Exercise fallback, report useful diagnostics, and measure total reuse cost.

| Milestone                                                        | Deliverable                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Fingerprint generation and upload**                         | Generate fingerprints and upload them with archives and checksums; [#9921](https://github.com/ROCm/TheRock/pull/9921).                                                                                                                                                                                         |
| **2. Validate AMD LLVM with the static artifact cache index**    | Integrate lookup, population, verification, and restoration of the `amd-llvm` artifact (subprojects `amd-llvm`, `amd-comgr`, and `hipcc`) in the compiler/runtime stage job. Use manually recorded entries and keep uncached projects building normally; [#10026](https://github.com/ROCm/TheRock/pull/10026). |
| **3. Connect the remote cache database and revalidate AMD LLVM** | Replace static lookup with automated trusted publication and remote lookup. Repeat validation across independent CI runs, including authorization, history queries, recovery, and operational ownership.                                                                                                       |
| **4. Expand to base projects and upstream dependencies**         | Enable the base subprojects `rocm-cmake` and `rocm-half`, fetched dependencies/sysdeps, and base helpers such as `therock-aux-overlay`. Audit fetch recipes, wrappers, and patches.                                                                                                                            |
| **5. Expand to remaining compiler and supporting projects**      | Enable `hipify` and other eligible separate Git checkouts or subtrees in dependency order.                                                                                                                                                                                                                     |
| **6. Expand to ROCm Systems**                                    | Enable runtime/HIP, profiler, debugger, emulation, and communication projects, including their shared sources, wrappers, and helpers.                                                                                                                                                                          |
| **7. Expand to ROCm Libraries and device-specific artifacts**    | Enable libraries and providers, accounting for shared generators and target-specific inputs. Implement required split/device fingerprint and restoration support.                                                                                                                                              |
| **8. Retire baseline-run reuse**                                 | Remove legacy selection and stage-copy decisions after required workflows demonstrate reliable reuse, fallback, downstream tests, and measurable savings.                                                                                                                                                      |

AMD LLVM's upstream dependencies must contribute correctly to its fingerprints
from milestone 2, even while those dependencies build from source. Milestone 4
enables reuse of their own artifacts. Later expansion follows declared
dependencies, including cross-repository inputs. In-tree helpers are validated
with the artifacts containing their outputs.

Use the input fixes applicable to each tested configuration. A same-location
prototype does not demonstrate cross-location stability. Track implementation
work under [#8651](https://github.com/ROCm/TheRock/issues/8651), linking component
issues as their scope and ownership are agreed.

## Example test scenarios

Use fresh top-level TheRock and per-subproject build directories. These are
acceptance criteria, not claims that every scenario is implemented. A predicted
hit assumes a compatible entry exists and its objects are available. Check both
affected projects and unaffected controls.

| Scenario                                                                        | Expected result                                                                                                                                                                    |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Identical fresh runs                                                            | Fingerprints match and complete matching artifacts can be reused.                                                                                                                  |
| Reconfigure after restoration                                                   | Selected fingerprints match their pre-restoration values; CMake skips compilation of restored projects and downstream builds succeed.                                              |
| ROCm Libraries PR changes only rocFFT sources                                   | Once its artifact support is enabled in milestone 7, rocFFT and affected consumers change. Eligible unchanged compiler and ROCm Systems artifacts remain reusable.                 |
| ROCm Systems PR changes runtime code outside LLVM's inputs                      | Runtime and affected consumers change; compiler fingerprints remain stable if none of their inputs changed.                                                                        |
| ROCm Systems PR changes HSA headers consumed by LLVM                            | LLVM and affected consumers change; old compiler entries must not be reused.                                                                                                       |
| TheRock PR changes an LLVM argument or selected recipe file                     | LLVM and affected consumers change; unaffected controls remain stable.                                                                                                             |
| TheRock PR changes documentation outside build inputs                           | Fingerprints remain stable if the workflow runs.                                                                                                                                   |
| Enabled upstream dependency changes, such as rocm-cmake or bundled zlib         | Its fingerprint and affected transitive consumers change. Restoring all inputs returns fingerprints to their previous values.                                                      |
| Identical HIPCC inputs at different times, then a relevant source change        | Time alone preserves fingerprints despite possible changelog differences. The source change invalidates affected fingerprints.                                                     |
| Checkout path, environment, or target changes                                   | Equivalent supported inputs remain stable; compatibility-relevant changes invalidate affected entries. Cross-location and device/split reuse require validation before enablement. |
| Invalid fingerprint, incomplete objects, transfer failure, or checksum mismatch | The affected restore group builds from source. Independent complete restore groups may still be reused.                                                                            |
| Reuse controls and migration                                                    | Automatic reuse follows the selected mode; each restore group has one owner. Manual inputs follow the documented policy, and releases consume no cached artifacts.                 |
| Untrusted publication or epoch change                                           | Unauthorized publication is rejected. Previous-epoch entries cannot match the new epoch.                                                                                           |

The repository examples follow actual dependencies, not blanket repository rules.
Detailed dirty/untracked-source checks belong to the fingerprint audit in #9525;
no unrepresented input may silently authorize incompatible reuse.

Record the tested revision, configuration, and coverage. Measure configuration,
lookup, transfer, verification, restoration, remaining compilation, and total
workflow time. A fast lookup alone does not demonstrate an equivalent reduction
in end-to-end build time.

## Open implementation decisions

- Remaining input coverage and split/device-specific fingerprint support (#9525).
- Database deployment, schema, compatibility scope, epoch handling, and ownership.
- Trusted producer eligibility, write permissions, publication retries/concurrency,
  retention, and recovery.
- CI mode precedence, restore ownership, and fallback implementation.
- Quantitative rollout criteria and conditions for retiring baseline-run reuse.
