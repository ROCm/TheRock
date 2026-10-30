---
author: T.J. Alumbaugh (talumbau)
created: 2026-10-28
modified: 2026-10-28
status: draft
---

# An Experimental API Tier for ROCm Libraries

Ship unfinished APIs in every release without turning them into permanent commitments.

## Summary

- **What.** Each ROCm library may expose an experimental API tier: a separate header, installed in every build, that consumers opt into explicitly. It ships in every TheRock release, so a new feature reaches customers without a custom build and without a stable-API commitment.
- **Versioned opt-in that breaks loudly.** A consumer states which experimental API version they reviewed. A missing or stale version stops compilation with an `#error`, and a runtime version getter catches a binary that runs against a different library. Changes are never silent.
- **Bump only on breaking changes, detected by CI.** Removals, signature or enum-value changes, behavior changes, and graduation bump the version; additions do not. CI compares the experimental declarations against the PR's merge base, and each feature's promised behavior is pinned by contract tests that cannot change without a bump.
- **Bounded, enforced lifetime.** Each feature has an owner and must graduate to the stable API or be removed within four ROCm releases. One release-long extension is allowed with an approver. The build fails once the deadline passes, and a cap on concurrent features keeps the tier from becoming a second, unpromised public API.
- **Graduation is a deliberate breaking change.** Features move to stable names and stable enum values with no compatibility aliases, so consumers rebuild and know the API is now supported.
- **First adopter: hipBLASLt.** A working implementation of the version-bump check and the deadline is in draft PR [rocm-libraries#14065](https://github.com/ROCm/rocm-libraries/pull/14065), and the fused GEMM + all-to-all epilogue is the first real feature.

## Overview

Two goals work against each other today.

Hyperscaler customers should run standard TheRock releases rather than bespoke builds.
Every custom build is a fork we build, test and support separately, and that does not scale past a handful of customers.
It is already failing in practice: one major customer's custom build does not work on the next GPU generation.

At the same time, new library features need real users before their API is final.
Today, getting a feature to a customer means either a custom build (slow, one customer at a time) or committing to a stable API before there is any feedback (risky, and slow to undo).
With TheRock's release cadence, the loop from "we built it" to "a customer ran it and told us what to change" could be as short as one nightly release.

Neither existing option serves both goals:

- **A build-time flag** keeps the API free to change, but the feature is absent from every shipped build, so a customer who wants it needs a bespoke build.
- **A public header** ships everywhere, but it is a compatibility promise from day one, and every change after feedback becomes a deprecation cycle.

What is missing is a tier that ships in every build, that consumers opt into explicitly, that is free to change between releases, and whose changes break loudly rather than silently.
This RFC defines that tier as a ROCm-wide convention, with the rules that keep it bounded.

This RFC asks for two decisions:

1. **Policy.** Endorse an experimental API tier with a bounded lifetime and enforced graduation as ROCm policy, not a per-library choice.
1. **Scope.** Either adopt in hipBLASLt first and extend to other libraries on evidence, or commit now to a convention every library follows when it first needs an experimental API.

## Background

There is no shared convention across rocm-libraries today.
A few libraries have a beta or experimental surface, gated by a boolean define (for example rocBLAS's `ROCBLAS_BETA_FEATURES_API`).
None of them versions that surface, so an upgrade that changes the semantics of a beta API compiles silently.
Nothing forces graduation either: the rocBLAS beta API landed in ROCm 5.5, and `rocblas_gemm_ex_get_solutions` is still beta today.

The lifecycle rules in this RFC exist to prevent that outcome.

## Design principles

**P1. Experimental is an API status, not a build configuration.** An experimental feature is compiled into every build and ships in every release. Only its visibility in the headers is gated, and the consumer controls that gate.

**P2. Consumers opt in explicitly, and changes break loudly.** No consumer should depend on an experimental API without saying so, and no change to one should reach a consumer silently.

**P3. Machines enforce the rules, not memory.** Version bumps and deadlines are checked by CI and by the build, with human review as a backstop rather than the mechanism.

**P4. Every feature leaves the tier.** It graduates or it is removed, on a schedule fixed when it enters.

## Non-goals

- **Replacing stable API or deprecation policy.** The stable API keeps its own compatibility rules. This RFC only governs the experimental tier.
- **Upstream framework builds.** Upstream builds of frameworks such as PyTorch should not depend on the experimental tier. Each version bump would break their builds against a new ROCm, by design. The intended consumers are partners and out-of-tree extensions, such as a PyTorch custom op built with `torch.utils.cpp_extension` that links the system library.
- **Per-feature build flags.** See P1.
- **Mandating immediate adoption.** No library has to add an experimental tier. A library that ships an experimental API follows this RFC.
- **ABI guarantees for experimental symbols.** The runtime version check is the guarantee. ELF symbol versioning is out of scope (see Alternatives considered).

## Proposal

In this section, `<lib>` and `<LIB>` stand for a library's usual prefix in lower and upper case, and hipBLASLt is used for examples.

### The experimental header and opt-in

**Separate header.** `<lib>/<lib>-experimental.h` is installed in every build but is not included by the library's main header.
Experimental APIs are invisible unless a consumer includes this header.
All experimental symbols are always compiled into and exported from the library.

**Versioned opt-in.** The header refuses to compile unless the consumer states which experimental API version they reviewed:

```c
#define HIPBLASLT_ENABLE_EXPERIMENTAL_API 1   // the version I reviewed
#include <hipblaslt/hipblaslt-experimental.h>
```

If the define is missing, or does not match the header's `<LIB>_EXPERIMENTAL_API_VERSION`, compilation stops with `#error`.
The message says the API may change or be removed in any release, and names the CHANGELOG section to read.

**Runtime check.** `<lib>ExperimentalGetApiVersion()` returns the version the loaded library implements.
Consumers compare it to the compiled-in macro to catch a binary built against one ROCm and run on a newer one.
The soname does not change for experimental changes, so nothing else catches this:

```c
if (hipblasLtExperimentalGetApiVersion() != HIPBLASLT_EXPERIMENTAL_API_VERSION) {
    /* built against a different experimental API: refuse to run */
}
```

### Naming and enum values

**Naming.** C functions use `<lib>Experimental*` (or `<lib>_experimental_*` in libraries with snake-case APIs), constants and enumerators use `<LIB>_EXPERIMENTAL_*`, and C++ APIs use a `<lib>_experimental` namespace.
"Experimental" is spelled out: `Exp` reads as "exponential", and in hipBLASLt it is one letter away from `Ext`.

**Extending stable enums.** Many features are new values of an existing enum, used through existing calls (for example a new matmul descriptor attribute).
Experimental values of a stable enum use a reserved range starting at `0x40000000` (bit 30), declared only in the experimental header.
Each adopting library confirms that no stable value reaches that range.
Brand-new types need no marker; they live in the experimental header.

### Version increment rules

`<LIB>_EXPERIMENTAL_API_VERSION` is a single integer per library.
It changes only for breaking changes:

| Change                                                | Bump? |
| :---------------------------------------------------- | :---- |
| Add a function, type, enumerator or feature           | No    |
| Remove a function, type, enumerator or feature        | Yes   |
| Change a signature, struct layout or enumerator value | Yes   |
| Change documented behavior (a contract test changes)  | Yes   |
| Graduate a feature to the stable API                  | Yes   |

Additions do not bump, so a user of feature A is not broken because feature B was added.
A binary that uses a newly added function and runs on an older library fails at load time with an undefined symbol, which is still loud.
Removals and signature changes would fail to compile anyway; the bump matters most for the changes a compiler cannot see, enum values and behavior.

Every change, breaking or not, gets an entry under an `### Experimental` heading in the library's CHANGELOG, with breaking changes marked.

### Detecting a missing bump

Whether an implementation change "matters" cannot be detected in general.
Instead, this RFC defines "behavior changed" as "a contract test changed", which CI can detect.
There are three layers: declarations and behavior are checked automatically, and a human backstop catches what they miss.

Both automated layers run in one checker on every PR that touches the experimental API.
It compares the PR with its merge base on the target branch, not with a stored snapshot.
There is no baseline file to update, so there is none to edit to get around the check, and every PR that breaks something must bump in that PR.

**Declarations (fully automated).** The checker extracts every experimental declaration: functions, typedefs, structs, macros, and each enumerator with its value, with comments and whitespace normalized away.
A removed or changed declaration, or a renumbered enumerator, without a version bump fails CI.
Enumerators must have explicit values, so inserting one cannot silently renumber the rest.

**Behavior (contract tests).** Each feature has a small contract-test suite covering only its documented promises: which inputs are accepted and rejected, default attribute values, required attributes, status codes, and results within tolerance of a reference.
Kernel selection, performance and bit-exact output are not part of the contract unless the docs promise them.
Two rules turn these tests into a bump detector:

- **Changing a contract test requires a bump.** The checker hashes each contract-test file, ignoring comments and whitespace. A PR that modifies or deletes one must also bump the version. Adding tests is free.
- **The contract tests run in normal CI.** Because they cannot change without a bump, the tests in a PR are the tests that shipped at the last bump. Running them against the PR's library runs the old contract against the new code. A behavior change makes a test fail; the author either fixes the bug or edits the test, and editing the test forces the bump.

The checker also fails if the version goes down, and warns when the version is bumped but nothing breaking was detected.

**Human backstop.** CODEOWNERS covers the experimental headers, implementation and contract tests, and the PR template asks whether the change alters documented behavior of an experimental API.

The docs and the contract tests define an experimental feature's behavior.
Behavior not covered is not promised, and a behavior change that should have bumped the version but did not means the contract is missing a test.

### Advertising the risk

Everywhere a user meets the tier, it says plainly that any new ROCm release will likely break their code:

- **The `#error` text**, as above.
- **A header banner** that repeats the warning and lists each feature with its introduction release and deadline. The list is generated from the feature registry (see Lifecycle), so it cannot go stale.
- **A docs page** that opens with a warning admonition, then gives this policy, the current version, and the per-feature table.
- **Release notes** for each ROCm release, stating each library's experimental version and whether it changed.
- **Partner communication.** When we point a partner at an experimental feature, we tell them it is pre-release and give the graduation target.

A `[[deprecated]]`-style warning on every call (the rocBLAS approach) is deliberately not used.
The versioned define is already an explicit acknowledgment, and per-call warnings break `-Werror` consumers without telling them anything new.

### Lifecycle

**Maximum lifetime: four ROCm releases.** A feature that enters in ROCm release N must graduate or be removed by release N+4.
Patch releases do not count.
Each feature records its introduction release and deadline in a per-library registry, which also generates the header banner and docs table.

**What counts as a release.** Releases are counted in ROCm versions, not in each library's own version, so the window is the same for every library and matches what customers see.
ROCm's numbering is not regular enough to count arithmetically, so the count uses a single ordered list of ROCm feature releases, maintained in one shared place and updated when a release is planned.
"Four releases" means four consecutive entries in that list.

**Enforcement.** A configure-time CMake check compares the ROCm version being built against each feature's deadline:

- before the feature's last release: silent;
- at its last release: a warning naming the owner;
- after it: configure fails, naming the owner and the tracking issue.

The build stays broken until the feature graduates, is removed, or gets an extension.
The failure first appears in the change that moves the build to the next ROCm release, so the default branch never breaks for everyone else, and a shipped source release keeps configuring because its own version never changes.

**One extension.** A feature may be extended once, by exactly one release, with a named approver recorded in the registry.
The check rejects a longer extension, a second extension, or one without an approver.

**Cap on concurrent features.** Each library may have at most a fixed number of experimental features at once (proposed: 3).
The cap counts features, not functions: a feature is one coherent capability with one owner, one docs entry and one deadline.
A feature over the cap waits until another graduates or is removed.
The cap keeps the tier from becoming a second, unpromised public API, and it forces a decision on stale features.

**Graduation forces a rebuild.** Graduation renames symbols (`<lib>ExperimentalFoo` to `<lib>Foo`) and gives enum values normal stable numbers.
An existing binary has the old values compiled in, so no alias can make it correct.
Graduation is therefore a breaking change: the experimental names are removed in the same release, and the experimental version bumps.
There is no alias period: a one-release alias would let source compile unchanged while binaries still carry the old enum values, and it delays the cleanup this policy exists to enforce.

**Removal** follows the same rules: the names disappear, the version bumps, and the CHANGELOG says what replaces the feature, if anything.

## What changes for a library repository

Nothing, until the library ships its first experimental API.
Then the library adds:

- `<lib>-experimental.h`, with the version macro, the opt-in `#error`s, and the runtime getter.
- One header per feature, each guarded so it cannot be included without going through the opt-in.
- A feature registry (`experimental/features.cmake` or equivalent) with each feature's header, owner, tracking issue, introduction release and last release.
- A call to the shared deadline helper in its top-level CMake.
- Contract tests per feature, in a known directory the checker reads.
- A path-filtered CI job that runs the shared bump checker against the PR's base.
- CODEOWNERS entries for the experimental headers, registry and contract tests.
- An experimental API docs page and an `### Experimental` CHANGELOG section.
- For TheRock builds, the ROCm version passed to its subproject.

## Shared tooling

The bump checker and the deadline helper should be shared, not copied into each library, so the rules cannot drift between libraries.
The proposal is to host them once in rocm-libraries (the top-level `cmake/` and `tools/` directories already hold cross-project tooling) and parameterize them by library prefix, header paths and contract-test paths.
The ordered list of ROCm releases lives with them.
Where libraries outside rocm-libraries get them from is an open question.

## Reference implementation: hipBLASLt

hipBLASLt is the first adopter.
Draft PR [rocm-libraries#14065](https://github.com/ROCm/rocm-libraries/pull/14065) implements the tier and its enforcement with a trivial demo feature:

- the experimental header, opt-in and runtime getter;
- the bump checker (`tools/scripts/check_experimental_api.py`) and its CI workflow;
- the feature registry and the CMake deadline helper, including the extension and cap rules;
- contract tests for the demo feature;
- pytest suites that exercise both checks against the real headers and CMake.

The PR is for review and is not meant to merge as is.
It counts hipBLASLt minor versions as a stand-in for ROCm releases; adopting this RFC means switching it to the ROCm release list and moving the checker and helper to the shared location.

**First real feature: fused GEMM + all-to-all.**
The fused GEMM + A2A epilogue is already on hipBLASLt's develop branch, but its whole API is compiled out of every ROCm build.
It sits behind `#if HIPBLASLT_HAS_GEMM_A2A_FUSION` blocks in `hipblaslt.h`, driven by a CMake option that is off by default.
That causes two problems: whether the API exists depends on how hipBLASLt was built, not on the consumer; and its descriptor attribute sits in the stable attribute enum, so `HIPBLASLT_MATMUL_DESC_MAX` differs between builds.
Migrating it means moving its types and functions into the experimental tier under `Experimental` names, giving its attribute a value in the reserved range, dropping the CMake option, and building its kernels by default.

## Rollout and success criteria

1. **hipBLASLt adopts** the tier, with GEMM + A2A as its first feature, and ships it in a TheRock release.
1. **Shared tooling.** The checker, deadline helper and ROCm release list move to the shared location, and hipBLASLt uses them.
1. **Other libraries adopt** when they first need an experimental API, either on evidence from hipBLASLt or immediately, depending on the scope decision above.

Success looks like:

- a customer uses an experimental feature from a standard TheRock release, with no custom build;
- every experimental feature graduates or is removed within its window, and none is extended twice;
- no experimental change reaches a consumer without a version bump.

## Open questions and risks

- **The feature cap.** 3 is proposed; 4 is under discussion.
- **Where the deadline failure lands.** With ROCm-release counting, the configure failure appears in whatever change moves a library's build to the next ROCm release, which may be in TheRock rather than the library. The warning one release earlier is meant to give owners time to act, but the change that crosses a deadline still needs a clear owner.
- **Standalone library builds.** TheRock knows the ROCm version it is building (`version.json`). A library built on its own also needs to know which ROCm release it targets, for example from a default kept in the library and updated when a release branch is cut.
- **Where the ROCm release list and shared tooling live,** and how libraries outside rocm-libraries consume them.
- **Libraries with C++-only APIs.** The namespace convention covers naming. The declaration checker needs a C++ mode, or an ABI-based check, for templates and classes.
- **Is the automated bump check enough?** It depends on contract tests covering every documented promise. A stronger declaration check (such as `abidiff` on the exported experimental symbols) is available on Linux.
- **Adoption cost.** Each adopting library takes on contract tests and CI wiring. Shared tooling keeps this small, but it is not zero.

## Alternatives considered

**Per-feature versioning** (one opt-in define per feature).
Users would only break when a feature they use changes, and each feature could have its own lifecycle.
Rejected for now as too fine-grained: it means N defines, N version macros and N runtime getters, in the mechanism that should be simplest.
The "additions do not bump" rule and the cap capture most of the benefit.
Revisit if the cap is raised or unrelated bumps start causing real churn.

**Boolean opt-in** (the existing rocBLAS and MIOpen style).
Simpler, but upgrades break silently on semantic changes, and nothing prompts the user to read the CHANGELOG.

**Accept version N-1 with a deprecation warning.**
This would give a one-release grace period.
Rejected: the header would have to support two surfaces at once, and the fix it delays is usually a one-character edit.

**Per-feature build flags** (the status quo).
The feature does not ship in TheRock, and the flag must be plumbed through TheRock and later removed.

**ELF symbol versioning for the experimental surface.**
It would strengthen the runtime story, but it is Linux-only and adds build complexity.
The runtime getter covers the practical case.

## Appendix: How a competing library does it (baseline)

The closest competing GPU BLAS library takes the opposite approach.
"Experimental" there is a documentation label only: it appears in release notes, docs section titles and header comments.
Experimental APIs sit in the main header next to stable ones, with ordinary names and ordinary enum values, and nothing requires an opt-in.
We found no published policy on how long something stays experimental or what happens when it stops being experimental.

|                 | Competitor practice                              | This proposal                                  |
| :-------------- | :----------------------------------------------- | :--------------------------------------------- |
| Where it lives  | Main header, mixed with stable API               | Separate header, opt-in only                   |
| Consumer opt-in | None; visible to everyone                        | Versioned define; `#error` if missing or stale |
| Change signal   | Release notes                                    | Version bump, compile error, runtime getter    |
| Time limit      | None stated                                      | Four releases, enforced at configure time      |
| Exit            | Label quietly dropped, or deprecated and removed | Explicit graduation or removal                 |

Following three of its experimental features across releases:

- One was deprecated as not performant and removed two releases later. It never graduated; it left through the normal deprecation path.
- One has been experimental for four minor releases, with its scope growing (new data types, new epilogues, a new architecture) while it stayed experimental. Under this RFC, its fourth release would be its last as experimental.
- One was called experimental at introduction and later appears in the regular API reference with no announcement, which looks like a quiet graduation.

So the two exits are what we expected: the label quietly disappears, or the feature is deprecated and removed.
Neither involves the consumer.
Code written against such an API compiles unchanged until the day it stops compiling.

Consumers respond by building their own opt-in.
PyTorch added an environment variable to gate one such API because the vendor considers it experimental, and a missing-symbol mismatch has been reported for the same API: when the library offers no opt-in, a large consumer builds one, and version skew shows up as a runtime failure.

**Why we deliberately differ.**
That approach rests on dense, mature documentation and a user base that reads release notes.
We want users to choose the experimental tier rather than drift into it, so the warning goes where it cannot be missed: in the build (the versioned `#error`) and at runtime (the version getter).
We also bound the lifetime, so the decision to graduate or remove is forced rather than left open.
The cost is real: graduation forces a rebuild and usually a rename, where the competitor's users get a silent promotion.
We think that trade is right, because rebuilding for production runs is a reasonable ask, and a silent, documentation-only promotion does not tell consumers that anything changed.

## Appendix: Enforcing the deadline

This appendix shows the mechanism from the hipBLASLt reference implementation, adapted to ROCm-release counting.
The release numbers are illustrative.

**The registry.** Each library lists its features in one file:

```cmake
<lib>_experimental_feature(GEMM_A2A
    HEADER       hipblaslt-experimental-gemm-a2a.h
    OWNER        "@owner"
    TRACKING     "https://github.com/ROCm/rocm-libraries/issues/NNNNN"
    INTRODUCED   10.3
    LAST_RELEASE 11.0)   # four ROCm releases, if the list is 10.3, 10.4, 10.5, 11.0
```

The helper checks each entry when the library is configured:

- `LAST_RELEASE` more than four releases after `INTRODUCED` fails immediately.
- More features than the cap fails.
- A `<lib>-experimental-*.h` header on disk that is not registered fails, so a feature cannot appear without a deadline.
- A missing `OWNER`, `TRACKING`, `INTRODUCED` or `LAST_RELEASE` fails.

The registry also generates the header that includes each feature header, with the banner listing every feature, its owner and its deadline.

**A feature's lifetime.**
In the four releases from 10.3 through 11.0, configure is silent, except at 11.0 itself, where it warns that this is the last release that may ship the feature.
When the build moves to the release after 11.0, configure fails with the owner and tracking link.
The owner has three ways out, each a normal PR:

- **Graduate.** Move the declarations into the stable header under stable names, giving reserved-range enum values stable numbers. Delete the feature header, its contract tests and its registry entry. Bump the experimental API version (the checker requires it, because declarations were removed), and add CHANGELOG entries.
- **Remove.** The same, without the move.
- **Extend by one release.** Add `EXTENDED_TO` (the next release) and `APPROVED_BY` to the entry. The helper rejects an extension of more than one release, a second extension, or one without an approver.

In the hipBLASLt reference implementation, these rules are covered by pytest suites that configure a small CMake project with the real helper at different versions.
