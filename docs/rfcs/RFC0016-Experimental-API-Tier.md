---
author: T.J. Alumbaugh (talumbau)
created: 2026-10-28
modified: 2026-11-13
status: draft
---

# An Experimental API Tier for ROCm Libraries

Ship unfinished APIs in every release without turning them into permanent commitments.

## Summary

- **What.** Each ROCm library may expose an experimental API tier: a separate header, installed in every build, that consumers opt into explicitly. It ships in every TheRock release, so a new feature reaches customers without a custom build and without a stable-API commitment.
- **One ROCm-wide opt-in.** A consumer defines `ROCM_ENABLE_EXPERIMENTAL_APIS` in its own build and includes the experimental header of each library it uses. Without the switch, an experimental header declares nothing.
- **Per-feature revisions, checked at compile time and run time.** Each feature defines a macro, `<LIB>_EXPERIMENTAL_<FEATURE>`, set to its revision, and each library answers a runtime query for the revision it actually has. On a mismatch, each consumer chooses to error out or to fall back gracefully to its non-experimental path.
- **Revisions bump only on breaking changes, detected by CI.** Additions do not bump. CI compares the experimental declarations against the PR's merge base, and each feature's promised behavior is pinned by contract tests that cannot change without a revision bump.
- **No in-place changes.** A breaking change to a function gives it a new name, and experimental enum values are never renumbered or reused. A stale binary therefore fails loudly or falls back; it never silently runs changed code.
- **Bounded, enforced lifetime.** Each feature has an owner and must graduate to the stable API or be removed within four ROCm releases. One release-long extension is allowed with an approver. The build fails once the deadline passes, and a cap on concurrent features keeps the tier from becoming a second, unpromised public API.
- **Frameworks can use it safely.** A consumer standard for framework code (detect, check the revision, keep a fallback, ship official builds with the switch off) means a ROCm upgrade never breaks a framework build.
- **First adopter: hipBLASLt.** A working implementation of the bump check and the deadline is in draft PR [rocm-libraries#14065](https://github.com/ROCm/rocm-libraries/pull/14065), and the fused GEMM + all-to-all epilogue is the first real feature.

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

What is missing is a tier that ships in every build, that consumers opt into explicitly, that is free to change between releases, and whose changes never reach a consumer silently.

Frameworks make this harder.
Experimental features are built to be tried, and frameworks such as PyTorch are the consumers most likely to try them first.
A framework supports several of its own releases against each ROCm release, and old framework releases cannot be patched.
If an experimental change simply broke every consumer's build, the next ROCm and the next framework release would have to land together, and older framework releases would stop building against the new ROCm.
So the tier lets each consumer decide what a change means for it: a customer who wants to know immediately can fail the build, and a framework can fall back to its non-experimental path and keep working.

This RFC defines that tier as a ROCm-wide convention, with the rules that keep it bounded.
It asks for two decisions:

1. **Policy.** Endorse an experimental API tier with a bounded lifetime and enforced graduation as ROCm policy, not a per-library choice.
1. **Scope.** Either adopt in hipBLASLt first and extend to other libraries on evidence, or commit now to a convention every library follows when it first needs an experimental API.

## Background

There is no shared convention across rocm-libraries today.
A few libraries have a beta or experimental surface, gated by a boolean define (for example rocBLAS's `ROCBLAS_BETA_FEATURES_API`).
None of them versions that surface, so an upgrade that changes the semantics of a beta API compiles silently.
Nothing forces graduation either: the rocBLAS beta API landed in ROCm 5.5, and `rocblas_gemm_ex_get_solutions` is still beta today.

The lifecycle rules in this RFC exist to prevent that outcome.

### Prior art: Vulkan provisional extensions

The gating model in this RFC follows Vulkan's [provisional extensions](https://registry.khronos.org/vulkan/specs/latest/man/html/provisional-headers.html), which solve the same problem: shipping an API that is ratified but may still change, in every driver, without committing to it.

- **Separate header, consumer-side switch.** Provisional extensions are declared in `vulkan_beta.h`. `vulkan.h` includes it only if the application defines `VK_ENABLE_BETA_EXTENSIONS` first. Every installed Vulkan SDK has the same headers; the application decides what it sees.
- **Per-extension compile-time macros.** Each extension defines a presence macro (for example `VK_KHR_portability_subset`) and a revision macro (`VK_KHR_PORTABILITY_SUBSET_SPEC_VERSION`). Applications test them before using the extension.
- **Runtime revision query.** `vkEnumerateDeviceExtensionProperties` reports which extensions the installed driver supports, each with its `specVersion`, so an application checks at run time what it compiled against.
- **Runtime function lookup.** Applications get extension functions from `vkGetDeviceProcAddr`, which returns `NULL` for a function the driver does not have. A binary built against a newer header therefore still loads on an older driver and can fall back.
- **Not for production.** When it released provisional extensions such as [ray tracing](https://www.khronos.org/blog/vulkan-ray-tracing-final-specification-release), Khronos asked driver vendors not to ship them in production drivers and developers not to use them in production applications.

The mapping to this RFC:

| Vulkan                                 | This RFC                                       |
| :------------------------------------- | :--------------------------------------------- |
| `VK_ENABLE_BETA_EXTENSIONS`            | `ROCM_ENABLE_EXPERIMENTAL_APIS`                |
| `vulkan_beta.h`                        | `<lib>/<lib>-experimental.h`                   |
| `VK_<EXT>_SPEC_VERSION`                | `<LIB>_EXPERIMENTAL_<FEATURE>`                 |
| `vkEnumerateDeviceExtensionProperties` | `<lib>ExperimentalGetFeatureRevision`          |
| `vkGetDeviceProcAddr`                  | Deferred (see Stale binaries)                  |
| No time limit                          | Four ROCm releases, enforced at configure time |
| Promotion renames and may change APIs  | Graduation renames; no aliases                 |

Two points of experience shape where this RFC differs:

- **Provisional can mean permanent.** Vulkan sets no time limit. [`VK_KHR_portability_subset`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_portability_subset.html), released provisionally in 2020 and required by MoltenVK on Apple platforms, is still provisional, so production applications define `VK_ENABLE_BETA_EXTENSIONS` to use it. This RFC adds the deadline and the cap so a feature cannot stay experimental indefinitely.
- **Promotion is a breaking change.** When the provisional ray tracing extension was finalized, it was split into three extensions and its API changed, so code written against the provisional version had to be rewritten. This RFC treats graduation the same way, as a deliberate rename with no aliases, and makes the consumer's fallback path the way through it.

## Design principles

**P1. Experimental is an API status, not a build configuration.** An experimental feature is compiled into every build and ships in every release. Only its visibility in the headers is gated, and the consumer controls that gate.

**P2. Consumers opt in explicitly, and changes never reach them silently.** No consumer depends on an experimental API without saying so. Every breaking change is visible to the consumer, at compile time and at run time, and the consumer decides whether it errors out or falls back.

**P3. Machines enforce the rules, not memory.** Revision bumps and deadlines are checked by CI and by the build, with human review as a backstop rather than the mechanism.

**P4. Every feature leaves the tier.** It graduates or it is removed, on a schedule fixed when it enters.

## Non-goals

- **Replacing stable API or deprecation policy.** The stable API keeps its own compatibility rules. This RFC only governs the experimental tier.
- **Per-feature build flags.** See P1.
- **Mandating immediate adoption.** No library has to add an experimental tier. A library that ships an experimental API follows this RFC.
- **Keeping stale binaries running.** A binary built against an older revision of a feature may fail to load against a library where that revision is gone (see Stale binaries). Runtime function lookup, which would let such binaries fall back instead, is deferred.

## Proposal

In this section, `<lib>` and `<LIB>` stand for a library's usual prefix in lower and upper case, `<FEATURE>` for a feature's registered name, and hipBLASLt is used for examples.

### The ROCm-wide switch

**One switch for all of ROCm.** A consumer enables the experimental tier by defining `ROCM_ENABLE_EXPERIMENTAL_APIS` as `1` in its own source or compile flags, then includes the experimental header of each library it uses:

```c
#define ROCM_ENABLE_EXPERIMENTAL_APIS 1   /* or -DROCM_ENABLE_EXPERIMENTAL_APIS=1 */
#include <hipblaslt/hipblaslt-experimental.h>
```

The switch is set by the consumer, not by the library's build.
The installed ROCm is the same for everyone: every experimental feature is compiled into the library, and every experimental header is installed.
With the switch on, a consumer sees everything in every experimental header it includes.
Which libraries and features it actually uses is up to which headers it includes and which feature macros it checks.
There is no per-library or per-feature switch.

**Without the switch, an experimental header declares nothing.** No functions, types or feature macros.
A consumer can therefore include an experimental header unconditionally, and its feature checks (below) take the fallback path when the switch is off.
A consumer who forgets the switch and calls an experimental function gets an "undeclared identifier" error; the docs page names that symptom.
Defining the switch as `0` is the documented way to say "off on purpose".
An `#error` or `#warning` here was rejected: the first forces every fallback-style consumer to wrap the `#include` in its own `#if`, and the second breaks `-Werror` builds.

A library's exported CMake package may offer a convenience target (for example `roc::hipblaslt_experimental`) that adds the define for CMake consumers.

**Separate header.** `<lib>/<lib>-experimental.h` is installed in every build but is not included by the library's main header.
Each feature lives in its own header, included by `<lib>-experimental.h` and guarded so it cannot be included on its own.
All experimental symbols are always compiled into and exported from the library.

### Per-feature revision macros

Each feature's header defines one macro, named `<LIB>_EXPERIMENTAL_<FEATURE>`, whose value is the feature's revision:

```c
#define HIPBLASLT_EXPERIMENTAL_GEMM_A2A 1
```

The macro being defined means the feature is present; its value is the revision of the declarations that follow it.
Feature names are registered (see Lifecycle), so they cannot collide with other experimental constants.

A consumer checks both that the macro exists and that it has the revision the consumer's code was written for:

```c
#if defined(HIPBLASLT_EXPERIMENTAL_GEMM_A2A) && HIPBLASLT_EXPERIMENTAL_GEMM_A2A == 1
    /* code written against revision 1 */
#endif
```

Testing only that the macro exists is a bug: it keeps compiling old code against a new revision.

### The runtime revision query

The compile-time macro tells a consumer which revision it compiled against.
A binary also has to know which revision the library it is running against has, because a binary built against one ROCm can run on another.
Each library therefore exports one query, with the same shape in every ROCm library:

```c
/* Revision of the named experimental feature in the loaded library, or 0 if absent. */
int hipblasLtExperimentalGetFeatureRevision(const char* feature);   /* e.g. "GEMM_A2A" */
```

The library is built with the switch on, so the query returns the revisions the library was compiled with.
The feature is named by a string so the query never depends on an experimental enum that could itself change.
This query is the one part of the experimental tier that never changes and is never removed.

### Handling a revision mismatch

When a feature's revision changes, the library does nothing else: the consumer's own code decides what a mismatch means.
Each consumer chooses to error out or to fall back gracefully.

**Falling back:**

```c
#if defined(HIPBLASLT_EXPERIMENTAL_GEMM_A2A) && HIPBLASLT_EXPERIMENTAL_GEMM_A2A == 1
    if (hipblasLtExperimentalGetFeatureRevision("GEMM_A2A") == 1) {
        /* fused GEMM + all-to-all path */
    } else {
        /* fallback: unfused GEMM, then a separate all-to-all */
    }
#else
    /* fallback */
#endif
```

Against a header at revision 2 the `#if` is false, so the fallback compiles; against a library at revision 2 the runtime check fails, so the fallback runs.
The consumer keeps working until someone updates its code for revision 2.
Falling back is quiet, so a consumer that falls back should have a test or benchmark that notices when it does.

**Erroring out:**

```c
#if !defined(HIPBLASLT_EXPERIMENTAL_GEMM_A2A) || HIPBLASLT_EXPERIMENTAL_GEMM_A2A != 1
#error "hipBLASLt GEMM+A2A is not revision 1; review the hipBLASLt CHANGELOG"
#endif
```

plus a runtime check that refuses to run if the query does not return 1.
`#error` is the standard preprocessor directive that fails compilation with the given message.

### Naming and enum values

**Naming.** C functions use `<lib>Experimental*` (or `<lib>_experimental_*` in libraries with snake-case APIs), constants, enumerators and feature macros use `<LIB>_EXPERIMENTAL_*`, and C++ APIs use a `<lib>_experimental` namespace.
"Experimental" is spelled out: `Exp` reads as "exponential", and in hipBLASLt it is one letter away from `Ext`.

**Extending stable enums.** Many features are new values of an existing enum, used through existing calls (for example a new matmul descriptor attribute).
Experimental values of a stable enum use a reserved range starting at `0x40000000` (bit 30), declared only in the experimental headers.
Each adopting library confirms that no stable value reaches that range.
Brand-new types need no marker; they live in the experimental headers.

### No in-place changes

A breaking change never reuses a name or a value:

- **Functions.** A function whose signature or documented behavior changes gets a new name, for example `hipblasLtExperimentalFoo` becomes `hipblasLtExperimentalFoo_v2`, and the old name is removed.
- **Enum values.** An experimental enum value is never renumbered or reused. A retired value is rejected by the library with an invalid-value status.

This is what keeps a stale binary from silently running changed code (see Stale binaries).

### Revision rules

Each feature's revision changes only for breaking changes to that feature:

| Change to a feature                                                      | Revision bump?                             |
| :----------------------------------------------------------------------- | :----------------------------------------- |
| Add a function, type or enumerator                                       | No                                         |
| Remove a function, type or enumerator                                    | Yes                                        |
| Replace a function with a renamed version (signature or behavior change) | Yes                                        |
| Retire an enum value                                                     | Yes                                        |
| Change documented behavior (a contract test changes)                     | Yes                                        |
| Graduate or remove the feature                                           | Feature macro removed; the query returns 0 |

Additions do not bump, so a consumer of a feature is not affected by unrelated growth.
Other features' revisions are untouched by a change to this one.

Every change, breaking or not, gets an entry under an `### Experimental` heading in the library's CHANGELOG, naming the feature and its new revision, with breaking changes marked.

### Detecting a missing revision bump

Whether an implementation change "matters" cannot be detected in general.
Instead, this RFC defines "behavior changed" as "a contract test changed", which CI can detect.
There are three layers: declarations and behavior are checked automatically, and a human backstop catches what they miss.

Both automated layers run in one checker on every PR that touches the experimental API.
It compares the PR with its merge base on the target branch, not with a stored snapshot.
There is no baseline file to update, so there is none to edit to get around the check, and every PR that breaks something must bump in that PR.
The checker attributes each change to the feature whose header or contract tests it touches, and requires that feature's revision to go up.

**Declarations (fully automated).** The checker extracts every experimental declaration: functions, typedefs, structs, macros, and each enumerator with its value, with comments and whitespace normalized away.
It fails a PR that:

- removes a declaration without bumping the feature's revision;
- changes the signature of a function under an existing name (the no-in-place rule);
- changes an enumerator's value, or gives a new enumerator a value below the library's experimental enum high-water mark (a macro that only goes up), so retired values cannot be reused;
- declares an enumerator without an explicit value;
- lowers a revision.

**Behavior (contract tests).** Each feature has a small contract-test suite covering only its documented promises: which inputs are accepted and rejected, default attribute values, required attributes, status codes, and results within tolerance of a reference.
Kernel selection, performance and bit-exact output are not part of the contract unless the docs promise them.
Two rules turn these tests into a bump detector:

- **Changing a contract test requires a revision bump.** The checker hashes each contract-test file, ignoring comments and whitespace. A PR that modifies or deletes one must also bump that feature's revision. Adding tests is free.
- **The contract tests run in normal CI.** Because they cannot change without a bump, the tests in a PR are the tests that shipped at the last bump. Running them against the PR's library runs the old contract against the new code. A behavior change makes a test fail; the author either fixes the bug or edits the test, and editing the test forces the bump.

A behavior change also triggers the no-in-place rule, so the affected function gets a new name.
The checker cannot see that from a contract-test change alone, so review enforces it.

The checker warns when a revision is bumped but nothing breaking was detected.

**Human backstop.** CODEOWNERS covers the experimental headers, implementation and contract tests, and the PR template asks whether the change alters documented behavior of an experimental API.

The docs and the contract tests define an experimental feature's behavior.
Behavior not covered is not promised, and a behavior change that should have bumped the revision but did not means the contract is missing a test.

### Consumer standard

Framework code that uses an experimental feature follows four rules:

- **Detect.** Use the feature only under `ROCM_ENABLE_EXPERIMENTAL_APIS` and an exact-revision test of its feature macro.
- **Check at run time.** Before the first call, confirm with the runtime query that the loaded library has the same revision.
- **Keep a fallback.** Keep a working non-experimental path until the feature is stable. The experimental path is never the only path.
- **Ship official builds with the switch off.** Official and release builds, such as release wheels, are built with `ROCM_ENABLE_EXPERIMENTAL_APIS` set to `0` until the feature is ready. Opt-in builds turn it on.

**Enforcement.** The standard binds AMD-authored framework code, and AMD reviewers check it on those PRs.
AMD-run PyTorch CI includes a job that builds with `ROCM_ENABLE_EXPERIMENTAL_APIS=0`, so a missing fallback fails to compile.
For other consumers, the standard is a recommendation.

### Stale binaries

The compile-time check runs once, against the headers on the build machine; the binary then runs against whatever library is installed where it runs.
For example, a framework wheel built in January against a ROCm where a feature is at revision 1 can be installed in June on a ROCm where it is at revision 2.
The `#if` was decided in January and is fixed in the binary.

With the runtime query and the no-in-place rule, such a binary never silently runs changed code:

- If it checks the runtime revision before using the feature, it falls back.
- If it calls a function that was renamed or removed, it fails with a dynamic-loader error ("undefined symbol ..."), at startup or at the first call depending on how it was linked. That is loud and names the symbol.
- If it passes a retired enum value, the library rejects it with an invalid-value status.

The cost is that a load failure takes down the whole application, including users who never touch the experimental feature.
For example, importing a framework can fail on a ROCm where a function its wheel references was removed.

**Deferred: runtime function lookup.** A library could provide a lookup like Vulkan's `vkGetDeviceProcAddr` (see Background), `<lib>ExperimentalGetProcAddress("name")`, that returns `NULL` for absent functions, with consumers calling experimental functions only through the pointers it returns.
A stale binary would then always load and fall back.
The cost is a function-pointer typedef for every experimental function and function-pointer calls in every consumer.
This is the next step if frameworks need prebuilt binaries to survive a revision change.

### Advertising the risk

Everywhere a user meets the tier, it says plainly that any new ROCm release may change or remove an experimental feature:

- **A header banner** in each experimental header repeats the warning and lists each feature with its revision, introduction release and deadline. The list is generated from the feature registry (see Lifecycle), so it cannot go stale.
- **A docs page** that opens with a warning admonition, then gives this policy, the consumer standard, and the per-feature table.
- **Release notes** for each ROCm release, listing each library's experimental features and their revisions, and which changed.
- **Partner communication.** When we point a partner at an experimental feature, we tell them it is pre-release, give the graduation target, and point them at the revision checks.

A `[[deprecated]]`-style warning on every call (the rocBLAS approach) is deliberately not used.
Defining the switch is already an explicit acknowledgment, and per-call warnings break `-Werror` consumers without telling them anything new.

### Lifecycle

**Maximum lifetime: four ROCm releases.** A feature that enters in ROCm release N must graduate or be removed by release N+4.
Patch releases do not count.
Revision bumps do not reset the deadline: the window runs from the release the feature first entered.
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
The cap counts features, not functions: a feature is one coherent capability with one owner, one docs entry, one revision macro and one deadline.
A feature over the cap waits until another graduates or is removed.
The cap keeps the tier from becoming a second, unpromised public API, and it forces a decision on stale features.

**Graduation.** Graduation moves a feature to the stable API: its symbols are renamed (`<lib>ExperimentalFoo` to `<lib>Foo`) and its enum values get normal stable numbers.
The feature's header, revision macro and registry entry are removed in the same release, and the runtime query returns 0 for it.
There is no alias period: a binary has the old enum values compiled in, so no alias can make it correct, and an alias delays the cleanup this policy exists to enforce.
A consumer that checks the feature falls back until it moves to the stable API; one that errors out on a mismatch gets its error and moves.

**Removal** follows the same rules, and the CHANGELOG says what replaces the feature, if anything.

## What changes for a library repository

Nothing, until the library ships its first experimental API.
Then the library adds:

- `<lib>-experimental.h`, which declares nothing unless `ROCM_ENABLE_EXPERIMENTAL_APIS` is `1`, and the runtime revision query.
- One header per feature, each defining its `<LIB>_EXPERIMENTAL_<FEATURE>` revision macro and guarded so it cannot be included on its own.
- The library's own build defines `ROCM_ENABLE_EXPERIMENTAL_APIS`, so every experimental symbol is compiled in and exported.
- A feature registry (`experimental/features.cmake` or equivalent) with each feature's header, owner, tracking issue, introduction release and last release.
- A call to the shared deadline helper in its top-level CMake.
- Contract tests per feature, in a known directory the checker reads.
- A path-filtered CI job that runs the shared checker against the PR's base.
- CODEOWNERS entries for the experimental headers, registry and contract tests.
- An experimental API docs page and an `### Experimental` CHANGELOG section.
- For TheRock builds, the ROCm version passed to its subproject.

## Shared tooling

The checker and the deadline helper should be shared, not copied into each library, so the rules cannot drift between libraries.
The proposal is to host them once in rocm-libraries (the top-level `cmake/` and `tools/` directories already hold cross-project tooling) and parameterize them by library prefix, header paths and contract-test paths.
The ordered list of ROCm releases lives with them.
Where libraries outside rocm-libraries get them from is an open question.

## Reference implementation: hipBLASLt

hipBLASLt is the first adopter.
Draft PR [rocm-libraries#14065](https://github.com/ROCm/rocm-libraries/pull/14065) implements the tier and its enforcement with a trivial demo feature:

- the experimental header and runtime getter;
- the bump checker (`tools/scripts/check_experimental_api.py`) and its CI workflow;
- the feature registry and the CMake deadline helper, including the extension and cap rules;
- contract tests for the demo feature;
- pytest suites that exercise both checks against the real headers and CMake.

The PR is for review and is not meant to merge as is.
It implements an earlier draft of this RFC: a single per-library version with a versioned opt-in define, and a deadline counted in hipBLASLt minor versions.
Adopting this RFC means updating it to the ROCm-wide switch, per-feature revisions, the runtime revision query, the no-in-place checks and the ROCm release list, and moving the checker and helper to the shared location.

**First real feature: fused GEMM + all-to-all.**
The fused GEMM + A2A epilogue is already on hipBLASLt's develop branch, but its whole API is compiled out of every ROCm build.
It sits behind `#if HIPBLASLT_HAS_GEMM_A2A_FUSION` blocks in `hipblaslt.h`, driven by a CMake option that is off by default.
That causes two problems: whether the API exists depends on how hipBLASLt was built, not on the consumer; and its descriptor attribute sits in the stable attribute enum, so `HIPBLASLT_MATMUL_DESC_MAX` differs between builds.
Migrating it means moving its types and functions into the experimental tier under `Experimental` names as feature `GEMM_A2A` (`HIPBLASLT_EXPERIMENTAL_GEMM_A2A`), giving its attribute a value in the reserved range, dropping the CMake option, and building its kernels by default.

## Rollout and success criteria

1. **hipBLASLt adopts** the tier, with GEMM + A2A as its first feature, and ships it in a TheRock release.
1. **Shared tooling.** The checker, deadline helper and ROCm release list move to the shared location, and hipBLASLt uses them.
1. **Framework CI.** AMD-run PyTorch CI adds the build with `ROCM_ENABLE_EXPERIMENTAL_APIS=0` before PyTorch's ROCm code uses an experimental feature.
1. **Other libraries adopt** when they first need an experimental API, either on evidence from hipBLASLt or immediately, depending on the scope decision above.

Success looks like:

- a customer uses an experimental feature from a standard TheRock release, with no custom build;
- a ROCm upgrade never breaks a framework build because an experimental feature changed;
- every experimental feature graduates or is removed within its window, and none is extended twice;
- no experimental change reaches a consumer without a revision bump.

## Open questions and risks

- **The feature cap.** 3 is proposed; 4 is under discussion.
- **Where the deadline failure lands.** With ROCm-release counting, the configure failure appears in whatever change moves a library's build to the next ROCm release, which may be in TheRock rather than the library. The warning one release earlier is meant to give owners time to act, but the change that crosses a deadline still needs a clear owner.
- **Standalone library builds.** TheRock knows the ROCm version it is building (`version.json`). A library built on its own also needs to know which ROCm release it targets, for example from a default kept in the library and updated when a release branch is cut.
- **Where the ROCm release list and shared tooling live,** and how libraries outside rocm-libraries consume them.
- **Libraries with C++-only APIs.** The namespace convention covers naming. The declaration checker needs a C++ mode, or an ABI-based check, for templates and classes.
- **Is the automated bump check enough?** It depends on contract tests covering every documented promise. A stronger declaration check (such as `abidiff` on the exported experimental symbols) is available on Linux.
- **Runtime function lookup.** Whether and when to add it (see Stale binaries).
- **Silent fallback.** A framework that falls back loses the experimental path without an error. The consumer standard relies on tests or benchmarks to notice.
- **Adoption cost.** Each adopting library takes on contract tests and CI wiring. Shared tooling keeps this small, but it is not zero.

## Alternatives considered

**A per-library version pin** (an earlier draft of this RFC).
Each consumer defined the library's experimental API version it reviewed, and any mismatch was an `#error`.
Rejected: every breaking change to any feature broke every consumer of the library, which forces framework and ROCm releases to land together and breaks older framework releases outright.
Per-feature revisions let each consumer decide, and only consumers of the changed feature are affected.

**Per-feature opt-in switches** (one define per feature).
Rejected as more machinery than needed: one ROCm-wide switch plus per-feature revision macros gives the same control, since a consumer already chooses which headers to include and which features to check.

**A boolean opt-in without revisions** (the existing rocBLAS and MIOpen style).
Simpler, but upgrades break silently on semantic changes, and nothing tells a consumer that a feature changed.

**Changing functions in place.**
Allowed by the earlier draft, provided the version was bumped.
Rejected: a stale binary that skips the runtime check would silently call changed code. Renaming turns that into a loud failure or a clean fallback.

**Runtime function lookup now.**
Deferred, not rejected (see Stale binaries).

**Per-feature build flags** (the status quo).
The feature does not ship in TheRock, and the flag must be plumbed through TheRock and later removed.

**ELF symbol versioning for the experimental surface.**
It would strengthen the runtime story, but it is Linux-only and adds build complexity.
The runtime query covers the practical case.

## Appendix: How a competing library does it (baseline)

The closest competing GPU BLAS library takes the opposite approach.
"Experimental" there is a documentation label only: it appears in release notes, docs section titles and header comments.
Experimental APIs sit in the main header next to stable ones, with ordinary names and ordinary enum values, and nothing requires an opt-in.
We found no published policy on how long something stays experimental or what happens when it stops being experimental.

|                 | Competitor practice                              | This proposal                                                                                 |
| :-------------- | :----------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| Where it lives  | Main header, mixed with stable API               | Separate header, opt-in only                                                                  |
| Consumer opt-in | None; visible to everyone                        | One ROCm-wide switch, plus per-feature revision macros                                        |
| Change signal   | Release notes                                    | Per-feature revision, checked at compile time and run time; consumer errors out or falls back |
| Time limit      | None stated                                      | Four releases, enforced at configure time                                                     |
| Exit            | Label quietly dropped, or deprecated and removed | Explicit graduation or removal                                                                |

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
We want users to choose the experimental tier rather than drift into it, so the opt-in is explicit, and every change is visible where a consumer cannot miss it: in its own feature checks at compile time and run time.
We also bound the lifetime, so the decision to graduate or remove is forced rather than left open.
The cost is real: graduation forces a code change and a rebuild, where the competitor's users get a silent promotion.
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

The feature's revision is not in the registry: it lives in the feature header's `<LIB>_EXPERIMENTAL_<FEATURE>` macro, where the checker reads it, and the generated banner picks it up from there.

The helper checks each entry when the library is configured:

- `LAST_RELEASE` more than four releases after `INTRODUCED` fails immediately.
- More features than the cap fails.
- A `<lib>-experimental-*.h` header on disk that is not registered fails, so a feature cannot appear without a deadline.
- A missing `OWNER`, `TRACKING`, `INTRODUCED` or `LAST_RELEASE` fails.

The registry also generates the header that includes each feature header, with the banner listing every feature, its revision, its owner and its deadline.

**A feature's lifetime.**
In the four releases from 10.3 through 11.0, configure is silent, except at 11.0 itself, where it warns that this is the last release that may ship the feature.
When the build moves to the release after 11.0, configure fails with the owner and tracking link.
The owner has three ways out, each a normal PR:

- **Graduate.** Move the declarations into the stable header under stable names, giving reserved-range enum values stable numbers. Delete the feature header, its revision macro, its contract tests and its registry entry, and add CHANGELOG entries.
- **Remove.** The same, without the move.
- **Extend by one release.** Add `EXTENDED_TO` (the next release) and `APPROVED_BY` to the entry. The helper rejects an extension of more than one release, a second extension, or one without an approver.

In the hipBLASLt reference implementation, these rules are covered by pytest suites that configure a small CMake project with the real helper at different versions.
