---
author: Zachary Garvey (zjgarvey)
created: 2026-11-06
modified: 2026-11-09
status: draft
---

# Runtime Conformance Testing in TheRock

## Summary

This RFC proposes developing a conformance test suite (CTS) in rocm-systems and
adding a TheRock CI job that applies it to the implementations in `runtimes/`
and the KFD driver environment they use. The CTS has three peer families: KFD
driver/firmware contracts, the direct rocddi interface, and the public AMDF/HSA
APIs.

The job should report conformance whenever these components or their test
dependencies change. The recommended first delivery is a bounded set of host and
native GPU cases spanning all three families. The roadmap expands contract
coverage and the architecture/platform matrix. The initial test set, runner
capacity, change-trigger integration, and enforcement policy are
[decisions for review](#decisions-for-review).

## Motivation and scope

The runtime depends on contracts at each boundary: what a driver guarantees
about a mapping or queue, what rocddi promises its frontend consumers, and what
an API promises an application. A regression at any boundary can affect the
stack above it. The CTS should document those obligations and evaluate them
directly, providing a foundation for bottom-up diagnosis.

This proposal has two deliverables:

1. A reusable CTS for the three families, owned in rocm-systems alongside the
   interfaces and implementations it evaluates.
1. TheRock CI integration that builds or obtains the required artifacts,
   supplies the execution environment, runs the CTS, and reports results for the
   selected `runtimes/` implementations and current KFD.

The HSA CTS should be capable of evaluating both the implementation in
`runtimes/` and ROCr against their common HSA contracts. Automated ROCr testing
is outside this proposal; ROCr maintainers can use the CTS if they find it
helpful.

The [runtime tree][runtime-build] already has a standalone CMake/Cargo build and
component tests. Existing smoke tests, device probes, and legacy KFD tests can
supply useful assertions or fixtures after review against the CTS contracts.
Implementation-specific unit tests remain with their components. This RFC
focuses on the CTS and its execution through TheRock.

## The conformance test suite

A CTS run evaluates a selected implementation and device or execution
environment against a defined set of interface contracts. It returns per-test
pass, fail, or skipped results. The CTS revision, contract versions, and
selected test set establish what those results mean.

Each case identifies its requirement and authority, preconditions, actions,
expected results, and cleanup obligations. Authority can come from a versioned
specification, API header, or agreed interface guarantee. Requirements that are
currently implicit need to be documented and agreed by interface owners as they
enter the CTS. Expectations remain independent of the particular implementation
selected for a run.

Host-only and device-required cases belong in the same CTS. Each case declares
the capabilities and environment it needs, including conditions for testing
specified rejection behavior.

### Families, subjects, and dependencies

| Family      | Subject under test                                                                                         | Dependency boundary and implementation selection                                                                                                                   |
| ----------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| KFD         | Driver/firmware contracts relied upon by the runtime, evaluated against production or simulated KFD.       | Test-owned, versioned kernel userspace API (UAPI) expectations and declared support. The prepared OS/device or simulation environment supplies the implementation. |
| Direct DDI  | rocddi's frontend-facing Rust interface, including device, memory, queue, and resource-lifetime contracts. | Build and link the tests against the selected rocddi revision and its dependencies. Record that revision with the executable.                                      |
| Runtime API | Public AMDF/HSA API and ABI contracts.                                                                     | Build against versioned public headers and test support; select a compatible runtime library at execution.                                                         |

Each family should be independently buildable and executable. KFD-only selection
resolves its headers and declared support; DDI selection resolves rocddi's build
dependencies; API consumers identify the runtime they load. The
[rocddi interface][rocddi] is a Rust library linked into its consumers;
selecting a different revision requires rebuilding the DDI tests.

KFD cases may use a fixed thunk or other support library. Each case records that
dependency and the driver behavior it observes. The support revision and linkage
are part of the test artifact's identity. The expected UAPI contract remains
separately versioned from the driver or model under test.

Production and simulated KFD implementations must satisfy the same applicable
contracts. Setup selects the implementation and device before launching tests.
Native results identify the kernel/driver, firmware, and hardware; simulated
results identify the model and configuration.

Related operations provide complementary evidence: KFD assertions check driver
contracts, DDI assertions check rocddi's frontend-facing obligations, and API
assertions check application-visible behavior. Each family reports its own
results. Cross-layer cases declare the interfaces they combine.

### Ownership and delivery

The proposed ownership split is:

- **rocm-systems:** contracts, CTS cases, family selection, test
  build/packaging, and the suite execution entry point. Interface maintainers
  resolve contract ambiguities and own the expected behavior.
- **TheRock:** CI orchestration, candidate artifact selection, runner and
  environment preparation, change-trigger integration, and result publication.

This follows the ownership direction of TheRock's
[test-script migration RFC][test-ownership]. CTS directory layout and helper
APIs can be settled in rocm-systems. The integration needs identifiable test
artifacts, declared dependencies, and an execution entry point usable by
developers and CI.

## Applying the CTS in TheRock CI

Add a runtime conformance job using TheRock's existing
[subproject build and test infrastructure][testing]. The job may contain
separate family and device executions while presenting a combined summary.

The proposed flow is:

1. Resolve the candidate source revision, CTS revision, contract/test set, build
   configuration, and required execution configurations.
1. Build or obtain the selected runtime and CTS artifacts with each family's
   dependencies. Package fixtures and support needed on separate test hosts.
1. Prepare the declared host, native GPU, or simulated environment. Native runs
   use the installed KFD and record its actual identity.
1. Execute the cases: API tests load the chosen runtime, DDI tests exercise
   their linked rocddi, and KFD tests exercise the prepared driver/model.
1. Reconcile results with the required inventory and link the report and
   diagnostics from the CI check.

Environment preparation owns hardware reservation or simulation sessions, access
setup, and cleanup. The CTS owns assertions and execution requirements.
Execution consumes prepared artifacts with a separate writable results location,
allowing reproduction against the same inputs.

### Change-driven execution

The objective is conformance feedback on every relevant candidate change. Start
by running the whole agreed initial test set for affected components; later
selection can use explicit contract/dependency mappings.

| Change                                                                                      | Required CTS execution                                                                                                        |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `runtimes/` implementation, interface/header, or CTS changes                                | Run selected family sets against artifacts from the exact candidate, including the agreed host and native cases.              |
| TheRock changes to runtime build inputs, packaging, CTS invocation, or runner configuration | Run affected CTS sets and verify artifact/environment selection and result collection.                                        |
| Kernel/KFD or firmware update to a test environment                                         | Run KFD contracts and applicable DDI/API device cases against the candidate environment before adopting the changed baseline. |
| Simulated KFD implementation or configuration change                                        | Run applicable KFD contracts against the candidate model, plus runtime cases assigned to that configuration.                  |

For rocm-systems changes, the integration must connect the candidate revision to
a TheRock CI invocation and return its result to the originating change. The
cross-repository dispatch and status-reporting mechanism needs agreement with CI
maintainers. TheRock dependency updates also need to select this job. Installed
driver/firmware changes need an explicit environment-update trigger through
runner provisioning.

Runtime libraries selected at execution and installed drivers need declared
test-selection relationships. TheRock's [consumer-selection RFC][test-selection]
discusses explicit policies for test relationships absent from build dependency
edges; the CTS integration must account for these consumers in the selection
mechanism it adopts.

Scheduled runs extend architecture, platform, and stress coverage beyond the
per-change set. Reports show which required configurations actually ran. Runner
shortages or failed setup leave required execution incomplete.

### Results and required coverage

Each case reports **pass**, **fail**, or **skipped**, with a reason and
diagnostics. The run also reports whether its required inventory completed.
Missing binaries, empty required suites, missing reports, and unavailable
required environments are execution/infrastructure failures. They prevent the
job from reporting successful required coverage.

Declare required tests, configurations, and expected capabilities before
execution. A skipped required case leaves coverage incomplete. An optional
capability may make a positive case inapplicable, with a recorded skip reason;
its specified rejection behavior remains testable. Unexpected loss of a required
capability is a failed expectation. An expected-rejection test passes when it
observes the specified rejection.

The report should retain:

- CTS and contract revisions, selected tests, per-case outcomes, and gaps.
- Candidate source and tested artifact identities, including the loaded API
  implementation or linked rocddi revision and declared test support.
- Device, CPU/OS, observed capabilities, kernel/driver, firmware identifiers,
  and model/configuration where used.
- Logs, fixtures or generated-input seeds, and information to reproduce a
  failure.

A conformance result applies to the named contracts and configurations executed.
Reports distinguish buildability, implemented operations, availability on the
selected device/driver, and conformance established by the executed set. CI
should publish failed assertions and incomplete execution from the first
integration; making the check a required merge condition is a
[policy decision](#decisions-for-review).

## Recommended first milestone

Deliver a working CTS slice and its TheRock job on one Linux CPU/build
configuration and one native GPU/driver baseline. Combine host-only contract
cases with native GPU cases spanning all three families. The exact subset and
configuration need maintainer review; these are starting candidates:

| Family      | Initial contract examples                                                                                                                                                           |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KFD         | Memory mapping and release; queue creation, submitted work, completion, a checked memory-visible result, and teardown. Version/capability queries establish prerequisites.          |
| Direct DDI  | A corresponding memory/queue operation through rocddi, asserting ownership, lifetime, and cleanup obligations. Include an applicable host-only contract.                            |
| Runtime API | [AMDF API negotiation][amdf-contract], including specified output preservation on rejection; an HSA dispatch or AMDF device operation with checked completion, output, and cleanup. |

The initial cases should exercise meaningful device work and its observable
result. Existing tests can supply reviewed workloads and assertions; new cases
cover the selected contracts' remaining requirements.

Completion requires:

1. Independently selectable family artifacts and a documented execution entry
   point, with identified subjects and support dependencies.
1. An automated run on the chosen host/native configurations, reporting the
   required inventory and each family's results.
1. A relevant rocm-systems candidate change selecting the job, exercising the
   exact candidate, and receiving its CI status.
1. Demonstrations that a contract failure produces a failing result and that
   missing required execution or reports prevent successful coverage. Verify
   that an environment update selects the corresponding KFD checks.
1. A reproducible failure using retained artifacts and configuration.

Per-change execution of this bounded set is the recommendation. Native merge
enforcement and expansion of the required matrix depend on agreed capacity and
ownership. Simulated configurations receive their own applicable contract sets
and retained environment identities when added.

## Coverage roadmap

The long-term goal is comprehensive conformance coverage across all three
families and every supported architecture/platform configuration. Maintain an
inventory mapping requirements to cases, applicability, execution results, and
owners of gaps. Expose untested contracts and configurations as it grows.

| Coverage area                                   | Expansion and completion evidence                                                                                                                                                                                                              |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Interface coverage                              | Inventory every KFD contract relied upon by the runtime, rocddi frontend-facing obligation, and public API/ABI requirement. Map valid inputs, errors, capability conditions, and resource lifetimes to assertions.                             |
| Execution semantics                             | Cover loading and dispatch, memory allocation/mapping/sharing, queues, events, synchronization, completion, visibility/ordering, images, and applicable extensions through their specifying interfaces.                                        |
| Failure and concurrency                         | Cover specified failures, partial construction and cleanup, resource pressure, overlapping operations, callbacks, initialization/shutdown, and process/device interactions. Retain reproducible inputs and checked lifetime/ordering outcomes. |
| Architecture and platform matrix                | Extend to every supported GPU architecture/revision, CPU ABI, kernel/firmware combination, memory configuration, and topology, including multiple devices and peer operations. Record required and executed combinations and gaps.             |
| Native and simulated implementations            | Apply common contracts to each applicable production/model configuration. Record modeled capabilities, implementation identity, and which configurations have native evidence.                                                                 |
| Systematic exploration and continuous execution | Add generated inputs, lifecycle/concurrency exploration, stress runs, and replayable regressions with contract-based assertions. Expand per-change and scheduled sets while tracking coverage and execution cost.                              |

New capabilities and contract revisions should add test requirements with the
implementation change. Sampling configurations or input/state spaces needs an
explicit rationale and a record of what ran. Implementation coverage can help
component maintainers identify additional cases; the CTS roadmap tracks the
interface obligations those cases establish.

## Decisions for review

The proposed defaults are a rocm-systems-owned CTS, TheRock orchestration, three
independent families, and a host/native first delivery with per-change
execution. Four decisions shape the initial integration:

1. **Contract slice and configurations.** Which contracts and native GPU/driver
   baseline provide the most useful first coverage? Confirm the required set and
   capability expectations. Which architectures and simulated configurations
   should follow?
1. **CI entry points and enforcement.** How should rocm-systems candidate
   changes invoke the job and receive status? Where should this fit into
   existing test selection, and when should the check become required for
   merging? Identify driver/model environment-update entry points.
1. **Ownership and resources.** Who maintains the CTS/interface contracts and
   TheRock integration, and who supplies native capacity and operates
   preparation, reservation, health checks, and recovery?
1. **Execution policy.** What per-change time/capacity budget and runner access
   policy should the initial job use? Which disruptive or expensive contract
   sets need separate scheduling and isolation as coverage grows?

Detailed CTS layout, result schema, retention, and exact CI stage can follow
these decisions using the owning repositories' conventions.

[amdf-contract]: https://github.com/ROCm/rocm-systems/blob/b21309d8249bacf94ebb2d8c1ebc51316314432c/runtimes/api-headers/include/amdf/api.h
[rocddi]: https://github.com/ROCm/rocm-systems/blob/b21309d8249bacf94ebb2d8c1ebc51316314432c/runtimes/ddi/rocddi/README.md
[runtime-build]: https://github.com/ROCm/rocm-systems/blob/b21309d8249bacf94ebb2d8c1ebc51316314432c/runtimes/README.md
[test-ownership]: https://github.com/ROCm/TheRock/blob/4e484fa197f342e5d1e1495a0fbdb0947b5507ed/docs/rfcs/RFC0010-Test-Scripts-Migration.md
[test-selection]: https://github.com/ROCm/TheRock/blob/4e484fa197f342e5d1e1495a0fbdb0947b5507ed/docs/rfcs/RFC0013-Consumer-Based-Test-Selection.md
[testing]: https://github.com/ROCm/TheRock/blob/4e484fa197f342e5d1e1495a0fbdb0947b5507ed/TESTING.md
