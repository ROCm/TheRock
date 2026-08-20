---
author: John Robbins (jorobbin)
created: 2026-07-28
modified: 2026-07-28
status: draft
discussion: TBD
---

# Code Coverage Infrastructure for TheRock

This RFC proposes integrating code coverage builds and reporting into TheRock CI, maintaining per-project isolation while leveraging TheRock's super-build infrastructure.

## Motivation

Currently, Math CI builds projects individually for code coverage:
```bash
mkdir projects/rocfft/build && cd projects/rocfft/build
cmake -DBUILD_CODE_COVERAGE ..
make -j
LLVM_PROFILE_FILE="./coverage-report/%m-%p-rocfft.profraw" make -j coverage
```

This workflow relies on prepackaged dependencies and component-specific coverage targets that handle test execution, profraw merging, and report generation.

TheRock CI needs to support code coverage while:
1. Maintaining per-project isolation for instrumentation
2. Accommodating the build/test node split architecture
3. Avoiding performance impacts on pre-checkin and performance testing builds
4. Supporting nightly coverage runs for individual components

## Background: Why Per-Project Isolation is Required

### Coverage Report Scope
Coverage reports for a given project should only instrument that project's code, not upstream or downstream dependencies. For example:
- **rocFFT coverage** should instrument only rocFFT, not hipFFT (downstream)
- **rocBLAS coverage** should not instrument hipBLASLt or downstream consumers
- **hipBLASLt coverage** should not instrument upstream dependencies like rocBLAS, rocRAND, or rocPRIM

**Why exclude downstream components?** Coverage in one component does not affect coverage in downstream components - they are independent concerns. For example:
- Generating a coverage report for rocBLAS has no effect on coverage for hipBLASLt
- Generating a coverage report for rocFFT has no effect on coverage for hipFFT
- Each component's coverage is measured independently against its own codebase

Downstream functional testing is already covered by TheRock CI's existing test infrastructure, which validates that downstream components work correctly with their dependencies. Coverage testing focuses solely on exercising the code paths within the component being measured.

### Instrumentation Behavior and Profraw Contamination
Instrumented binaries emit `.profraw` files during execution. If we instrument code upstream to the project being measured, those upstream libraries will emit profraw files whenever the downstream project calls their APIs.

Example: If both hipBLASLt and rocBLAS are instrumented, but we only care about hipBLASLt coverage:
- When hipBLASLt calls rocBLAS APIs, rocBLAS emits profraw files
- These upstream profraw files contaminate the coverage report
- We must either:
  - Explicitly ignore them with `--ignore-filename-regex`, or
  - Accept non-deterministic fluctuations in the coverage denominator that change with upstream modifications

This contamination propagates up the entire dependency chain to rocRAND and rocPRIM, which are high in the dependency tree and consumed by many downstream projects.

### Performance Impact
Coverage builds have significant performance degradation:
- **Example (rocPRIM)**: Normal pre-checkin tests take ~20 minutes; coverage runs take ~3 hours
- Most projects have less extreme degradation, but the impact is still substantial
- Coverage builds add debug instrumentation and disable compiler optimizations

### Separation from Pre-Checkin Testing
Coverage builds must be separate from pre-checkin builds:
- **Pre-checkin goal**: Test binaries as close to production as possible
- **Coverage builds**: Add debug lines, disable optimizations, enable instrumentation
- Using coverage builds for pre-checkin creates an inconsistent testing environment
- Performance testing (planned for TheRock CI) cannot use coverage builds

## Current Math CI Workflow

Projects in rocm-libraries already support coverage builds via `-DBUILD_CODE_COVERAGE`:

```bash
mkdir projects/rocfft/build && cd projects/rocfft/build
cmake -DBUILD_CODE_COVERAGE ..
make -j
LLVM_PROFILE_FILE="./coverage-report/%m-%p-rocfft.profraw" make -j coverage
```

The `coverage` target:
1. Runs the test suite
2. Drops `*.profraw` files to `LLVM_PROFILE_FILE` location (with `%m-%p` interpolation for unique identifiers)
3. Executes report generation commands:
   ```bash
   llvm-profdata merge coverage-report/*.profraw -o coverage.profdata
   llvm-cov show -object <lib>.so -instr-profile=coverage.profdata
   llvm-cov export -format=lcov -object <lib>.so -instr-profile=coverage.profdata > coverage.lcov
   ```
4. Uploads to codecov.io, which:
   - Comments on PRs showing coverage deltas
   - Can fail CI if coverage drops below 80%

### Nightly Runs
Math CI performs nightly coverage runs on `develop` for each component individually. This approach avoids resource contention and provides per-component coverage trends.

## Challenges in TheRock

### Build/Test Node Split
TheRock CI splits build nodes and test nodes and does not retain the entire CMake tree between nodes, breaking the `make coverage` target workflow.

### Solution for Most Projects
For most projects, the solution is straightforward:
1. Instrument code during build
2. Run tests using the Test Filter Standard (test_categories.yaml or test_categories_coverage.yaml)
3. Schema additions to test_categories_coverage.yaml:
   - **`llvm_profile_file`**: Path pattern for profraw file output (with `%m-%p` interpolation)
   - **`ignore-filename-regex`**: Pattern to exclude test code from coverage reports
   - **`test_names`**: List of test binary names for `-object` flags (header-only libraries only)

**Declarative approach:** These fields can potentially be calculated at the top level rather than explicitly configured per-project:
- **`test_names`**: Discovered by listing test executables matching `test_*` in a standardized test directory
- **`llvm_profile_file`**: Hardcoded to standard path (e.g., `./coverage-report/%m-%p.profraw`)
- **`ignore-filename-regex`**: Set to exclude anything outside the src directory (e.g., patterns matching test files)

This reduces per-project configuration overhead and ensures consistency across components.

**Error handling:** If profraw files are missing entirely, the coverage job fails. This indicates a fundamental problem with instrumentation or test execution.

### Problematic: Header-Only Static Libraries
Some components are header-only static libraries (e.g., rocPRIM, hipCUB) and do not produce a simple `.so` file for `llvm-cov show -object <lib>.so`.

**For these projects:**
- Must instrument the test binaries themselves (headers are compiled into the test binary)
- Must use `--ignore-filename-regex` to exclude test code from the report
- Requires explicit reference to test binaries rather than library objects
- **The `-object` flag does not accept wildcards** - cannot use `-object test_*`, must expand to `-object test_1 -object test_2 ... -object test_n`

**Handling mixed library types:**
- **Shared libraries (.so)**: Use `-object <lib>.so`, only need `llvm_profile_file`
- **Static libraries**: May require `-object` for each test binary plus `ignore-filename-regex` and `test_names`
- **Header-only libraries**: Require all three fields (test binaries, ignore patterns, profraw path)

The test runner can determine library type and adjust llvm-cov invocation accordingly. With declarative field calculation, the same test_categories_coverage.yaml schema works for all library types.

**Why test_categories_coverage.yaml may be needed:**
Because `llvm-cov show -object` does not accept wildcard patterns, we must explicitly list each test binary for header-only libraries. However, with standardized `test_*` naming:
```bash
# Declaratively generate -object flags for all tests
llvm-cov show $(for test in $(ls tests/); do echo -n "-object $test "; done) -instr-profile=coverage.profdata
```

This allows the test runner to discover test binaries automatically rather than requiring explicit per-project configuration.

## Design Proposal

### TheRock CMake Changes

#### Component-Specific Coverage Flags
Rename `-DBUILD_CODE_COVERAGE` to project-specific flags to prevent cross-contamination:
- `-DROCPRIM_ENABLE_COVERAGE`
- `-DROCFFT_ENABLE_COVERAGE`
- `-DHIPBLASLT_ENABLE_COVERAGE`
- etc.

Add TheRock passthrough:
```cmake
# In TheRock super-build
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DROCFFT_ENABLE_COVERAGE=ON
```

#### Required Compiler Flags (for device-side coverage)
Recent amd-llvm changes enable device-side coverage, requiring:
1. `-DCOMPILER_RT_BUILD_PROFILE_ROCM=ON`
   - Ensures device profiler is built into instrumented device code
2. `-DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS=OFF`
   - Build only one artifact for testing (no multi-GPU architecture splits)

### TheRock CI Workflow

#### Separate Coverage Pipeline
Code coverage requires its own build/test pipeline separate from pre-checkin:
- **Reason**: Custom instrumented build with different compiler flags
- **Architecture strategy**: Phased approach to multi-architecture coverage
  - **Phase 1 (initial)**: Single default architecture (gfx942 or gfx950) for all coverage runs - achieves parity with Math CI
  - **Phase 2+**: Multi-architecture coverage for architecture-specific code
    - Always test on default/base architecture (covers common code paths)
    - Detect architecture-specific code changes in PRs
    - Run coverage on additional architectures only when architecture-specific code changes
    - Aggregate reports across architectures via tagging (codecov.io) or profraw merging
    - Optional optimization: Replace default architecture with changed architecture when architecture-specific code changes (since changed arch covers common code too)

#### Build Phase
1. Determine coverage-enabled projects from PR changes via `therock_configure_coverage.py`:
   - Leverages existing TheRock CI infrastructure (`therock_matrix.py`) to detect changed projects
   - Maintains a whitelist of coverage-enabled projects in `COVERAGE_PROJECT_METADATA`
   - For each coverage-enabled project that changed:
     - Outputs project-specific CMake options (e.g., `-DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF`)
     - Pins build to single project only (not merged mega-group)
     - References project's `test_categories_coverage.yaml` file
   - Creates independent coverage job per project (even if multiple projects changed)
   - **Multi-component PRs**: Each coverage-enabled project gets its own isolated build → test → report pipeline

   Example metadata output for hipRAND:
   ```json
   {
     "project_name": "HIPRAND",
     "cmake_target": "hipRAND",
     "build_dir": "TheRock/build-coverage/ml-libs/hipRAND/build",
     "cmake_options": "-DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF",
     "coverage_config": "projects/hiprand/test_categories_coverage.yaml",
     "projects_to_test": "hiprand"
   }
   ```

2. Run coverage build for each project:
   ```bash
   cmake -B build -GNinja \
     -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
     -DCOMPILER_RT_BUILD_PROFILE_ROCM=ON \
     -DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS=OFF \
     -D<PROJECT_NAME>_ENABLE_COVERAGE=ON
   ninja -C build
   ```

3. Upload artifacts:
   ```bash
   python build_tools/artifact_manager.py
   python build_tools/post_build_upload.py
   ```

#### Test Phase
Modify existing test workflows:

**`therock-ci-test-packages.yml`:**
- Add `coverage_enabled` input parameter (boolean, default: false)
- Pass through to `therock-ci-test-component.yml`

**`therock-ci-test-component.yml` modifications:**

1. **Input parameter:**
   ```yaml
   coverage_enabled:
     description: "When true, collect llvm profraw from the test run and generate a coverage report."
     type: boolean
     default: false
   ```

2. **Artifact download:**
   - Pass `AMDGPU_TARGETS: ${{ inputs.coverage_enabled && inputs.amdgpu_families || '' }}` to setup_test_environment
   - Downloads coverage-instrumented artifacts when coverage_enabled is true

3. **Prepare coverage profile directory** (conditional on `coverage_enabled`):
   ```bash
   mkdir -p "${GITHUB_WORKSPACE}/coverage-report/profraw"
   echo "LLVM_PROFILE_FILE=${GITHUB_WORKSPACE}/coverage-report/profraw/${TEST_COMPONENT}-shard${matrix.shard}-%p-%m.profraw" >> "${GITHUB_ENV}"
   ```
   - Creates directory for profraw files
   - Sets `LLVM_PROFILE_FILE` environment variable with component name and shard index
   - Pattern `%p-%m` provides unique identifiers per process/module

4. **Merge profraw and generate coverage report** (conditional on `coverage_enabled`):
   ```bash
   cd ${GITHUB_WORKSPACE}/coverage-report
   ${GITHUB_WORKSPACE}/build/lib/llvm/bin/llvm-profdata merge -sparse -o coverage.profdata profraw/*.profraw
   ${GITHUB_WORKSPACE}/build/lib/llvm/bin/llvm-cov export \
     -object ${GITHUB_WORKSPACE}/build/lib/lib${COMPONENT_NAME}.so \
     -instr-profile=coverage.profdata --format=lcov > coverage.info
   ```
   - Merges all profraw files from all shards into single profdata
   - Exports coverage in lcov format for codecov.io upload
   - Uses component-specific library object file

**Artifact naming conventions:**
- Coverage builds use same artifact naming as regular builds
- Artifacts downloaded via existing `fetch_artifact_args` mechanism
- Profraw files: `${TEST_COMPONENT}-shard${SHARD_INDEX}-%p-%m.profraw`
- Coverage data: `coverage.profdata`
- Coverage report: `coverage.info` (lcov format)

### Coverage Report Upload

**Codecov.io integration:**
- Already configured for Math CI at https://app.codecov.io/gh/ROCm/rocm-libraries
- Upload token stored as GitHub repository secret
- Service handles PR commenting and coverage aggregation automatically
- Configuration via `codecov.yaml` at repository root
- Upload step in CI workflow sends `coverage.info` to codecov.io

**Provider flexibility:**
The specific coverage reporting service (codecov.io vs alternatives) and its detailed configuration are outside the scope of this RFC. The design supports any service that accepts lcov format reports.

### Resource Allocation

**Architecture scope (phased approach):**
- **Phase 1**: Single default architecture (gfx942/gfx950) - common code paths only
- **Phase 2+**: Multi-architecture coverage
  - Default architecture always tested
  - Additional architectures tested when architecture-specific code changes detected
  - Requires architecture-specific change detection capability (team buy-in and potential refactoring)
  - Nightly jobs test various architectures based on hash-range changes
- No downstream testing - only the changed project itself

**Node allocation:**
- One build node per coverage-enabled project that changed
- One test node per coverage-enabled project that changed
- Resource sizing should align with single-project testing requirements

**Expected characteristics:**
- Build time: Similar to regular builds plus instrumentation overhead (exact metrics TBD)
- Test time: Significantly longer than regular tests due to instrumentation (example: rocPRIM 20min → 3hr)
- Storage: Coverage artifacts similar size to regular artifacts; profraw files are temporary and merged/deleted after report generation

Specific node sizing, runtime benchmarks, and storage quotas remain open topics for discussion and will be refined based on initial deployment experience.

### Failure Handling

**Coverage test failures:**
- If a test fails during coverage runs, the pipeline fails
- Coverage failures block PR merges (must be fixed before merge)
- Treated the same as any other test failure in the CI pipeline

**Missing profraw files:**
- If profraw files are missing entirely, the coverage job fails
- Indicates fundamental instrumentation or test execution problem

**Flaky tests:**
- Can be disabled at component owner's discretion
- Same policy as regular pre-checkin test handling

### Multi-Architecture Coverage Strategy

**Problem:**
Architecture-specific code paths (e.g., gfx90a vs gfx942 optimizations) require architecture-specific coverage testing. Testing all architectures for all components on every PR creates unsustainable resource overhead, especially given coverage test runtime (e.g., rocPRIM: 20min → 3hr with instrumentation).

**Solution - Phased Rollout:**

**Phase 1: Default Architecture Only (Math CI parity)**
- Test all components on single default architecture (gfx942 or gfx950 - most abundant nodes)
- Covers all common code paths shared across architectures
- Establishes baseline coverage infrastructure

**Phase 2: Architecture-Specific Change Detection**
- Implement detection of architecture-specific code changes in PRs
- Requires team buy-in and potentially refactoring to identify arch-specific code paths
- Foundation for conditional multi-arch testing

**Phase 3: Nightly Multi-Architecture Coverage**
- Run coverage on various architectures using hash-range changes (between nightly runs)
- Builds baseline coverage reports for all architectures
- Populates coverage history for architecture-specific code

**Phase 4: PR-Triggered Multi-Architecture Coverage**
- Always test on default architecture (covers common code)
- When architecture-specific code changes detected:
  - Trigger coverage jobs for affected architectures
  - Aggregate reports with default architecture report

**Phase 5: Optimization - Replace Default When Appropriate**
- When PR modifies only architecture-specific code for a single architecture:
  - Run coverage on that architecture only (it covers common code too)
  - Skip default architecture to avoid redundancy

**Report Aggregation Options:**

1. **Preferred: Tag-based aggregation (codecov.io)**
   - Upload each architecture's report with architecture tag
   - Coverage service handles aggregation automatically
   - Requires verification that chosen platform supports this

2. **Alternative: Profraw merging**
   - Accumulate profraw files from all tested architectures
   - Merge on aggregate node before uploading
   - More complex, less flexible if changing platforms

**Baseline Coverage Initialization:**
The aggregate report includes:
- Current PR's default architecture coverage
- Current PR's architecture-specific coverage (if arch-specific changes detected)
- Historical coverage for untested architectures (from nightly runs or previous PRs)

**Open question:** How to handle missing baseline reports? Options:
- Self-healing: Kick off all-architecture coverage run when baseline missing (expensive, susceptible to transient failures)
- Manual initialization: Run comprehensive all-architecture coverage once at Phase 3 start
- Graceful degradation: Report only tested architectures until baseline available

### Nightly Runs

**Recommendation:** Achieve parity with Math CI:
- Separate nightly build/test for each component
- Rely on codecov.io (or chosen platform) for aggregation
- Avoids resource stress from full-ROCm coverage builds

**Multi-architecture nightly strategy (Phase 3+):**
- Test different architectures on different nights (round-robin) OR
- Test all architectures weekly/monthly to refresh baselines

## Open Questions

1. **Codecov.io vs. alternatives**: Should we standardize on codecov.io or evaluate other platforms? Must support tag-based architecture aggregation for Phase 4+.
2. **Nightly run frequency**: Per-component nightly or consolidated runs?
3. **Coverage thresholds**: Should we enforce 80% coverage gates at the TheRock level or per-component?
4. **Test naming standardization**: Should we enforce `test_*` naming convention across all projects?
5. **Artifact retention**: How long should coverage artifacts be retained?
6. **Report aggregation**: Should TheRock provide a unified coverage dashboard across all components?
7. **Architecture-specific detection**: How to identify architecture-specific code paths? Requires team input on refactoring needs.
8. **Baseline initialization**: Self-healing all-arch coverage vs manual initialization vs graceful degradation?
9. **Multi-arch nightly cadence**: Round-robin architectures or weekly/monthly full sweeps?

## Revision History

- 2026-07-28: jorobbin: Initial version
- 2026-07-30: jorobbin: Clarified downstream independence, llvm-cov wildcard limitations, added therock_configure_coverage.py design, test schema details, CI workflow integration, codecov.io integration, resource allocation, and failure handling
- 2026-08-20: jorobbin: Added phased multi-architecture coverage strategy to address architecture-specific code coverage needs
