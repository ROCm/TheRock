---
author: John Robbins (jorobbin)
created: 2026-07-28
modified: 2026-08-24
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

**Problem:**
Projects currently use generic `-DBUILD_CODE_COVERAGE` flag. If multiple projects use the same flag name, enabling coverage for one project accidentally enables it for others, causing unwanted instrumentation and cross-contamination.

**Solution:**
Use project-specific coverage flags that match the project's logical target name:
- `-DROCPRIM_ENABLE_COVERAGE=ON`
- `-DROCFFT_ENABLE_COVERAGE=ON` (matches casing of rocFFT target)
- `-DHIPBLASLT_ENABLE_COVERAGE=ON`
- `-DHIPDNN_ENABLE_COVERAGE=ON` (uppercase conversion of hipDNN)

**Passthrough Options:**

TheRock provides several mechanisms to pass coverage flags from super-build to subprojects:

**Option 1: Direct project-specific flags (implemented)**
```cmake
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DROCFFT_ENABLE_COVERAGE=ON \
  -DHIPBLASLT_ENABLE_COVERAGE=ON
```

Implementation in `cmake/therock_subproject.cmake`:
```cmake
# Passthrough -D<PROJECT_NAME>_ENABLE_COVERAGE=ON to the subproject
# Convert logical target name to uppercase (e.g., hipDNN -> HIPDNN_ENABLE_COVERAGE)
string(TOUPPER "${_logical_target_name}" _coverage_project_name)
set(_coverage_var_name "${_coverage_project_name}_ENABLE_COVERAGE")

if(DEFINED ${_coverage_var_name})
  list(APPEND _cmake_args "-D${_coverage_var_name}=${${_coverage_var_name}}")
endif()
```

**Option 2: Existing {project}_CMAKE_ARGS mechanism**
```cmake
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DhipDNN_CMAKE_ARGS="-DHIPDNN_ENABLE_COVERAGE=ON"
```

**Option 3: Dedicated coverage project list**
```cmake
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_COVERAGE_PROJECTS="rocFFT;hipBLASLt;rocPRIM"
```

Would require additional logic to convert project list to individual `<PROJECT>_ENABLE_COVERAGE` flags.

**Option 4: Comma-separated list (alternative syntax)**
```cmake
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DENABLE_COVERAGE="rocFFT,hipBLASLt,rocPRIM"
```

Also supports special values `none` or `all`.

**Coverage-for-all flags:**

For nightly runs and comprehensive coverage testing, TheRock provides flags to enable coverage for all components in specific groups:

```cmake
# Enable coverage for all rocm-libraries components
-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON

# Enable coverage for all rocm-systems components
-DTHEROCK_COVERAGE_ROCM_SYSTEMS_ALL=ON

# Enable coverage for all components (both libraries and systems)
-DTHEROCK_COVERAGE_ALL=ON
```

These flags are independent:
- `THEROCK_COVERAGE_ROCM_LIBRARIES_ALL` only instruments math-libs, ml-libs, cv-libs, etc.
- `THEROCK_COVERAGE_ROCM_SYSTEMS_ALL` only instruments base/rocm-systems components
- `THEROCK_COVERAGE_ALL` is equivalent to enabling both

This separation allows nightly runs to instrument all libraries without instrumenting the entire ROCm stack, reducing build time and resource usage when system-level coverage is not needed.

**CI Workflow Integration Consideration:**

CI workflows detect changed projects and need a simple way to pass project names to TheRock. The project names may be in various cases (hiprand, rocFFT, hipBLASLt) depending on the source. TheRock should accept project names case-insensitively and handle the conversion internally.

**Recommended approach: Option 3 or 4 (list-based)**

Options 3 and 4 are better suited for CI integration because:
- Accept project names as values, not as formatted arguments
- No formatting burden on the caller - TheRock handles case conversion
- Single flag instead of multiple project-specific flags
- Easier to programmatically construct from CI-detected changes

Example implementation in TheRock CMake:
```cmake
# Option 3: -DTHEROCK_COVERAGE_PROJECTS="rocfft;hiprand;hipblaslt"
if(DEFINED THEROCK_COVERAGE_PROJECTS)
  foreach(_proj IN LISTS THEROCK_COVERAGE_PROJECTS)
    string(TOUPPER "${_proj}" _proj_upper)
    set(${_proj_upper}_ENABLE_COVERAGE ON)
  endforeach()
endif()
```

This approach:
- Accepts any case: "hiprand", "hipRAND", "HIPRAND" all work
- Converts to correct uppercase flag: `HIPRAND_ENABLE_COVERAGE=ON`
- Simplifies caller code - just pass the project name

**Option 1 still useful for manual builds:**
Direct project flags remain valuable for developers doing local coverage builds on specific projects without needing to construct lists.

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
   python build_tools/github_actions/post_stage_upload.py
   ```

#### Test Phase (PR Coverage)

**Note:** This describes PR coverage testing. Nightly coverage uses a different approach - see Nightly Runs section.

For PR coverage, modify existing test workflows:

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

4. **Upload profraw files as artifacts** (conditional on `coverage_enabled`):
   - Each test shard uploads its profraw files as workflow artifacts
   - Do NOT merge profraw on test nodes - aggregation happens later
   - Artifact name: `coverage-profraw-${COMPONENT_NAME}-shard${SHARD_INDEX}`

5. **Coverage aggregation job** (separate node that runs after all test shards complete):
   ```bash
   # Download profraw artifacts from all shards
   # (GitHub Actions: actions/download-artifact downloads all matching artifacts)
   
   # Merge profraw from ALL shards
   cd ${GITHUB_WORKSPACE}/coverage-report
   ${GITHUB_WORKSPACE}/build/lib/llvm/bin/llvm-profdata merge -sparse \
     -o coverage.profdata \
     shard1/*.profraw shard2/*.profraw shard3/*.profraw shard4/*.profraw
   
   # Generate final coverage report
   ${GITHUB_WORKSPACE}/build/lib/llvm/bin/llvm-cov export \
     -object ${GITHUB_WORKSPACE}/build/lib/lib${COMPONENT_NAME}.so \
     -instr-profile=coverage.profdata --format=lcov > coverage.info
   ```
   
   **Why aggregation is required:**
   Tests are sharded across multiple nodes for performance. Without aggregation:
   - Shard 1 only covers tests 1-25 (25% coverage)
   - Shard 2 only covers tests 26-50 (25% coverage)
   - Shard 3 only covers tests 51-75 (25% coverage)
   - Shard 4 only covers tests 76-100 (25% coverage)
   
   Aggregation merges profraw from all shards to produce complete 100% coverage report.

**Artifact naming conventions:**
- Coverage builds require unique artifact names to avoid confusion with regular builds
- Add `-coverage` suffix to artifact names (similar to `-asan` for ASAN builds)
- Example: `hiprand-gfx942-coverage.tar.gz` vs `hiprand-gfx942.tar.gz`
- Prevents developers/CI from accidentally downloading instrumented coverage artifacts when expecting regular builds
- Artifacts downloaded via `fetch_artifact_args` mechanism with coverage-specific paths
- Profraw files: `${TEST_COMPONENT}-shard${matrix.shard}-%p-%m.profraw`
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
- Build nodes per coverage-enabled project:
  - Coverage builds can leverage multi-arch CI pipeline structure with build stages
  - May use multiple build nodes in parallel (e.g., different stages running concurrently)
  - Can reuse prebuilt artifacts for unmodified dependencies/stages
  - Exact staging depends on whether coverage build uses single-stage or multi-stage approach
- Multiple test nodes per coverage-enabled project (tests are sharded for performance)
  - Number of test shards determined by project test suite size
  - Example: 4 shards = 4 parallel test nodes
  - Each shard runs subset of tests and generates profraw files
- One aggregation node per coverage-enabled project
  - Runs after all test shards complete
  - Downloads profraw artifacts from all test shards
  - Merges profraw files and generates final coverage report
  - Uploads to codecov.io
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

### Critical Dependency: amd-llvm

**Problem:**
Code coverage infrastructure is critically dependent on amd-llvm's code coverage support. Changes to amd-llvm's profiling implementation can break the entire coverage pipeline across all ROCm components.

**Historical incident:**
A change in ROCm 7.14 broke the code coverage process, preventing coverage report generation for all projects. This failure was not caught before merge, blocking coverage reporting until the issue was diagnosed and fixed.

**Required safeguard:**
amd-llvm changes must be gatekept with code coverage smoke tests before merge:
- Smoke test builds a sample instrumented project with coverage flags enabled
- Runs minimal test suite to generate profraw files
- Verifies profraw merging with `llvm-profdata merge`
- Verifies coverage report generation with `llvm-cov export`
- Test must pass before amd-llvm changes are merged

Without this safeguard, amd-llvm regressions can break coverage reporting for weeks across the entire ROCm ecosystem.

### Specialized Coverage Scenarios

Coverage testing requires different strategies for three distinct types of code paths that cannot be covered by default single-architecture testing:

1. **Multi-Architecture Code**: Architecture-specific optimizations (e.g., gfx90a vs gfx942)
2. **Multi-GPU Code**: Code paths requiring multiple GPUs (parallel operations, multi-GPU algorithms)
3. **Mock-Required Code**: Error handling paths that need upstream dependency error injection

#### Multi-Architecture Coverage Strategy

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

**Phased Rollout Strategy:**

Nightly coverage follows a phased approach with increasing sophistication:

**Phase 1: Full coverage on default architecture**
- Build entire instrumented ROCm stack once (`-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON` or `-DTHEROCK_COVERAGE_ALL=ON`)
- Run coverage for every component regardless of changes (sanity check + baseline)
- Execute on single default architecture (gfx942 or gfx950)

**Phase 2: Change-based selection**
- Detect what changed on develop branch since last nightly run
- Run coverage only for components with changes
- Reduces nightly resource usage while maintaining coverage trends

**Phase 3: Multi-architecture coverage**
- Expand to multiple architectures for architecture-specific code
- May use round-robin (different architectures on different nights) OR weekly/monthly full sweeps
- Requires architecture-specific change detection (likely comes after PR multi-arch support)

**Hybrid Artifact Approach (Nightly Only):**

Unlike PR coverage builds (which only instrument changed projects), nightly runs instrument the entire ROCm stack but test components separately:

1. **Single instrumented stack build:**
   ```bash
   cmake -B build -GNinja \
     -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
     -DCOMPILER_RT_BUILD_PROFILE_ROCM=ON \
     -DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS=OFF \
     -DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON
   ninja -C build
   ```
   
   Produces instrumented artifacts for all components at once.

2. **Separate test jobs per component:**
   - Each component gets its own test job (hipRAND, rocFFT, rocBLAS, etc.)
   - Test job pulls:
     - **Instrumented artifact** for component under test (from nightly instrumented build)
     - **Non-instrumented dependencies** from most recent regular build (pre-built artifacts)
   - This hybrid approach isolates coverage measurement to one component at a time
   - Maintains per-project isolation while amortizing instrumented build cost

**Why this approach?**
- **Single instrumented build**: Amortizes build cost - don't rebuild entire stack per component
- **Separate test jobs**: Maintains per-project isolation - only one instrumented component per test run
- **Hybrid artifacts**: Non-instrumented dependencies avoid profraw contamination from upstream code
- **Nightly-specific**: PRs only instrument changed projects; nightlies instrument everything for comprehensive baseline

**Resource implications:**
- One large instrumented build (entire ROCm stack) per nightly run
- N separate test jobs (one per component)
- Each test job uses hybrid artifacts: one instrumented + rest non-instrumented
- Relies on codecov.io (or chosen platform) for coverage aggregation across components

#### Multi-GPU Coverage Strategy

**Problem:**
Some code paths only execute when multiple GPUs are available (e.g., multi-GPU GEMM operations, distributed algorithms, peer-to-peer memory transfers). Default single-GPU coverage testing cannot exercise these paths.

**Detection:**
- Identify PRs that modify multi-GPU specific code
- Requires code organization/annotation to distinguish multi-GPU paths
- Similar to architecture-specific detection but for GPU count

**Testing approach:**
- Default coverage runs on single-GPU nodes (majority of code)
- When multi-GPU code changes detected:
  - Trigger coverage job on multi-GPU node
  - Aggregate multi-GPU report with default single-GPU report
- Multi-GPU testing likely limited to specific components (not all projects have multi-GPU code)

**Resource implications:**
- Multi-GPU nodes are scarcer than single-GPU nodes
- May require dedicated multi-GPU coverage node pool
- Nightly multi-GPU coverage sweeps to maintain baseline

#### Mock-Based Coverage Strategy

**Problem:**
Error handling code for upstream dependency failures cannot be covered without error injection. Example: Component A calls Component B's API - to cover A's error handling when B fails, we need to inject failures into B. However:
- Cannot safely inject errors into real upstream components during coverage testing
- Instrumenting upstream components provides no value for downstream coverage
- Need controlled error injection to trigger error paths

**Solution: Mocking upstream dependencies**
- Create mock implementations of upstream APIs that can inject controlled errors
- Mock testing exercises error handling paths in component under test
- **Does NOT require instrumentation of upstream components** - only the component under test is instrumented
- Mocks simulate failures without affecting real dependency behavior

**Integration with default architecture:**
- Mock-based coverage tests can run on default architecture (gfx942/gfx950)
- No special hardware requirements - mocks are software-level abstractions
- Can be integrated into Phase 1 (no multi-arch dependency)

**Detection:**
- Identify components with error handling for upstream dependencies
- May require component teams to flag mock-requiring code paths
- Build/test matrix determines which components need mock coverage

**Open questions:**
- Should all components provide mock implementations for error injection?
- How to maintain mocks as upstream APIs evolve?
- Should mock coverage be mandatory or optional?

## Open Questions

1. **Codecov.io vs. alternatives**: Should we standardize on codecov.io or evaluate other platforms? Must support tag-based architecture aggregation for Phase 4+.
2. **Nightly run frequency**: Per-component nightly or consolidated runs?
3. **Coverage thresholds**: Should we enforce 80% coverage gates at the TheRock level or per-component?
4. **Test naming standardization**: Should we enforce `test_*` naming convention across all projects?
5. **Artifact retention**: How long should coverage artifacts be retained?
6. **Report aggregation**: Should TheRock provide a unified coverage dashboard across all components?
7. **Architecture-specific detection**: How to identify architecture-specific code paths? Requires team input on refactoring needs.
8. **Multi-GPU detection**: How to identify multi-GPU specific code paths? Similar refactoring/annotation needs.
9. **Mock coverage mandate**: Should all components provide mocks for upstream error injection? Mandatory or optional?
10. **Baseline initialization**: Self-healing all-arch coverage vs manual initialization vs graceful degradation?
11. **Multi-arch nightly cadence**: Round-robin architectures or weekly/monthly full sweeps?
12. **Multi-GPU node allocation**: Dedicated pool vs shared with other multi-GPU workloads?

## Revision History

- 2026-07-28: jorobbin: Initial version
- 2026-07-30: jorobbin: Clarified downstream independence, llvm-cov wildcard limitations, added therock_configure_coverage.py design, test schema details, CI workflow integration, codecov.io integration, resource allocation, and failure handling
- 2026-08-20: jorobbin: Added phased multi-architecture coverage strategy; distinguished multi-arch, multi-GPU, and mock-based coverage scenarios; documented coverage flag passthrough options with case-insensitive project name handling; updated post_build_upload.py to post_stage_upload.py; required unique -coverage suffix for coverage artifacts; documented profraw aggregation for sharded tests and multi-node builds
- 2026-08-24: jorobbin: Documented amd-llvm dependency and smoke test requirements; clarified profraw naming patterns and aggregation node separation; added nightly coverage phased rollout (full→change-based→multi-arch); documented hybrid artifact approach for nightly (single instrumented build + separate per-component tests with non-instrumented dependencies); added coverage-for-all flags for rocm-libraries, rocm-systems, and all components
