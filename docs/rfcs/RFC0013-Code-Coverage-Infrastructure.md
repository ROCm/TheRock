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
2. Run tests using the Test Filter Standard (test_categories.yaml)
3. Minor modifications to test_categories_coverage.yaml:
   - Add `llvm_profile_file:` key to control profraw output location
   - Modify test_runner.py to handle coverage mode

### Problematic: Header-Only Static Libraries
Some components are header-only static libraries (e.g., rocPRIM, hipCUB) and do not produce a simple `.so` file for `llvm-cov show -object <lib>.so`.

**For these projects:**
- Must instrument the test binaries themselves (headers are compiled into the test binary)
- Must use `--ignore-filename-regex` to exclude test code from the report
- Requires explicit reference to test binaries rather than library objects

**Proposed standardization:**
- Standardize test naming: `test_<test_name>`
- Allows declarative control of `--ignore-filename-regex` at the top level
- Reuses existing test_categories.yaml structure without modifications

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
- **Architecture**: Single architecture (typically gfx942)
  - Minimal variation between architectures
  - Negligible gain from multi-arch coverage

#### Build Phase
1. Determine coverage flag from PR changes:
   ```bash
   python build_tools/therock_configure_coverage.py
   ```
   Output: `-D<PROJECT_NAME>_ENABLE_COVERAGE=ON`

2. Run coverage build:
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
- Add `coverage_enabled` passthrough flag

**`therock-ci-test-components.yml`:**
- Condition on `coverage_enabled` flag
- Download artifacts
- Run tests with `LLVM_PROFILE_FILE` environment variable set
- Execute post-test coverage commands:
  ```bash
  llvm-profdata merge <profraw_dir>/*.profraw -o coverage.profdata
  llvm-cov show -object <lib_or_test_binary> -instr-profile=coverage.profdata
  llvm-cov export -format=lcov -object <lib_or_test_binary> -instr-profile=coverage.profdata > coverage.lcov
  ```
- Upload to codecov.io

### Nightly Runs

**Recommendation:** Achieve parity with Math CI:
- Separate nightly build/test for each component
- Rely on codecov.io (or chosen platform) for aggregation
- Avoids resource stress from full-ROCm coverage builds

**Open question:** Can we consolidate to fewer nightly runs without resource issues?

## Implementation Plan

### Phase 1: CMake Infrastructure
1. Add component-specific coverage flags to each project's CMakeLists.txt
2. Add TheRock passthrough for coverage flags
3. Implement `therock_configure_coverage.py` script

### Phase 2: Test Runner Updates
1. Add `llvm_profile_file:` support to test_categories.yaml schema
2. Update test_runner.py to handle coverage mode
3. Standardize test naming where needed (header-only libraries)

### Phase 3: CI Workflow
1. Create new `therock-ci-coverage.yml` workflow
2. Add coverage job to nightly runs
3. Integrate codecov.io upload

### Phase 4: Validation
1. Validate coverage reports for sample projects (rocFFT, rocBLAS, rocPRIM)
2. Verify no cross-contamination between components
3. Confirm performance isolation from pre-checkin builds

## Open Questions

1. **Codecov.io vs. alternatives**: Should we standardize on codecov.io or evaluate other platforms?
2. **Nightly run frequency**: Per-component nightly or consolidated runs?
3. **Coverage thresholds**: Should we enforce 80% coverage gates at the TheRock level or per-component?
4. **Multi-arch coverage**: Is there any benefit to running coverage on multiple GPU architectures?
5. **Test naming standardization**: Should we enforce `test_*` naming convention across all projects?
6. **Artifact retention**: How long should coverage artifacts be retained?
7. **Report aggregation**: Should TheRock provide a unified coverage dashboard across all components?

## Areas Needing More Detail

1. **therock_configure_coverage.py implementation**:
   - How does it detect which project changed in a PR?
   - How does it handle multi-component PRs?
   - What happens if changes span multiple projects?

2. **Test runner modifications**:
   - Exact schema changes to test_categories.yaml
   - How to handle mixed library types (shared vs static vs header-only) in a single coverage run
   - Error handling when profraw files are missing

3. **Header-only library handling**:
   - Specific regex patterns for `--ignore-filename-regex`
   - How to reference test binaries in llvm-cov commands
   - Directory structure expectations

4. **CI workflow integration**:
   - Exact modifications to therock-ci-test-packages.yml
   - Exact modifications to therock-ci-test-components.yml
   - Artifact naming conventions for coverage builds
   - Upload/download artifact paths

5. **Codecov.io configuration**:
   - Repository setup
   - Token management
   - PR comment configuration
   - Failure threshold configuration

6. **Resource allocation**:
   - Build node sizing for coverage builds
   - Test node sizing for coverage test runs
   - Expected runtime for coverage builds vs normal builds
   - Storage requirements for coverage artifacts

7. **Failure handling**:
   - What happens if coverage generation fails?
   - Should coverage failures block PR merges?
   - How to handle flaky coverage tests?

8. **Migration path**:
   - How to migrate existing Math CI coverage infrastructure?
   - Can we maintain both during transition?
   - What's the rollout plan across components?

## Revision History

- 2026-07-28: jorobbin: Initial version
