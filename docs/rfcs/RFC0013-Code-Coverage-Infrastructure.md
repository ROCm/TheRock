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
- **The `-object` flag does not accept wildcards** - cannot use `-object test_*`, must expand to `-object test_1 -object test_2 ... -object test_n`

**Why test_categories_coverage.yaml may be needed:**
Because `llvm-cov show -object` does not accept wildcard patterns, we must explicitly list each test binary. This requires either:
1. **Separate test_categories_coverage.yaml** that explicitly lists all test binaries for coverage processing
2. **Declarative approach with standard naming** - if tests follow `test_*` convention:
   ```bash
   # Generate -object flags for all tests
   llvm-cov show $(for test in $(ls tests/); do echo -n "-object $test "; done) -instr-profile=coverage.profdata
   ```

**Proposed standardization:**
- Standardize test naming: `test_<test_name>` across all components
- Allows declarative generation of `-object` flags at the top level
- With standard naming, may be able to reuse test_categories.yaml structure without a separate coverage variant
- **This remains an open discussion topic** - the tradeoffs between explicit listing vs. declarative generation need team input

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

## Open Questions

1. **Codecov.io vs. alternatives**: Should we standardize on codecov.io or evaluate other platforms?
2. **Nightly run frequency**: Per-component nightly or consolidated runs?
3. **Coverage thresholds**: Should we enforce 80% coverage gates at the TheRock level or per-component?
4. **Multi-arch coverage**: Is there any benefit to running coverage on multiple GPU architectures?
5. **Test naming standardization**: Should we enforce `test_*` naming convention across all projects?
6. **Artifact retention**: How long should coverage artifacts be retained?
7. **Report aggregation**: Should TheRock provide a unified coverage dashboard across all components?

## Revision History

- 2026-07-28: jorobbin: Initial version
- 2026-07-30: jorobbin: Clarified downstream independence and llvm-cov wildcard limitations
