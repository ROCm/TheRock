# Coverage Workflow Implementation: Work Summary and Roadblocks

## Executive Summary

This document summarizes the comprehensive work completed on implementing code coverage in TheRock's GitHub Actions CI workflows, covering both:
1. **TheRock CI Integration** - Workflow orchestration, matrix generation, and build/test job separation
2. **Coverage Process Implementation** - LLVM coverage tooling, test execution, and report generation

The work follows two parallel tracks that must converge, but encountered critical architectural roadblocks requiring a fundamental redesign.

**Target Audience:** Developers continuing this work (both human and AI), technical leadership reviewing architecture decisions.

**Branch Context:**
- TheRock: `enable-codecov` (commit: `6afcd5ac52`)
- rocm-libraries: `therock-codecov` (commit: `bf8ad30220`)

---

## Part 1: TheRock CI Integration Work

### Objective

Integrate code coverage as a first-class workflow in TheRock's CI system, running coverage builds/tests only for changed components with coverage-enabled projects.

### Architecture Overview

TheRock CI uses a **matrix-based orchestration pattern** where the main workflow (`therock-ci.yml`) calls reusable workflows for different platforms and configurations:

```
therock-ci.yml (orchestrator)
├─ setup job → determines what to build/test
│  ├─ therock_configure_ci.py → projects for Linux/Windows
│  ├─ therock_configure_coverage.py → projects for coverage
│  └─ fetch_package_targets.py → runner labels per GPU family
│
├─ therock-ci-linux.yml (reusable, matrix per project × GPU)
├─ therock-ci-windows.yml (reusable, matrix per project × GPU)
└─ therock-ci-coverage.yml (NEW - coverage-specific workflow)
```

### Work Completed: TheRock Repository

#### 1. CMake Coverage Flag Passthrough (Commit: `341808faae`)

**File:** `cmake/therock_subproject.cmake`

**Change:** Added passthrough mechanism for `<PROJECT>_ENABLE_COVERAGE` CMake flags from super-project to subprojects.

```cmake
# Passthrough -D<PROJECT_NAME>_ENABLE_COVERAGE=ON to the subproject
set(_coverage_arg)
if(DEFINED ${_logical_target_name}_ENABLE_COVERAGE)
  set(_coverage_arg "-D${_logical_target_name}_ENABLE_COVERAGE=${${_logical_target_name}_ENABLE_COVERAGE}")
endif()

add_custom_command(
  ...
  COMMAND ${CMAKE_COMMAND}
    ${_cmake_args}
    ${_coverage_arg}  # <-- NEW
  ...
)
```

**Impact:** Allows workflows to enable coverage per-component via CMake flags like `-DHIPDNN_ENABLE_COVERAGE=ON`.

**Status:** ✅ Complete and working

#### 2. LLVM Coverage Tools Build (Commit: `1ce765508b`)

**File:** `compiler/CMakeLists.txt`

**Change:** Enabled `llvm-cov` and `llvm-profdata` tools in the amd-llvm build.

```cmake
# Coverage tools (needed for code coverage builds)
-DLLVM_TOOL_LLVM_COV_BUILD=ON
-DLLVM_TOOL_LLVM_PROFDATA_BUILD=ON
```

**Impact:** Makes LLVM coverage tools available in `dist/rocm/lib/llvm/bin/` for projects to use in coverage targets.

**Status:** ✅ Complete and working

#### 3. Container Build Dependencies (Commit: `295f365101`)

**File:** `compiler/pre_hook_amd-llvm.cmake`

**Change:** Added `rocm-llvm-dev` package to build container to ensure LLVM development headers are available.

**Impact:** Required for components that need LLVM headers during coverage-instrumented builds.

**Status:** ✅ Complete and working

#### 4. LLVM Tools Export (Commits: `f70ca7b755`, `6afcd5ac52`)

**Files:** `compiler/pre_hook_amd-llvm.cmake`, build scripts

**Change:** Exported `llvm-profdata`, `llvm-cov`, and `llvm-cxxfilt` as required LLVM tools.

**Impact:** Ensures coverage tools are included in TheRock build artifacts and available in test environments.

**Status:** ✅ Complete and working

### Work Completed: rocm-libraries Repository

#### 1. Coverage Workflow Creation (Initial: `12d9d71263`, Final: `bf8ad30220`)

**File:** `.github/workflows/therock-ci-coverage.yml`

**Purpose:** Reusable workflow for coverage builds and tests, accepting:
- `project_name` - Uppercase project name (e.g., "HIPDNN")
- `cmake_target` - CMake target to build (e.g., "hipDNN")
- `build_dir` - Build subdirectory path (e.g., "ml-libs/hipDNN/build")
- `amdgpu_families` - GPU architecture (e.g., "gfx94X")
- `test_runs_on` - Runner label for test job

**Structure:**
```yaml
jobs:
  therock-coverage-build:
    runs-on: azure-linux-scale-rocm
    container: manylinux build image
    steps:
      - Configure with -D${project_name}_ENABLE_COVERAGE=ON
      - Build therock-archives and therock-dist
      - Upload artifacts to S3

  therock-coverage-test:
    runs-on: ${{ inputs.test_runs_on }}  # GPU runner
    container: no_rocm_image (test environment)
    needs: [therock-coverage-build]
    steps:
      - Download artifacts from S3
      - Reconfigure CMake (recreate build tree)
      - Run tests: cmake --build . --target ${cmake_target}+test
      - Generate reports: cmake --build . --target coverage
```

**Status:** 🔴 **BLOCKED** (see Roadblocks section)

#### 2. Coverage Project Configuration (Commit: `549974c348` and earlier)

**File:** `.github/scripts/therock_configure_coverage.py`

**Purpose:** Determines which projects should run coverage based on:
1. Changed subtrees (via `pr_detect_changed_subtrees.py`)
2. Coverage-enabled project mapping (`COVERAGE_PROJECT_METADATA`)

```python
COVERAGE_PROJECT_METADATA = {
    "hipdnn": ("hipDNN", "ml-libs/hipDNN"),
    "prim": ("rocPRIM", "math-libs/PRIM"),
}
```

**Logic:**
- Only runs coverage for projects with changed code (efficiency optimization)
- Maps subtree paths → project keys → CMake metadata
- Outputs matrix: `[{project_name, cmake_target, build_dir, cmake_options}, ...]`

**Integration Point:** Called from `therock-ci.yml` setup job:

```yaml
- name: Determine Linux projects for coverage (changed subtrees only)
  id: coverage_projects
  run: python .github/scripts/therock_configure_coverage.py
```

**Status:** ✅ Complete and working

#### 3. Main Workflow Integration (Commit: `b256e80121` and iterations)

**File:** `.github/workflows/therock-ci.yml`

**Changes:**

1. **Added coverage_projects output to setup job:**
```yaml
outputs:
  coverage_projects: ${{ steps.coverage_projects.outputs.coverage_projects }}
  coverage_package_targets: ${{ steps.configure_coverage_targets.outputs.package_targets }}
```

2. **Added fetch coverage targets step (gfx94X only):**
```yaml
- name: Fetch coverage targets (gfx94X only)
  env:
    AMDGPU_FAMILIES: 'gfx94X'
  id: configure_coverage_targets
  run: python ./TheRock/build_tools/github_actions/fetch_package_targets.py
```

3. **Added coverage job with matrix strategy:**
```yaml
therock-ci-coverage:
  name: Coverage (${{ matrix.projects.project_name }} | ${{ matrix.target_bundle.amdgpu_family }})
  needs: setup
  if: ${{ needs.setup.outputs.coverage_projects != '[]' }}
  strategy:
    fail-fast: false
    matrix:
      projects: ${{ fromJSON(needs.setup.outputs.coverage_projects) }}
      target_bundle: ${{ fromJSON(needs.setup.outputs.coverage_package_targets) }}
  uses: ./.github/workflows/therock-ci-coverage.yml
  with:
    cmake_options: ${{ matrix.projects.cmake_options }}
    project_name: ${{ matrix.projects.project_name }}
    cmake_target: ${{ matrix.projects.cmake_target }}
    build_dir: ${{ matrix.projects.build_dir }}
    amdgpu_families: ${{ matrix.target_bundle.amdgpu_family }}
    test_runs_on: ${{ matrix.target_bundle.test_machine }}
```

**Status:** ✅ Complete and working (orchestration layer functional, execution blocked)

#### 4. Iterative Debugging Work (Commits: `667358d96c` through `bf8ad30220`)

Resolved multiple integration issues:

1. **Runner label alignment** - Fixed mismatched GPU runner labels (mi325 → gfx94X-dcgpu)
2. **Container image updates** - Updated build/test container SHAs for DVC compatibility
3. **CMake preset migration** - Changed `github-oss-presubmit` → `github-oss-dev`
4. **Test environment setup** - Inlined action steps to avoid path resolution issues
5. **Build directory handling** - Multiple attempts at preserving/recreating CMake build tree

### Design Patterns and Decisions

#### Matrix Strategy

Coverage jobs use the same **matrix multiplication pattern** as standard CI:

```
projects × target_bundles = job matrix

Example:
  projects: [hipdnn, prim]
  target_bundles: [{amdgpu_family: gfx94X, test_machine: linux-gfx942-1gpu-ossci-rocm}]
  
  Result: 2 jobs
    - Coverage (HIPDNN | gfx94X)
    - Coverage (PRIM | gfx94X)
```

**Rationale:** Consistent with existing CI patterns, leverages existing infrastructure.

#### Changed-Subtrees-Only Filtering

Coverage only runs for **components with code changes** (not all coverage-enabled projects on every PR).

**Mechanism:** `therock_configure_coverage.py` uses the same `pr_detect_changed_subtrees` logic as regular CI.

**Rationale:** Coverage builds are expensive (instrumentation overhead + profiling); only run when code changes.

**Trade-off:** May miss coverage regressions in unchanged downstream dependencies.

#### Build/Test Job Separation

Coverage workflow splits into two jobs:

```
Build Job (no GPU)              Test Job (with GPU)
├─ azure-linux-scale-rocm       ├─ linux-gfx942-1gpu-ossci-rocm
├─ manylinux container          ├─ no_rocm_image container
├─ Configure + Build            ├─ Download artifacts
├─ Create .tar.xz artifacts     ├─ Reconfigure CMake
└─ Upload to S3                 ├─ Run tests
                                ├─ Generate coverage
                                └─ Upload reports
```

**Rationale:**
- GPU runners are scarce/expensive - don't waste on compilation
- Matches standard `therock-ci-linux.yml` pattern
- Build artifacts can be reused across multiple test shards (future)

**Status:** ✅ Pattern implemented correctly, but execution blocked (see Roadblocks)

---

## Part 2: Coverage Process Implementation

### Objective

Generate code coverage reports (text, HTML, LCOV) using LLVM instrumentation-based coverage for rocm-libraries projects.

### LLVM Coverage Workflow

All components follow this standard pattern:

```bash
# 1. Build with instrumentation flags
cmake -DBUILD_CODE_COVERAGE=ON
  # Sets: -fprofile-instr-generate -fcoverage-mapping

# 2. Run tests with LLVM_PROFILE_FILE environment
export LLVM_PROFILE_FILE="coverage-report/profraw/%m.profraw"
./test_binary  # or ctest

# 3. Merge profraw files into profdata
llvm-profdata merge -sparse -o coverage.profdata coverage-report/profraw/*.profraw

# 4. Generate reports
llvm-cov report <-object args> -instr-profile=coverage.profdata
llvm-cov show <-object args> -instr-profile=coverage.profdata --format=html
llvm-cov export <-object args> -instr-profile=coverage.profdata --format=lcov
```

### Analysis of Reference Implementations

Examined three coverage-enabled components to understand patterns:

#### Type 1: Shared Library (.so) - rocFFT

**File:** `projects/rocfft/clients/tests/CMakeLists.txt`

**Coverage Objects:** Single shared library
```cmake
add_custom_target(
  coverage
  COMMAND ${LLVM_COV} report 
    -object ./library/src/librocfft.so
    -instr-profile=./coverage-report/rocfft.profdata
```

**Test Execution:** Direct binary invocation with explicit LLVM_PROFILE_FILE
```cmake
COMMAND ${CMAKE_COMMAND} -E env 
  LLVM_PROFILE_FILE="./coverage-report/profraw/rocfft-coverage_%p.profraw" 
  $<TARGET_FILE:rocfft-test>
```

**Profile Pattern:** `%p` (process ID)

**Key Characteristic:** Clean separation - library is instrumented, tests are not.

#### Type 2: Shared Library (.so) with CTest - hiprand

**File:** `projects/hiprand/test/CMakeLists.txt`

**Coverage Objects:** Single shared library
```cmake
COMMAND ${LLVM_COV} report 
  -object ./library/libhiprand.so
  -instr-profile=./coverage-report/hiprand.profdata
```

**Test Execution:** CTest with `set_tests_properties()`
```cmake
if(BUILD_CODE_COVERAGE)
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "LLVM_PROFILE_FILE=${PROFILE_DIR}/profraw/hiprand-coverage_%m.profraw"
    )
endif()
```

**Profile Pattern:** `%m` (merge/thread ID)

**Key Characteristic:** Uses `set_tests_properties()` for environment propagation instead of direct invocation.

#### Type 3: Header-Only Library - rocPRIM

**File:** `projects/rocprim/test/CMakeLists.txt`

**Coverage Objects:** Multiple test binaries accumulated via GLOBAL property
```cmake
# During test creation:
if(BUILD_CODE_COVERAGE)
  set(ABS_EXE "${EXE_PATH}/${EXE_NAME}")
  set_property(GLOBAL APPEND PROPERTY LLVM_COV_OBJECT_ARGS 
    -object "${ABS_EXE}")
endif()

# In coverage target:
get_property(LLVM_COV_OBJECT_ARGS GLOBAL PROPERTY LLVM_COV_OBJECT_ARGS)
COMMAND ${LLVM_COV} report 
  ${LLVM_COV_OBJECT_ARGS}  # Multiple -object flags
  -instr-profile=./coverage-report/rocprim.profdata
```

**Test Execution:** CTest with `set_tests_properties()`
```cmake
if(BUILD_CODE_COVERAGE)
  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "LLVM_PROFILE_FILE=${PROFILE_DIR}/profraw/rocprim-coverage_%m.profraw"
  )
endif()
```

**Profile Pattern:** `%m` (merge/thread ID)

**Key Characteristic:** No shared library to instrument - test binaries ARE the instrumented code.

### Coverage Implementation Patterns Summary

| Component | Library Type | Coverage Objects | Environment Method | Profile Pattern |
|-----------|--------------|------------------|-------------------|-----------------|
| rocFFT | .so | Single (-object .so) | Direct test invocation | %p (process ID) |
| hiprand | .so | Single (-object .so) | set_tests_properties() | %m (thread ID) |
| rocPRIM | Header-only | Multiple (test binaries) | set_tests_properties() | %m (thread ID) |
| hipDNN | Hybrid | Multiple (.so + tests) | Custom target | %m (thread ID) |

**Common Elements:**
- `BUILD_CODE_COVERAGE` CMake flag enables instrumentation
- `-fprofile-instr-generate -fcoverage-mapping` compilation flags
- `LLVM_PROFILE_FILE` environment variable controls profraw output location
- Coverage targets use project-specific `--ignore-filename-regex` to exclude tests/deps

**Key Divergence: Coverage Object Determination**

Different library architectures require fundamentally different approaches:

1. **.so libraries:** Instrument library, analyze library, single `-object` parameter
2. **Header-only libraries:** Instrument test binaries, analyze all test binaries, multiple `-object` parameters
3. **Hybrid:** Instrument both library and tests, analyze both

**Critical Insight:** The `-object` parameter list **cannot be determined without CMake metadata**.

---

## Part 3: Roadblocks Encountered

### Roadblock #1: Build Tree Not Preserved After Artifact Download

**Issue:** TheRock CI uploads only `.tar.xz` artifact files (installed binaries), not the full CMake build tree.

**Impact:** After downloading artifacts in test job, paths like this don't exist:
```
TheRock/build-coverage/ml-libs/hipDNN/build/
```

Only this exists:
```
TheRock/build-coverage/dist/rocm/  (installed files)
```

**Error Message:**
```
TheRock/build-coverage/ml-libs/hipDNN/build is not a directory
```

**Why It Matters:** CMake coverage targets are defined relative to the build tree:
```cmake
add_custom_target(
  coverage
  COMMAND find ${CMAKE_BINARY_DIR}/coverage-report/profraw -name "*.profraw" ...
  COMMAND ${LLVM_PROFDATA} merge ... -o ${CMAKE_BINARY_DIR}/coverage-report/hipdnn.profdata
  COMMAND ${LLVM_COV} report ... > ${CMAKE_BINARY_DIR}/coverage-report/code_cov.report
```

**Attempted Fix (Commit: `bf8ad30220`):**

Added CMake reconfiguration step in test job to recreate build tree:
```yaml
- name: Reconfigure CMake for coverage
  working-directory: TheRock
  run: python3 build_tools/github_actions/build_configure.py
```

**Why It Failed:** Reconfiguration creates an *empty* build tree with CMakeLists.txt metadata but no built artifacts in expected locations.

**Status:** 🔴 **Fundamental architectural mismatch**

### Roadblock #2: Coverage Object Determination Requires CMake Metadata

**Critical Discovery:** "Some libraries have static headers while others use .so... this makes it such that we kind of have to rebuild the CMake tree." (User quote)

**The Problem:**

Cannot determine which `-object` parameters to pass to `llvm-cov` without CMake build metadata.

**Examples:**

**hipDNN (hybrid):** Needs 7 object parameters
```cmake
OBJECTS="$OBJECTS -object lib/libhipdnn_backend.so"
OBJECTS="$OBJECTS -object bin/hipdnn_backend_tests"
OBJECTS="$OBJECTS -object bin/hipdnn_frontend_tests"
OBJECTS="$OBJECTS -object bin/hipdnn_public_backend_tests"
OBJECTS="$OBJECTS -object bin/hipdnn_data_sdk_tests"
OBJECTS="$OBJECTS -object bin/hipdnn_plugin_sdk_tests"
OBJECTS="$OBJECTS -object bin/hipdnn_test_sdk_tests"
```

**rocPRIM (header-only):** Needs dynamic list from GLOBAL property
```cmake
get_property(LLVM_COV_OBJECT_ARGS GLOBAL PROPERTY LLVM_COV_OBJECT_ARGS)
# Contains: -object <path1> -object <path2> ... -object <pathN>
```

**rocFFT (.so):** Needs single object
```cmake
-object ./library/src/librocfft.so
```

**Why Hardcoding Doesn't Work:**

1. List is project-specific and changes as code evolves
2. Header-only libraries accumulate test binaries dynamically during CMake configuration
3. No way to know which tests exist without reading CMake cache/properties
4. Cannot distinguish header-only from .so-based libraries without build metadata

**Why Artifacts Don't Contain This Information:**

TheRock's artifact system uploads:
- Installed binaries (`.tar.xz` of `dist/rocm/`)
- Metadata manifests (component versions, dependencies)
- **NOT** CMake build metadata (CMakeCache.txt, GLOBAL properties, target definitions)

**Status:** 🔴 **Show stopper** - Cannot proceed without CMake metadata

### Roadblock #3: LLVM_PROFILE_FILE Environment Propagation with CTest

**Issue:** "I have not had any luck with the LLVM_PROFILE_FILE being properly propagated to the tests" (User quote)

**Expected Behavior:**
```bash
export LLVM_PROFILE_FILE="coverage-%m.profraw"
ctest -L full
# Should produce: coverage-<id1>.profraw, coverage-<id2>.profraw, ...
```

**Actual Behavior:**
Only `default.profraw` generated (all tests clobber the same file).

**Root Cause:** CTest doesn't reliably inherit environment variables to child test processes.

**Reference Solution (from hiprand):**

Environment must be set via `set_tests_properties()` at CMake configuration time:
```cmake
if(BUILD_CODE_COVERAGE)
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "LLVM_PROFILE_FILE=${PROFILE_DIR}/profraw/hiprand-coverage_%m.profraw"
    )
endif()
```

**Why Inline Bash Export Doesn't Work:**

CTest spawns test processes via its own executor which doesn't inherit shell environment in all cases.

**Pattern Differences:**
- `%m` = Merge/thread ID (more reliable for parallel tests) - used by hiprand, rocPRIM, hipDNN
- `%p` = Process ID - used by rocFFT (direct test invocation, not ctest)

**Implication:** Must set environment at **CMake configure time** (build job), not at test runtime (test job).

**Status:** 🔴 **Architectural constraint** - Requires CMake reconfiguration in environment where tests will run

### Roadblock #4: Two-Stage Pipeline Requires Full Build Tree in Test Job

**Fundamental Tension:**

```
Build Job Constraints          vs.    Test Job Requirements
─────────────────────                 ────────────────────────
• No GPU available                    • GPU required for tests
• CMake metadata generated            • CMake metadata required
• Build tree exists                   • Build tree required for coverage targets
• Artifacts uploaded (binaries only)  • Full build tree needed
```

**Why Reconfiguration Doesn't Solve It:**

1. **Reconfiguration creates empty build tree** - no built artifacts in place
2. **Tests need built binaries** - can get from artifacts, but...
3. **Coverage targets expect build-tree layout** - binaries in `build/bin/`, `build/lib/`
4. **Artifacts have dist layout** - binaries in `dist/rocm/bin/`, `dist/rocm/lib/`
5. **Symlinking doesn't work** - CMake targets use absolute paths from original configure

**Circular Dependency:**

```
Coverage target needs:               But we only have:
├─ Build tree structure              ├─ Empty reconfigured tree
├─ Built binaries in build/          ├─ Installed binaries in dist/rocm/
├─ CMake metadata (GLOBAL props)     ├─ Fresh CMake metadata (no accumulated properties)
└─ Coverage object list              └─ No way to reconstruct object list
```

**Status:** 🔴 **Design-level incompatibility** between artifact-based testing and CMake coverage targets

---

## Salvageable vs. Must-Scrap Analysis

### ✅ Salvageable: TheRock CI Integration Layer

**Keep as-is:**

1. **Coverage flag passthrough** (`cmake/therock_subproject.cmake`)
   - Mechanism works correctly
   - Projects can receive `-D<PROJECT>_ENABLE_COVERAGE=ON`

2. **LLVM tools build** (`compiler/CMakeLists.txt`)
   - llvm-cov, llvm-profdata, llvm-cxxfilt build correctly
   - Tools available in dist/rocm/lib/llvm/bin/

3. **Coverage project configuration** (`.github/scripts/therock_configure_coverage.py`)
   - Changed-subtrees detection works
   - Project metadata mapping correct
   - Matrix generation functional

4. **Main workflow orchestration** (`.github/workflows/therock-ci.yml`)
   - Setup job correctly determines coverage projects
   - Matrix strategy correct
   - Job dependencies correct

5. **Build job structure** (`.github/workflows/therock-ci-coverage.yml` build job)
   - Container setup correct
   - CMake configuration correct
   - Artifact upload works

**Requires minor updates:**

6. **Test job container/environment** (`.github/workflows/therock-ci-coverage.yml` test job)
   - Container image correct
   - Python/UV setup correct
   - Artifact download works
   - **Must change:** Test execution and coverage generation approach

### 🔴 Must Scrap: Two-Stage CMake Coverage Target Approach

**Cannot salvage:**

1. **Test job CMake reconfiguration**
   - Creates empty build tree without accumulated GLOBAL properties
   - Cannot reconstruct coverage object lists

2. **Running CMake coverage target in test job**
   ```yaml
   - name: Generate coverage report
     run: cmake --build . --target coverage
   ```
   - Coverage targets expect full build tree context
   - Requires binaries at build-tree-relative paths

3. **Assumption that artifacts preserve CMake metadata**
   - Artifact system only uploads installed files
   - CMakeCache.txt, GLOBAL properties, target definitions not preserved

**Root cause:** Attempting to use **build-time CMake targets** in a **test-time artifact environment**.

---

## What Needs to Change: High-Level Direction

### From: Two-Stage with CMake Coverage Target

```
Build Job (azure-linux)          Test Job (GPU runner)
├─ Configure with coverage       ├─ Download artifacts
├─ Build binaries                ├─ Reconfigure CMake ❌
├─ Upload artifacts              ├─ Run cmake --target coverage ❌
                                 └─ Upload reports
```

### To: Inline Coverage Generation in Test Job

```
Build Job (azure-linux)          Test Job (GPU runner)
├─ Configure with coverage       ├─ Download artifacts
├─ Build binaries                ├─ Set LLVM_PROFILE_FILE env
├─ Upload artifacts              ├─ Run tests directly (not via CMake)
├─ Upload CMake metadata ✨      ├─ Merge profraw → profdata
                                 ├─ Generate reports (inline llvm-cov commands)
                                 └─ Upload reports
```

**Key Changes:**

1. **Upload CMake metadata from build job**
   - CMakeCache.txt
   - Coverage object lists (extracted from CMake)
   - Test binary manifests

2. **Run tests without CMake in test job**
   - Direct test binary invocation OR
   - Custom Python script that mimics ctest but sets environment correctly OR
   - Reconstruct ctest invocation with explicit environment

3. **Generate coverage reports inline**
   - Use llvm-profdata/llvm-cov directly in bash
   - Read coverage object list from uploaded metadata
   - No dependency on CMake targets

### Alternative: Single-Job Coverage (Requires GPU in Build)

```
Coverage Job (GPU runner with build capacity)
├─ Configure with coverage
├─ Build binaries
├─ Run tests (LLVM_PROFILE_FILE set at configure time)
├─ Generate coverage (cmake --target coverage)
└─ Upload reports
```

**Trade-offs:**
- ✅ Pros: CMake coverage targets work, no metadata transport needed
- ❌ Cons: Wastes expensive GPU time on compilation, doesn't scale, blocks other GPU jobs

---

## Conclusion

**Coverage Process Understanding:** ✅ Complete
- LLVM instrumentation workflow documented
- Reference implementations analyzed (.so vs header-only patterns)
- Coverage object determination logic understood

**TheRock CI Integration:** ✅ Structurally complete, ❌ Execution blocked
- Workflow orchestration functional
- Project detection and matrix generation working
- Build job successfully creates coverage-instrumented artifacts
- Test job architecture incompatible with CMake coverage targets

**Fundamental Blocker:** CMake coverage targets require full build tree context with accumulated metadata, but TheRock's artifact-based testing pipeline only preserves installed binaries.

**Required Next Step:** Redesign coverage generation to work with artifacts-only environment, either by:
1. Embedding coverage metadata in artifacts and generating reports inline, OR
2. Consolidating build+test into single GPU job (expensive but simpler)

---

**Commits Reference:**

**TheRock (`enable-codecov`):**
- `341808faae` - Coverage flag passthrough
- `cca6f651a5` - Uppercase project name handling
- `1ce765508b` - Enable LLVM coverage tools
- `295f365101` - Add rocm-llvm-dev to container
- `f70ca7b755` - Export llvm-cov and llvm-profdata
- `6afcd5ac52` - Add llvm-cxxfilt to tools

**rocm-libraries (`therock-codecov`):**
- `12d9d71263`-`511bad7f51` - Initial coverage workflow setup
- `b256e80121` - Split build and test jobs
- `549974c348` - Align with therock-ci-linux test flow
- `adc90b3bc6` - Add GPU sanity check
- `667358d96c` - Fix build dir for coverage
- `bf8ad30220` - Attempt CMake reconfiguration (blocked)
