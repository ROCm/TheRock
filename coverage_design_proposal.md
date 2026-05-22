# Code Coverage Design Proposal: Standardized Artifact-Based Approach

## Executive Summary

This document proposes a standardized, artifact-based code coverage design for TheRock CI that works within the existing two-stage build/test pipeline. The design addresses the fundamental incompatibility between CMake coverage targets (which expect full build trees) and TheRock's artifact-only testing architecture.

**Core Principle:** Move coverage metadata generation from test-time to build-time, standardize LLVM_PROFILE_FILE embedding, and implement unified coverage report generation via Python tooling.

---

## Current State Analysis

### Projects with Coverage Support

**Comprehensive audit of all rocm-libraries projects:**

| Project | Status | LLVM Tools | Objects | LLVM_PROFILE_FILE | Notes |
|---------|--------|------------|---------|-------------------|-------|
| **hipblas** | ✅ Working | find_program | Single .so | Custom target env | Standard |
| **hipblaslt** | ✅ Working | find_program | Single .so | Custom target env | Uses ignore-regex for Tensile |
| **hipcub** | ⚠️ Incomplete | N/A | N/A | N/A | Flag defined, no target |
| **hipdnn** | ✅ Working | Custom function | Multiple (GLOBAL) | Custom target env | Most complex |
| **hipfft** | ✅ Working | find_program | Two .so files | Custom target env | Multiple libraries |
| **hiprand** | ⚠️ Incomplete | N/A | N/A | N/A | Flag defined, no target |
| **hipsolver** | ✅ Working | find_program | Single .so | Custom target env | Standard |
| **hipsparse** | ❌ Non-standard | lcov/gcov | N/A | N/A | Uses gcov NOT llvm-cov |
| **hipsparselt** | ✅ Working | find_program | Single .so | Custom target env | Best practices |
| **hiptensor** | ⚠️ Incomplete | N/A | N/A | N/A | Flag defined, no target |
| **rocblas** | ✅ Working | find_program | Single .so | Python test driver | Python-driven |
| **rocfft** | ✅ Working | find_program | Single .so | Custom target env | Standard |
| **rocprim** | ✅ Working | find_program | Multiple (GLOBAL) | set_tests_properties | Header-only |
| **rocrand** | ⚠️ Partial | N/A | N/A | N/A | Some flags, incomplete |
| **rocsolver** | ✅ Working | find_program | Single .so | Custom target env | Standard |
| **rocsparse** | ✅ Working | find_program | Single .so | Custom target env | Uses Python filter |
| **rocthrust** | ⚠️ Incomplete | N/A | N/A | N/A | Only compile flags |
| **rocwmma** | ⚠️ Incomplete | N/A | N/A | N/A | Compile flags only |

### Key Findings

**Critical Issues:**

1. **Inconsistent LLVM tool discovery:** Most use `find_program(HINTS ${ROCM_PATH}/llvm/bin)`, hipblaslt uses `CMAKE_PREFIX_PATH`, hipdnn has custom function
2. **hipsparse uses gcov/lcov:** Incompatible with LLVM coverage standard
3. **Six projects incomplete:** hipcub, hiprand, hiptensor, rocrand, rocthrust, rocwmma define flags but lack coverage targets
4. **LLVM_PROFILE_FILE methods vary:**
   - Most: `${CMAKE_COMMAND} -E env` in custom target
   - rocprim: `set_tests_properties(ENVIRONMENT)`
   - rocblas: Python test driver
   - All set at build/configure time, not runtime

5. **Coverage objects determination:**
   - .so libraries (majority): Single `-object` parameter
   - Header-only (rocprim): GLOBAL property accumulation
   - Hybrid (hipdnn): Custom list
   - Requires CMake metadata in ALL cases

**Ignore Patterns Used:**

- hipblaslt: `.*Tensile.*|.*origami.*`
- hipdnn: `.*deps.*|.*tests.*|.*data_sdk.*data_objects.*|.*HipErrorHandler.*`
- rocprim: `test_*`
- Most projects: None (rely on object selection)

---

## Design Proposal

### Validation of User's 7 Points

#### ✅ Point 1: Standardize LLVM_PROFILE_FILE Embedding

**Status:** VALID - Required change

**Current Reality:**
- All projects currently set LLVM_PROFILE_FILE at **build/configure time** (either in custom target or via set_tests_properties)
- No projects set it at runtime via environment export
- Pattern varies: some use `%m` (merge ID), rocFFT uses `%p` (process ID)

**Required Changes:**
1. Standardize on `%m` pattern (more reliable for parallel tests)
2. Set via `set_tests_properties()` for ctest-based tests (like rocprim/hiprand)
3. For non-ctest tests, embed in test invocation wrapper

**Implementation Path:** Add to `<PROJECT>_ENABLE_COVERAGE` passthrough:
```cmake
if(${PROJECT}_ENABLE_COVERAGE)
  set(LLVM_PROFILE_FILE_PATTERN "${CMAKE_BINARY_DIR}/coverage-report/profraw/%m.profraw")
  # Set for all tests via properties
  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "LLVM_PROFILE_FILE=${LLVM_PROFILE_FILE_PATTERN}"
  )
endif()
```

#### ✅ Point 2: Standardize LLVM_PROFILE_FILE Location

**Status:** VALID - Per-project standard acceptable

**Recommendation:**
- Build job: `${CMAKE_BINARY_DIR}/coverage-report/profraw/%m.profraw`
- After artifact download: `${OUTPUT_DIR}/coverage-report/profraw/%m.profraw`
- Pattern stored in metadata file uploaded from build job

**Why per-project is OK:** Each project's build tree is isolated in TheRock, no cross-project conflicts.

#### ✅ Point 3: Coverage Objects in test_categories.yaml

**Status:** VALID - Essential for standardization

**Current Problems:**
- rocprim accumulates via GLOBAL property (only available during CMake config)
- hipdnn has hardcoded list in CMakeLists.txt
- No way to extract from artifacts

**Proposed Solution:** Extend test_categories.yaml (or create test_categories_coverage.yaml):

```yaml
hipdnn:
  coverage:
    enabled: true
    objects:
      - type: library
        path: lib/libhipdnn_backend.so
      - type: test_binary
        pattern: bin/hipdnn_*_tests  # glob pattern
    ignore_regex: '.*deps.*|.*tests.*|.*data_sdk.*data_objects.*|.*HipErrorHandler.*'
    llvm_tools_prefix: dist/rocm/lib/llvm/bin

rocprim:
  coverage:
    enabled: true
    objects:
      - type: test_binary
        pattern: bin/test_*  # Header-only: all test binaries
    ignore_regex: 'test_*'
    llvm_tools_prefix: dist/rocm/lib/llvm/bin

rocfft:
  coverage:
    enabled: true
    objects:
      - type: library
        path: lib/librocfft.so
    llvm_tools_prefix: dist/rocm/lib/llvm/bin
```

**Alternative:** Generate during build job and upload as artifact.

#### ✅ Point 4: ignore-filename-regex in YAML

**Status:** VALID - Necessary for consistency

**Rationale:** Currently scattered across CMakeLists.txt files. Centralizing in YAML makes it discoverable and modifiable without CMake changes.

#### ⚠️ Point 5: Auto-detection of Coverage Targets

**Status:** FUTURE WORK - Complex, defer to v2

**Problem:**
- Requires hooking into CMake target creation
- Risk of instrumenting third-party code (Tensile, gtest, etc.)
- No clear boundary between "project code" and "dependencies"

**Recommendation:**
- **Phase 1 (this design):** Explicit YAML-based configuration
- **Phase 2 (future):** Investigate CMake hooks with explicit exclusion lists

**Defer rationale:** Adds significant complexity; explicit config is safer initially.

#### ✅ Point 6: coverage_runner.py Script

**Status:** VALID - Core of new design

**Purpose:** Replace CMake coverage target with Python script that:
1. Reads coverage metadata from YAML or uploaded artifact
2. Runs tests with LLVM_PROFILE_FILE set
3. Merges profraw files using llvm-profdata
4. Generates reports using llvm-cov
5. Uploads coverage artifacts

**Relationship to test_runner.py:**
- Extend or wrap test_runner.py
- Add coverage-specific pre/post steps
- Reuse test filtering and sharding logic

#### ⚠️ Point 7: LLVM Tools from amd-llvm

**Status:** PARTIALLY VALID - Needs flexibility

**Current Reality:**
- Most projects: `find_program(HINTS ${ROCM_PATH}/llvm/bin)`
- hipblaslt: `CMAKE_PREFIX_PATH` hints
- hipdnn: Custom `findandcheckllvmtools()` function

**Problems:**
1. `find_program()` runs at **configure time** (build job)
2. Tools are in artifacts at **different path** (test job): `dist/rocm/lib/llvm/bin`
3. Hardcoding paths breaks flexibility

**Proposed Solution:**
```python
# In coverage_runner.py
def find_llvm_tools(artifact_dir, yaml_config):
    # Priority order:
    # 1. YAML-specified path (if absolute and exists)
    # 2. Artifact path: {artifact_dir}/dist/rocm/lib/llvm/bin
    # 3. System ROCM_PATH: ${ROCM_PATH}/llvm/bin
    # 4. System paths: /opt/rocm/llvm/bin
    
    search_paths = [
        yaml_config.get('llvm_tools_prefix'),
        f"{artifact_dir}/dist/rocm/lib/llvm/bin",
        f"{os.getenv('ROCM_PATH', '/opt/rocm')}/llvm/bin",
        "/opt/rocm/llvm/bin"
    ]
    
    for path in search_paths:
        if path and os.path.exists(f"{path}/llvm-cov"):
            return path
    raise FileNotFoundError("llvm-cov not found in search paths")
```

---

## Missing Issues Identified

### Issue #8: hipsparse gcov Incompatibility

**Problem:** hipsparse uses `gcov`/`lcov` instead of `llvm-cov`.

**Impact:** Cannot use standard coverage pipeline.

**Options:**
1. Exclude hipsparse from coverage (short term)
2. Migrate hipsparse to llvm-cov (requires project-level changes)
3. Support dual toolchain in coverage_runner.py (complex)

**Recommendation:** Exclude from initial implementation, add migration as separate effort.

### Issue #9: Incomplete Projects

**Projects with flags but no targets:** hipcub, hiprand, hiptensor, rocrand, rocthrust, rocwmma

**Impact:** Cannot enable coverage even if requested.

**Recommendation:**
- Document as "not yet supported"
- therock_configure_coverage.py filters these out (already does via COVERAGE_PROJECT_METADATA)
- Add incrementally as projects implement coverage targets

### Issue #10: Python Test Drivers

**Projects with custom test execution:** rocblas

**Problem:** rocblas uses `rtest.py` instead of direct ctest invocation.

**Impact:** coverage_runner.py must support multiple test invocation patterns.

**Solution:** Make test invocation configurable in YAML:

```yaml
rocblas:
  coverage:
    test_command: python3 {build_dir}/clients/rtest.py
    # vs. standard:
    # test_command: ctest -L {category}
```

### Issue #11: Multiple Library Objects

**Projects with multiple .so files:** hipfft (libhipfft.so + libhipfftw.so)

**Impact:** Must support multiple `-object` flags for .so libraries (not just header-only).

**Solution:** YAML objects list handles this:
```yaml
hipfft:
  coverage:
    objects:
      - type: library
        path: lib/libhipfft.so
      - type: library
        path: lib/libhipfftw.so
```

### Issue #12: Generator Expressions in CMake

**Projects using:** hipblaslt (`$<TARGET_FILE:hipblaslt>`)

**Problem:** Generator expressions only resolve at build time, not available in artifacts.

**Solution:** Build job must resolve and export to metadata:
```cmake
# During build, export coverage metadata
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/coverage_metadata.json"
  CONTENT "{\"objects\": [\"$<TARGET_FILE:hipblaslt>\"]}"
)
```

---

## Proposed Architecture

### Build Job Changes

**New Step: Export Coverage Metadata**

```yaml
- name: Export coverage metadata
  if: inputs.project_name != ''  # Coverage build
  working-directory: ${{ github.workspace }}/TheRock/build-coverage
  run: |
    python3 ../build_tools/export_coverage_metadata.py \
      --project ${{ inputs.project_name }} \
      --build-dir . \
      --output coverage_metadata.json
```

**export_coverage_metadata.py** extracts:
- Coverage object paths (from CMake cache or YAML config)
- LLVM_PROFILE_FILE pattern used
- LLVM tools search prefix
- Ignore regex patterns
- Test binary list (for header-only)

**Upload as artifact:**
```yaml
- name: Upload coverage metadata
  uses: actions/upload-artifact@v4
  with:
    name: coverage-metadata-${{ inputs.project_name }}
    path: TheRock/build-coverage/coverage_metadata.json
```

### Test Job Changes

**Replace CMake coverage target with coverage_runner.py:**

```yaml
- name: Download coverage metadata
  uses: actions/download-artifact@v4
  with:
    name: coverage-metadata-${{ inputs.project_name }}
    path: coverage-metadata

- name: Create coverage directories
  run: mkdir -p coverage-report/profraw

- name: Run tests with coverage profiling
  env:
    LLVM_PROFILE_FILE: ${{ github.workspace }}/coverage-report/profraw/%m.profraw
  run: |
    python3 TheRock/build_tools/github_actions/coverage_runner.py \
      --metadata coverage-metadata/coverage_metadata.json \
      --artifact-dir TheRock/build-coverage \
      --project ${{ inputs.project_name }} \
      --test-category quick \
      --output-dir coverage-report

- name: Upload coverage reports
  uses: actions/upload-artifact@v4
  with:
    name: coverage-reports-${{ inputs.project_name }}
    path: coverage-report/*.{info,report,html}
```

### coverage_runner.py Implementation

**High-level structure:**

```python
def main():
    # 1. Load metadata (from artifact or YAML fallback)
    metadata = load_coverage_metadata(args.metadata, args.project)
    
    # 2. Find LLVM tools
    llvm_tools = find_llvm_tools(args.artifact_dir, metadata)
    
    # 3. Set LLVM_PROFILE_FILE environment
    profraw_dir = f"{args.output_dir}/profraw"
    os.environ['LLVM_PROFILE_FILE'] = f"{profraw_dir}/%m.profraw"
    
    # 4. Run tests (delegate to test_runner.py or custom command)
    run_tests(metadata, args.test_category, args.artifact_dir)
    
    # 5. Merge profraw files
    merge_profraw_files(llvm_tools, profraw_dir, args.output_dir)
    
    # 6. Generate coverage reports
    generate_coverage_reports(
        llvm_tools,
        metadata.objects,
        metadata.ignore_regex,
        args.output_dir
    )
```

**Key functions:**

- `load_coverage_metadata()`: Read JSON from artifact or parse YAML
- `find_llvm_tools()`: Search for llvm-cov, llvm-profdata, llvm-cxxfilt
- `run_tests()`: Invoke test_runner.py or custom test command
- `merge_profraw_files()`: Execute llvm-profdata merge
- `generate_coverage_reports()`: Execute llvm-cov report/show/export

---

## Migration Path

### Phase 1: Proof of Concept (hipDNN only)

1. Create test_categories_coverage.yaml with hipDNN config
2. Implement export_coverage_metadata.py (reads YAML)
3. Implement coverage_runner.py (basic version)
4. Modify therock-ci-coverage.yml to use new approach
5. Test on single project

**Success Criteria:** hipDNN coverage reports generated in test job without CMake reconfiguration.

### Phase 2: Standardization (3-5 .so-based projects)

1. Add rocFFT, hiprand, rocblas to YAML
2. Extend coverage_runner.py to handle variations (Python test drivers, multiple .so)
3. Validate across different project types

**Success Criteria:** Multiple .so-based projects generate reports successfully.

### Phase 3: Header-Only Support (rocprim, rocthrust)

1. Add rocprim to YAML with test_binary pattern
2. Enhance export_coverage_metadata.py to enumerate test binaries
3. Test header-only coverage flow

**Success Criteria:** rocprim generates coverage with multiple test binary objects.

### Phase 4: Complete Rollout

1. Add remaining supported projects to YAML
2. Document YAML schema and coverage_runner.py usage
3. Migrate all projects from CMake coverage targets to YAML config

**Exclusions:** hipsparse (gcov), incomplete projects (until they add targets)

---

## YAML Schema Definition

### test_categories_coverage.yaml

```yaml
# Schema for coverage configuration
# Located at: rocm-libraries/.github/test_categories_coverage.yaml

<project_name>:
  coverage:
    enabled: bool  # Whether coverage is supported
    
    # Coverage objects to analyze
    objects:
      - type: library | test_binary
        path: string  # Exact path relative to artifact root
        # OR
        pattern: string  # Glob pattern for multiple objects
    
    # Filename exclusion pattern (regex)
    ignore_regex: string  # Optional, default: none
    
    # Test execution configuration
    test_command: string  # Optional, default: use test_runner.py
    test_categories: [string]  # Optional, default: [quick]
    
    # LLVM tools location
    llvm_tools_prefix: string  # Optional, default: dist/rocm/lib/llvm/bin
    
    # LLVM_PROFILE_FILE pattern
    profile_file_pattern: string  # Optional, default: %m.profraw

# Example: .so-based library
rocfft:
  coverage:
    enabled: true
    objects:
      - type: library
        path: dist/rocm/lib/librocfft.so
    llvm_tools_prefix: dist/rocm/lib/llvm/bin

# Example: Header-only library
rocprim:
  coverage:
    enabled: true
    objects:
      - type: test_binary
        pattern: bin/test_*
    ignore_regex: 'test_*'
    llvm_tools_prefix: dist/rocm/lib/llvm/bin

# Example: Multiple libraries
hipfft:
  coverage:
    enabled: true
    objects:
      - type: library
        path: dist/rocm/lib/libhipfft.so
      - type: library
        path: dist/rocm/lib/libhipfftw.so
    llvm_tools_prefix: dist/rocm/lib/llvm/bin

# Example: Hybrid (library + test binaries)
hipdnn:
  coverage:
    enabled: true
    objects:
      - type: library
        path: dist/rocm/lib/libhipdnn_backend.so
      - type: test_binary
        pattern: bin/hipdnn_*_tests
    ignore_regex: '.*deps.*|.*tests.*|.*data_sdk.*data_objects.*|.*HipErrorHandler.*'
    llvm_tools_prefix: dist/rocm/lib/llvm/bin

# Example: Custom test driver
rocblas:
  coverage:
    enabled: true
    objects:
      - type: library
        path: dist/rocm/lib/librocblas.so
    test_command: "python3 clients/rtest.py"
    llvm_tools_prefix: dist/rocm/lib/llvm/bin
```

---

## Risk Assessment and Mitigations

### Risk 1: LLVM_PROFILE_FILE Still Doesn't Propagate

**Likelihood:** Low (rocprim/hiprand prove set_tests_properties works)

**Mitigation:**
- Validate profraw generation in Phase 1
- If issues persist, fall back to direct test binary invocation (like rocFFT)

### Risk 2: Coverage Object Paths Wrong After Artifact Download

**Likelihood:** Medium (artifacts have different layout than build tree)

**Mitigation:**
- Metadata export resolves paths relative to artifact root
- YAML uses artifact-relative paths (dist/rocm/...)
- Validation step in coverage_runner.py checks file existence

### Risk 3: Project-Specific Quirks Break Generic Script

**Likelihood:** Medium (rocblas Python driver, hipfft multiple .so)

**Mitigation:**
- Make test_command configurable in YAML
- Support multiple object types in YAML schema
- Phase 2 explicitly tests variations

### Risk 4: Incomplete Projects Block Rollout

**Likelihood:** Low (therock_configure_coverage.py already filters)

**Mitigation:**
- Maintain explicit supported project list
- Document unsupported projects clearly
- Incremental rollout doesn't require all projects

---

## Success Criteria

**Phase 1 (PoC) Success:**
- [ ] hipDNN coverage workflow completes without CMake coverage target
- [ ] coverage_metadata.json uploaded from build job
- [ ] profraw files generated in test job
- [ ] coverage.info created and uploaded
- [ ] No CMake reconfiguration errors

**Phase 2 (Standardization) Success:**
- [ ] 3+ .so-based projects generate reports
- [ ] Variations handled (multiple .so, custom test drivers)
- [ ] YAML configuration covers common patterns

**Phase 3 (Header-Only) Success:**
- [ ] rocprim coverage works with test_binary pattern
- [ ] Multiple -object parameters generated correctly
- [ ] Header-only pattern documented

**Final Rollout Success:**
- [ ] 10+ projects migrated to YAML-based coverage
- [ ] CMake coverage targets deprecated (but preserved for local use)
- [ ] Documentation complete
- [ ] therock-ci-coverage.yml stable across projects

---

## Open Questions

1. **Should YAML be project-specific or centralized?**
   - Proposed: Centralized at rocm-libraries/.github/test_categories_coverage.yaml
   - Alternative: Per-project coverage.yaml
   - **Recommendation:** Centralized for discoverability

2. **Should we modify project CMakeLists.txt to standardize LLVM_PROFILE_FILE?**
   - Proposed: Yes, via therock_cmake_subproject passthrough helpers
   - Alternative: Set in coverage_runner.py only
   - **Recommendation:** Both - set in CMake for local dev, override in CI script

3. **How to handle projects that add/remove test binaries frequently?**
   - Proposed: Glob patterns in YAML (pattern: bin/test_*)
   - Alternative: Auto-enumerate in export_coverage_metadata.py
   - **Recommendation:** Globs initially, auto-enum in Phase 5

4. **Should coverage_runner.py replace or wrap test_runner.py?**
   - Proposed: Wrap - call test_runner.py for test execution
   - Alternative: Replace - duplicate test logic
   - **Recommendation:** Wrap for code reuse

---

## Conclusion

**Validated User Points:**
- ✅ Point 1 (LLVM_PROFILE_FILE embedding): Required
- ✅ Point 2 (Standardize location): Valid, per-project OK
- ✅ Point 3 (YAML configuration): Essential
- ✅ Point 4 (ignore-regex in YAML): Necessary
- ⚠️ Point 5 (Auto-detection): Defer to future
- ✅ Point 6 (coverage_runner.py): Core implementation
- ⚠️ Point 7 (LLVM tools from amd-llvm): Needs flexibility

**Additional Issues Found:**
- Issue #8: hipsparse gcov incompatibility
- Issue #9: Incomplete project implementations
- Issue #10: Python test driver support needed
- Issue #11: Multiple library object support
- Issue #12: Generator expression resolution

**Recommendation:** Proceed with phased implementation, starting with hipDNN proof of concept.
