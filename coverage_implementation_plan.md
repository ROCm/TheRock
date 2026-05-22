# Coverage Implementation Plan

**Prerequisites:** Read `coverage_workflow_summary.md` and `coverage_design_proposal.md` first.

**Goal:** Implement YAML-based coverage workflow compatible with GitHub Actions artifact-based testing.

## Architecture Overview

```
TheRock Build Job (no GPU)
├─> Build with -D<PROJECT>_ENABLE_COVERAGE=ON
├─> Run CMake coverage target to generate coverage_metadata.json
├─> Upload coverage_metadata.json as artifact
└─> Upload ROCm dist/ as artifact

Test Job (with GPU)
├─> Download artifacts
├─> coverage_runner.py reads coverage_metadata.json
├─> Set LLVM_PROFILE_FILE environment
├─> Run tests via CTest/direct invocation
├─> Merge .profraw files with llvm-profdata
├─> Generate reports with llvm-cov
└─> Upload coverage reports to artifacts/codecov.io
```

## Phase 1: hipDNN Proof of Concept

### 1.1: Create YAML Configuration Schema

**File:** `rocm-libraries/test_categories_coverage.yaml`

```yaml
# Coverage configuration for rocm-libraries projects
# Defines which projects have coverage enabled and their metadata

projects:
  hipdnn:
    enabled: true
    coverage_objects:
      # Libraries to analyze
      libraries:
        - lib/libhipdnn_backend.so
      # Test binaries to analyze
      test_binaries:
        - bin/hipdnn_backend_tests
        - bin/hipdnn_frontend_tests
        - bin/hipdnn_public_backend_tests
        - bin/hipdnn_data_sdk_tests
        - bin/hipdnn_plugin_sdk_tests
        - bin/hipdnn_test_sdk_tests
    ignore_filename_regex: '.*deps.*|.*tests.*|.*data_sdk.*data_objects.*|.*HipErrorHandler.*'
    llvm_profile_pattern: '%m'  # Use %m (merge ID) for parallel tests
    test_category: quick  # Which test category to run for coverage
```

**Validation:**
```bash
python3 -c "import yaml; yaml.safe_load(open('rocm-libraries/test_categories_coverage.yaml'))"
```

### 1.2: Create Coverage Metadata Export Script

**File:** `rocm-libraries/.github/scripts/export_coverage_metadata.py`

```python
#!/usr/bin/env python3
"""Export coverage metadata from CMake to JSON for use in test jobs."""
import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_coverage_config(config_path: Path) -> dict:
    """Load test_categories_coverage.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_coverage_objects(build_dir: Path, project_key: str, config: dict) -> dict:
    """Find coverage objects (libraries and test binaries) in build tree.
    
    Args:
        build_dir: Path to TheRock/build-coverage
        project_key: Project key (e.g., 'hipdnn')
        config: Coverage config from YAML
    
    Returns:
        Dict with 'libraries' and 'test_binaries' lists of absolute paths
    """
    project_config = config['projects'][project_key]
    
    found_objects = {
        'libraries': [],
        'test_binaries': []
    }
    
    # Find libraries
    for lib_relpath in project_config['coverage_objects']['libraries']:
        lib_path = build_dir / lib_relpath
        if lib_path.exists():
            found_objects['libraries'].append(str(lib_path))
        else:
            logging.warning(f"Library not found: {lib_path}")
    
    # Find test binaries
    for bin_relpath in project_config['coverage_objects']['test_binaries']:
        bin_path = build_dir / bin_relpath
        if bin_path.exists():
            found_objects['test_binaries'].append(str(bin_path))
        else:
            logging.warning(f"Test binary not found: {bin_path}")
    
    return found_objects


def export_metadata(build_dir: Path, project_key: str, config_path: Path, output_path: Path):
    """Generate coverage_metadata.json for a project."""
    config = load_coverage_config(config_path)
    
    if project_key not in config['projects']:
        raise ValueError(f"Project '{project_key}' not found in coverage config")
    
    project_config = config['projects'][project_key]
    
    if not project_config['enabled']:
        logging.info(f"Coverage disabled for {project_key}")
        return
    
    # Find coverage objects
    coverage_objects = find_coverage_objects(build_dir, project_key, config)
    
    # Build metadata
    metadata = {
        'project': project_key,
        'coverage_objects': coverage_objects,
        'ignore_filename_regex': project_config['ignore_filename_regex'],
        'llvm_profile_pattern': project_config['llvm_profile_pattern'],
        'test_category': project_config['test_category'],
        'llvm_tools': {
            'llvm_profdata': 'dist/rocm/lib/llvm/bin/llvm-profdata',
            'llvm_cov': 'dist/rocm/lib/llvm/bin/llvm-cov',
            'llvm_cxxfilt': 'dist/rocm/lib/llvm/bin/llvm-cxxfilt',
        }
    }
    
    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Exported coverage metadata to {output_path}")
    logging.info(f"  Libraries: {len(coverage_objects['libraries'])}")
    logging.info(f"  Test binaries: {len(coverage_objects['test_binaries'])}")


def main():
    parser = argparse.ArgumentParser(description='Export coverage metadata from CMake')
    parser.add_argument('--build-dir', type=Path, required=True,
                        help='Path to TheRock/build-coverage')
    parser.add_argument('--project', type=str, required=True,
                        help='Project key (e.g., hipdnn)')
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to test_categories_coverage.yaml')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output JSON file path')
    
    args = parser.parse_args()
    export_metadata(args.build_dir, args.project, args.config, args.output)


if __name__ == '__main__':
    main()
```

**Test locally:**
```bash
cd /home/jorobbin/git/rocm-libraries
python3 .github/scripts/export_coverage_metadata.py \
  --build-dir ../TheRock/build-coverage \
  --project hipdnn \
  --config test_categories_coverage.yaml \
  --output coverage_metadata.json

cat coverage_metadata.json
```

### 1.3: Create Coverage Runner Script

**File:** `rocm-libraries/.github/scripts/coverage_runner.py`

```python
#!/usr/bin/env python3
"""Run tests with coverage profiling and generate coverage reports."""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def set_coverage_environment(metadata: dict, coverage_dir: Path):
    """Set LLVM_PROFILE_FILE environment variable."""
    pattern = metadata['llvm_profile_pattern']
    profraw_dir = coverage_dir / 'profraw'
    profraw_dir.mkdir(parents=True, exist_ok=True)
    
    profile_file = str(profraw_dir / f'{pattern}.profraw')
    os.environ['LLVM_PROFILE_FILE'] = profile_file
    logging.info(f"Set LLVM_PROFILE_FILE={profile_file}")


def run_tests(build_dir: Path, metadata: dict):
    """Run tests to generate .profraw files."""
    test_category = metadata['test_category']
    
    logging.info(f"Running {test_category} tests...")
    
    # Option 1: Use ctest with label filter
    cmd = [
        'ctest',
        '--test-dir', str(build_dir),
        '-L', test_category,
        '--output-on-failure',
        '--parallel', '8',
        '--timeout', '7200',
    ]
    
    logging.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=build_dir)
    
    if result.returncode != 0:
        logging.warning(f"Tests exited with code {result.returncode}")
    
    return result.returncode


def merge_profraw_files(build_dir: Path, metadata: dict, coverage_dir: Path) -> Path:
    """Merge .profraw files into .profdata."""
    profraw_dir = coverage_dir / 'profraw'
    profdata_path = coverage_dir / f"{metadata['project']}.profdata"
    
    # Find all .profraw files
    profraw_files = list(profraw_dir.glob('*.profraw'))
    
    if not profraw_files:
        raise RuntimeError(f"No .profraw files found in {profraw_dir}")
    
    logging.info(f"Found {len(profraw_files)} .profraw files")
    
    # Get llvm-profdata tool
    llvm_profdata = build_dir / metadata['llvm_tools']['llvm_profdata']
    if not llvm_profdata.exists():
        raise FileNotFoundError(f"llvm-profdata not found: {llvm_profdata}")
    
    # Merge command
    cmd = [str(llvm_profdata), 'merge', '-sparse', '-o', str(profdata_path)]
    cmd.extend(str(p) for p in profraw_files)
    
    logging.info(f"Merging {len(profraw_files)} profraw files...")
    subprocess.run(cmd, check=True)
    
    logging.info(f"Created {profdata_path}")
    return profdata_path


def generate_coverage_reports(build_dir: Path, metadata: dict, coverage_dir: Path, profdata_path: Path):
    """Generate coverage reports (text, HTML, LCOV)."""
    # Get LLVM tools
    llvm_cov = build_dir / metadata['llvm_tools']['llvm_cov']
    llvm_cxxfilt = build_dir / metadata['llvm_tools']['llvm_cxxfilt']
    
    for tool in [llvm_cov, llvm_cxxfilt]:
        if not tool.exists():
            raise FileNotFoundError(f"LLVM tool not found: {tool}")
    
    # Build object arguments
    object_args = []
    for lib in metadata['coverage_objects']['libraries']:
        lib_path = build_dir / lib
        object_args.extend(['-object', str(lib_path)])
    
    for binary in metadata['coverage_objects']['test_binaries']:
        bin_path = build_dir / binary
        object_args.extend(['-object', str(bin_path)])
    
    ignore_regex = metadata['ignore_filename_regex']
    
    # Generate text report
    logging.info("Generating text coverage report...")
    text_report = coverage_dir / f"code_cov_{metadata['project']}.report"
    cmd = [
        str(llvm_cov), 'report',
        *object_args,
        f'-instr-profile={profdata_path}',
        f'-ignore-filename-regex={ignore_regex}',
    ]
    with open(text_report, 'w') as f:
        subprocess.run(cmd, stdout=f, check=True)
    logging.info(f"Created {text_report}")
    
    # Print summary
    with open(text_report) as f:
        print(f.read())
    
    # Generate HTML report
    logging.info("Generating HTML coverage report...")
    cmd = [
        str(llvm_cov), 'show',
        f'-Xdemangler={llvm_cxxfilt}',
        *object_args,
        f'-instr-profile={profdata_path}',
        f'-ignore-filename-regex={ignore_regex}',
        '--format=html',
        f'--output-dir={coverage_dir}',
    ]
    subprocess.run(cmd, check=True)
    logging.info(f"Created HTML report in {coverage_dir}/")
    
    # Generate LCOV export
    logging.info("Generating LCOV export...")
    lcov_file = coverage_dir / 'coverage.info'
    cmd = [
        str(llvm_cov), 'export',
        *object_args,
        f'-instr-profile={profdata_path}',
        f'-ignore-filename-regex={ignore_regex}',
        '--format=lcov',
    ]
    with open(lcov_file, 'w') as f:
        subprocess.run(cmd, stdout=f, check=True)
    logging.info(f"Created {lcov_file}")


def main():
    parser = argparse.ArgumentParser(description='Run coverage tests and generate reports')
    parser.add_argument('--build-dir', type=Path, required=True,
                        help='Path to TheRock/build-coverage')
    parser.add_argument('--metadata', type=Path, required=True,
                        help='Path to coverage_metadata.json')
    parser.add_argument('--coverage-dir', type=Path, required=True,
                        help='Output directory for coverage reports')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Skip test execution (use existing .profraw files)')
    
    args = parser.parse_args()
    
    # Load metadata
    with open(args.metadata) as f:
        metadata = json.load(f)
    
    logging.info(f"Running coverage for project: {metadata['project']}")
    
    # Set environment
    set_coverage_environment(metadata, args.coverage_dir)
    
    # Run tests
    if not args.skip_tests:
        test_exit_code = run_tests(args.build_dir, metadata)
        if test_exit_code != 0:
            logging.warning("Tests failed but continuing with coverage generation")
    
    # Merge profraw
    profdata_path = merge_profraw_files(args.build_dir, metadata, args.coverage_dir)
    
    # Generate reports
    generate_coverage_reports(args.build_dir, metadata, args.coverage_dir, profdata_path)
    
    logging.info("Coverage generation complete!")


if __name__ == '__main__':
    main()
```

**Test locally:**
```bash
cd /home/jorobbin/git/rocm-libraries
python3 .github/scripts/coverage_runner.py \
  --build-dir ../TheRock/build-coverage \
  --metadata coverage_metadata.json \
  --coverage-dir ../TheRock/build-coverage/coverage-report
```

### 1.4: Modify GitHub Actions Workflow

**File:** `rocm-libraries/.github/workflows/therock-ci-coverage.yml`

**Changes to Build Job (lines 82-96):**

Replace:
```yaml
      - name: Configure coverage build
        env:
          amdgpu_families: ${{ env.AMDGPU_FAMILIES }}
          package_version: ADHOCBUILD
          extra_cmake_options: >-
            -DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=../
            -D${{ inputs.project_name }}_ENABLE_COVERAGE=ON
            -DLLVM_TOOLS_SEARCH_PREFIX=${{ github.workspace }}/TheRock/build-coverage/dist/rocm/lib/llvm
            ${{ inputs.cmake_options }}
          BUILD_DIR: build-coverage
        run: |
          python3 TheRock/build_tools/github_actions/build_configure.py

      - name: Build therock-archives and therock-dist
        run: cmake --build TheRock/build-coverage --target therock-archives therock-dist -- -k 0
```

With:
```yaml
      - name: Configure coverage build
        env:
          amdgpu_families: ${{ env.AMDGPU_FAMILIES }}
          package_version: ADHOCBUILD
          extra_cmake_options: >-
            -DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=../
            -D${{ inputs.project_name }}_ENABLE_COVERAGE=ON
            -DLLVM_TOOLS_SEARCH_PREFIX=${{ github.workspace }}/TheRock/build-coverage/dist/rocm/lib/llvm
            ${{ inputs.cmake_options }}
          BUILD_DIR: build-coverage
        run: |
          python3 TheRock/build_tools/github_actions/build_configure.py

      - name: Build therock-archives and therock-dist
        run: cmake --build TheRock/build-coverage --target therock-archives therock-dist -- -k 0

      - name: Export coverage metadata
        run: |
          python3 .github/scripts/export_coverage_metadata.py \
            --build-dir TheRock/build-coverage \
            --project ${{ inputs.project_name }} \
            --config test_categories_coverage.yaml \
            --output TheRock/build-coverage/coverage_metadata.json

      - name: Upload coverage metadata artifact
        uses: actions/upload-artifact@v4
        with:
          name: coverage-metadata-${{ inputs.project_name }}-${{ inputs.amdgpu_families }}
          path: TheRock/build-coverage/coverage_metadata.json
          retention-days: 7
```

**Changes to Test Job (lines 197-230):**

Replace:
```yaml
      - name: Reconfigure CMake for coverage
        env:
          amdgpu_families: ${{ inputs.amdgpu_families }}
          package_version: ADHOCBUILD
          extra_cmake_options: >-
            -DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=../rocm-libraries/
            -D${{ inputs.project_name }}_ENABLE_COVERAGE=ON
            -DLLVM_TOOLS_SEARCH_PREFIX=${{ github.workspace }}/TheRock/build-coverage/dist/rocm/lib/llvm
            ${{ inputs.cmake_options }}
          BUILD_DIR: build-coverage
        working-directory: ${{ github.workspace }}/TheRock
        run: |
          python3 build_tools/github_actions/build_configure.py

      - name: Run tests to generate profraw files
        working-directory: ${{ github.workspace }}/TheRock/build-coverage
        run: |
          cmake --build . --target ${{ inputs.cmake_target }}+test

      - name: Generate coverage report
        working-directory: ${{ github.workspace }}/TheRock/build-coverage
        run: |
          cmake --build . --target coverage
```

With:
```yaml
      - name: Download coverage metadata
        uses: actions/download-artifact@v4
        with:
          name: coverage-metadata-${{ inputs.project_name }}-${{ inputs.amdgpu_families }}
          path: ${{ github.workspace }}/

      - name: Run tests and generate coverage reports
        run: |
          python3 rocm-libraries/.github/scripts/coverage_runner.py \
            --build-dir ${{ github.workspace }}/TheRock/build-coverage \
            --metadata ${{ github.workspace }}/coverage_metadata.json \
            --coverage-dir ${{ github.workspace }}/TheRock/build-coverage/coverage-report

      - name: Upload coverage reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-reports-${{ inputs.project_name }}-${{ inputs.amdgpu_families }}
          path: |
            ${{ github.workspace }}/TheRock/build-coverage/coverage-report/*.report
            ${{ github.workspace }}/TheRock/build-coverage/coverage-report/*.info
            ${{ github.workspace }}/TheRock/build-coverage/coverage-report/index.html
          retention-days: 30
```

### 1.5: Phase 1 Acceptance Criteria

**Success criteria:**
1. ✅ YAML config loads without errors
2. ✅ export_coverage_metadata.py runs successfully in build job
3. ✅ coverage_metadata.json artifact uploaded
4. ✅ coverage_runner.py downloads metadata and runs in test job
5. ✅ Multiple .profraw files generated (verify with debug step)
6. ✅ Coverage reports generated (text, HTML, LCOV)
7. ✅ LCOV file uploaded to artifacts
8. ✅ Coverage percentage shown in GitHub Actions logs

**Validation commands:**
```bash
# In test job logs, verify:
grep "Found .* .profraw files" 
grep "Coverage report generated"
grep "TOTAL" coverage-report/code_cov_hipdnn.report
```

**Debug step (add if needed):**
```yaml
- name: Debug - List profraw files
  if: always()
  run: |
    echo "=== Profraw files generated ==="
    find ${{ github.workspace }}/TheRock/build-coverage/coverage-report/profraw \
      -name "*.profraw" -ls | head -20
```

## Phase 2: Extend to .so-based Projects

**Target projects:** rocfft, rocblas, rocsparse, rocrand, hiprand

### 2.1: Add Projects to YAML Config

**File:** `rocm-libraries/test_categories_coverage.yaml`

```yaml
projects:
  hipdnn:
    # ... existing config ...

  rocfft:
    enabled: true
    coverage_objects:
      libraries:
        - lib/librocfft.so
      test_binaries:
        - bin/rocfft-test
        - bin/rocfft-test-fp16
    ignore_filename_regex: '.*test.*|.*client.*'
    llvm_profile_pattern: '%m'
    test_category: quick

  rocblas:
    enabled: true
    coverage_objects:
      libraries:
        - lib/librocblas.so
      test_binaries:
        - bin/rocblas-test
    ignore_filename_regex: '.*test.*|.*client.*'
    llvm_profile_pattern: '%m'
    test_category: quick

  # Add rocsparse, rocrand, hiprand similarly...
```

### 2.2: Update therock_configure_coverage.py

**File:** `rocm-libraries/.github/scripts/therock_configure_coverage.py`

Update `COVERAGE_PROJECT_METADATA` (lines 17-20):

```python
# Remove hardcoded metadata - now loaded from YAML
COVERAGE_CONFIG_PATH = SCRIPT_DIR / ".." / ".." / "test_categories_coverage.yaml"

def load_coverage_config():
    """Load coverage configuration from YAML."""
    import yaml
    with open(COVERAGE_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return {
        key: value 
        for key, value in config['projects'].items() 
        if value.get('enabled', False)
    }

# Replace get_build_metadata() function:
def get_build_metadata(project_key: str, base_dir: str = "TheRock/build-coverage"):
    """Get build directory for a coverage-enabled project."""
    coverage_config = load_coverage_config()
    if project_key not in coverage_config:
        return None
    
    # Infer build_subdir from project structure
    # This is a heuristic - may need project-specific mapping
    subdir_map = {
        'hipdnn': 'ml-libs/hipDNN',
        'rocfft': 'math-libs/rocFFT',
        'rocblas': 'math-libs/rocBLAS',
        'rocsparse': 'math-libs/rocSPARSE',
        'rocrand': 'math-libs/rocRAND',
        'hiprand': 'math-libs/hipRAND',
    }
    
    if project_key not in subdir_map:
        logging.warning(f"No subdir mapping for {project_key}")
        return None
    
    build_dir = f"{base_dir}/{subdir_map[project_key]}/build"
    return project_key.upper(), project_key, build_dir
```

### 2.3: Phase 2 Validation

**For each project:**
1. Add to `test_categories_coverage.yaml`
2. Verify coverage objects exist in build (run export_coverage_metadata.py locally)
3. Push branch and trigger CI
4. Verify coverage reports generated
5. Check codecov.io integration (if configured)

## Phase 3: Header-Only Libraries

**Target projects:** rocprim, hipcub, rocthrust, hip

### 3.1: Handle Test-Binary-Only Coverage

**Update YAML for header-only projects:**

```yaml
projects:
  rocprim:
    enabled: true
    coverage_objects:
      libraries: []  # No .so for header-only
      test_binaries:
        - bin/test_hipcub_block
        - bin/test_hipcub_device
        # ... all test binaries
    ignore_filename_regex: '.*test.*'
    llvm_profile_pattern: '%m'
    test_category: quick

  hipcub:
    enabled: true
    coverage_objects:
      libraries: []
      test_binaries:
        - bin/test_hipcub_block_discontinuity
        - bin/test_hipcub_block_exchange
        # ... all test binaries
    ignore_filename_regex: '.*test.*'
    llvm_profile_pattern: '%m'
    test_category: quick
```

### 3.2: Auto-Discover Test Binaries

**Update export_coverage_metadata.py:**

Add function to discover test binaries:

```python
def auto_discover_test_binaries(build_dir: Path, project_key: str) -> list:
    """Auto-discover test binaries for header-only libraries."""
    bin_dir = build_dir / 'bin'
    if not bin_dir.exists():
        return []
    
    # Common patterns for test binaries
    patterns = [
        f'test_{project_key}_*',
        f'{project_key}_test_*',
        f'*{project_key}*test*',
    ]
    
    found = []
    for pattern in patterns:
        found.extend(bin_dir.glob(pattern))
    
    # Filter executables only
    executables = [f for f in found if f.is_file() and os.access(f, os.X_OK)]
    
    return [str(f.relative_to(build_dir)) for f in executables]
```

Update `find_coverage_objects()` to use auto-discovery:

```python
def find_coverage_objects(build_dir: Path, project_key: str, config: dict) -> dict:
    project_config = config['projects'][project_key]
    
    found_objects = {'libraries': [], 'test_binaries': []}
    
    # Libraries (explicit list)
    for lib_relpath in project_config['coverage_objects']['libraries']:
        lib_path = build_dir / lib_relpath
        if lib_path.exists():
            found_objects['libraries'].append(str(lib_path))
        else:
            logging.warning(f"Library not found: {lib_path}")
    
    # Test binaries (explicit or auto-discover)
    test_binaries = project_config['coverage_objects']['test_binaries']
    
    if test_binaries:
        # Use explicit list
        for bin_relpath in test_binaries:
            bin_path = build_dir / bin_relpath
            if bin_path.exists():
                found_objects['test_binaries'].append(str(bin_path))
            else:
                logging.warning(f"Test binary not found: {bin_path}")
    else:
        # Auto-discover for header-only libraries
        logging.info(f"Auto-discovering test binaries for {project_key}")
        discovered = auto_discover_test_binaries(build_dir, project_key)
        found_objects['test_binaries'].extend(discovered)
        logging.info(f"Discovered {len(discovered)} test binaries")
    
    return found_objects
```

### 3.3: Phase 3 Validation

1. Verify auto-discovery finds all test binaries
2. Confirm coverage reports include header-only code
3. Check coverage percentages match previous CMake-based reports

## Phase 4: Complete Rollout

### 4.1: Remaining Projects

**Add to YAML:**
- rccl
- rocalution
- rocfft (if not in Phase 2)
- Any other projects from audit table

### 4.2: Codecov.io Integration

**Add step to test job:**

```yaml
- name: Upload to Codecov
  if: ${{ !github.event.pull_request.head.repo.fork }}
  uses: codecov/codecov-action@v4
  with:
    files: ${{ github.workspace }}/TheRock/build-coverage/coverage-report/coverage.info
    flags: ${{ inputs.project_name }}
    name: ${{ inputs.project_name }}-${{ inputs.amdgpu_families }}
    fail_ci_if_error: false
```

### 4.3: Cleanup Old Coverage Targets

**For each project:**
1. Verify YAML-based coverage produces identical results
2. Remove old CMake coverage targets from project CMakeLists.txt
3. Keep `<PROJECT>_ENABLE_COVERAGE` flag (still used for instrumentation)

### 4.4: Documentation

**Create:** `rocm-libraries/docs/coverage_workflow.md`

```markdown
# Coverage Workflow

## Overview

Coverage is enabled per-project via YAML configuration and runs in GitHub Actions.

## Adding Coverage to a New Project

1. Edit `test_categories_coverage.yaml`:
   - Add project key
   - List coverage objects (libraries and/or test binaries)
   - Set ignore regex for test/client code
   - Choose test category

2. Test locally:
   ```bash
   python3 .github/scripts/export_coverage_metadata.py \
     --build-dir ../TheRock/build-coverage \
     --project myproject \
     --config test_categories_coverage.yaml \
     --output coverage_metadata.json
   ```

3. Run coverage:
   ```bash
   python3 .github/scripts/coverage_runner.py \
     --build-dir ../TheRock/build-coverage \
     --metadata coverage_metadata.json \
     --coverage-dir ../TheRock/build-coverage/coverage-report
   ```

4. Push and verify in CI

## Troubleshooting

**No .profraw files generated:**
- Check `LLVM_PROFILE_FILE` is set correctly
- Verify binaries built with `-fprofile-instr-generate`
- Ensure tests actually run (check ctest output)

**Coverage report empty:**
- Verify coverage objects exist in build
- Check ignore regex isn't too broad
- Confirm .profdata file is non-zero size

**Tests fail in coverage run:**
- Coverage mode may expose race conditions
- Check test timeout settings
- Try running fewer tests in parallel
```

## Testing Strategy

### Local Testing

**Before pushing to CI:**

1. **Build with coverage:**
   ```bash
   cd /home/jorobbin/git/TheRock
   cmake -B build-coverage -GNinja \
     -DTHEROCK_ENABLE_HIPDNN=ON \
     -DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=../rocm-libraries \
     -DHIPDNN_ENABLE_COVERAGE=ON \
     -DLLVM_TOOLS_SEARCH_PREFIX=$PWD/build-coverage/dist/rocm/lib/llvm
   
   ninja -C build-coverage
   ```

2. **Export metadata:**
   ```bash
   cd /home/jorobbin/git/rocm-libraries
   python3 .github/scripts/export_coverage_metadata.py \
     --build-dir ../TheRock/build-coverage \
     --project hipdnn \
     --config test_categories_coverage.yaml \
     --output coverage_metadata.json
   
   cat coverage_metadata.json
   ```

3. **Run coverage:**
   ```bash
   python3 .github/scripts/coverage_runner.py \
     --build-dir ../TheRock/build-coverage \
     --metadata coverage_metadata.json \
     --coverage-dir ../TheRock/build-coverage/coverage-report
   
   ls -lh ../TheRock/build-coverage/coverage-report/
   ```

4. **Verify reports:**
   ```bash
   cat ../TheRock/build-coverage/coverage-report/code_cov_hipdnn.report
   firefox ../TheRock/build-coverage/coverage-report/index.html
   ```

### CI Testing

**Phase 1 checklist:**
- [ ] Create test_categories_coverage.yaml
- [ ] Add export_coverage_metadata.py
- [ ] Add coverage_runner.py
- [ ] Modify therock-ci-coverage.yml
- [ ] Push branch to rocm-libraries
- [ ] Trigger CI manually or via PR
- [ ] Check build job uploads coverage_metadata.json artifact
- [ ] Check test job downloads metadata artifact
- [ ] Verify .profraw files generated (debug step)
- [ ] Verify coverage reports uploaded
- [ ] Download and inspect coverage.info

## Rollback Plan

**If issues arise:**

1. **Revert workflow changes:**
   ```bash
   cd /home/jorobbin/git/rocm-libraries
   git revert <commit-sha>
   ```

2. **Disable in YAML:**
   ```yaml
   projects:
     hipdnn:
       enabled: false  # Temporarily disable
   ```

3. **Fall back to CMake targets:**
   - Keep old CMake coverage targets as backup
   - Remove only after YAML approach validated in production

## Migration Timeline

| Phase | Projects | Timeline | Success Metric |
|-------|----------|----------|----------------|
| Phase 1 | hipDNN | Week 1 | Coverage report generated, uploaded to codecov.io |
| Phase 2 | rocfft, rocblas, rocsparse, rocrand, hiprand | Week 2-3 | All 5 projects producing reports |
| Phase 3 | rocprim, hipcub, rocthrust, hip | Week 4 | Header-only coverage working |
| Phase 4 | Remaining projects | Week 5-6 | All 18 projects migrated |

## File Checklist

**New files to create:**

- [ ] `rocm-libraries/test_categories_coverage.yaml`
- [ ] `rocm-libraries/.github/scripts/export_coverage_metadata.py`
- [ ] `rocm-libraries/.github/scripts/coverage_runner.py`
- [ ] `rocm-libraries/docs/coverage_workflow.md`

**Files to modify:**

- [ ] `rocm-libraries/.github/workflows/therock-ci-coverage.yml` (build job: add export + upload, test job: replace reconfigure/cmake targets with coverage_runner.py)
- [ ] `rocm-libraries/.github/scripts/therock_configure_coverage.py` (load config from YAML instead of hardcoded dict)

**Files to review (no changes yet):**

- [ ] `TheRock/cmake/therock_subproject.cmake` (coverage passthrough already exists)
- [ ] `TheRock/compiler/pre_hook_amd-llvm.cmake` (LLVM tools already exported)

## Success Metrics

**Per-project:**
- Coverage percentage within ±2% of previous CMake-based reports
- .profraw file count > 1 (proves parallel test execution)
- HTML report renders correctly
- LCOV export valid for codecov.io

**Overall:**
- All 18 projects producing coverage reports
- CI runtime comparable to previous approach
- No manual CMake reconfiguration required in test job
- Coverage workflow independent of build tree structure

## Future Enhancements

**Issue #8:** Test sharding support
- Merge .profraw from multiple shards
- Aggregate coverage across shards

**Issue #10:** Windows support
- Adapt scripts for Windows paths
- Handle .exe extensions

**Issue #11:** Selective coverage builds
- Only build/test coverage-enabled projects when YAML changes
- Cache coverage metadata between runs

**Issue #12:** Coverage regression detection
- Store historical coverage percentages
- Fail CI if coverage drops >5%
