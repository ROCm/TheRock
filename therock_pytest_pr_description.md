## Motivation

This PR extends TheRock's test filter standardization to support pytest-based components (Tensile and Tensilite). Currently, test filter standardization only works for CTest-based components via `test_runner.py`. This PR adds equivalent support for pytest-based components, enabling consistent test categorization across both testing frameworks.

This work achieves test parity between Math CI (Jenkins) and TheRock CI (GitHub Actions) for pytest-based components by implementing the same 4-category test filter model (quick, standard, comprehensive, full) using pytest markers instead of CTest labels.

## Technical Details

This PR implements a new `pytest_runner.py` that mirrors the architecture of the existing `test_runner.py` for CTest components. The implementation follows established TheRock patterns and reuses the `test_categories.yaml` configuration format.

**Changes:**

1. **`build_tools/github_actions/test_executable_scripts/pytest_runner.py`** (new file - 6 commits)
   - Implements pytest test collection and sharding (modulo arithmetic matching GTest behavior)
   - Parses `test_categories.yaml` to build pytest marker expressions
   - Adds GPU architecture filtering via `-m "not skip-gfxXXXX"` markers
   - Executes pytest with category-based timeouts and parallel workers
   - Supports both source-based and installed test execution

2. **`build_tools/github_actions/fetch_test_configurations.py`** (modified - 2 commits)
   - Added Tensile configuration with 1/4/8/8 sharding for quick/standard/comprehensive/full
   - Added Tensilite configuration with 1/2/4/4 sharding
   - Both components use `pytest_runner.py` as test script

**Key Design Decisions:**

- **Source-based execution**: Tensile and Tensilite tests run from the source tree (`GITHUB_WORKSPACE/shared/tensile` and `GITHUB_WORKSPACE/projects/hipblaslt/tensilite`) instead of installed artifacts. This matches Jenkins behavior and avoids CMake installation complexity.

- **Pytest sharding implementation**: Since pytest doesn't have built-in shard support like GTest (`GTEST_SHARD_INDEX`), we implement it manually:
  1. Collect all test IDs using `pytest --collect-only`
  2. Filter by index: `test_index % TOTAL_SHARDS == (SHARD_INDEX - 1)`
  3. Run pytest with specific test IDs

- **GPU isolation**: Three-level architecture (unchanged):
  - Level 1: Runner-level via `/etc/podinfo/gha-gpu-isolation-settings` (each shard gets isolated GPUs)
  - Level 2: Test sharding via `SHARD_INDEX`/`TOTAL_SHARDS` (this PR implements for pytest)
  - Level 3: Pytest-xdist worker assignment via `HIP_VISIBLE_DEVICES` (already in conftest.py)

- **Marker expression building**: Maps `test_categories.yaml` pytest_markers to pytest `-m` expressions:
  - Single marker: `-m "unit"`
  - Multiple markers: `-m "pre_checkin or extended"`
  - With exclusions: `-m "(pre_checkin or extended) and not disabled"`
  - With GPU skip: `-m "(pre_checkin or extended) and not disabled and not skip-gfx1151"`

**Related work:**

This PR works in conjunction with rocm-libraries PR (branch: pytest-test-filter-support) which adds:
- `shared/tensile/test_categories.yaml` - Tensile test category configuration
- `projects/hipblaslt/tensilite/test_categories.yaml` - Tensilite test category configuration

**Commit breakdown:**

1. `5046cdc754` - Add pytest test collection and sharding utility
2. `af76ed90c6` - Add test_categories.yaml parser for pytest markers
3. `e22e025ccf` - Add GPU architecture marker filtering for pytest
4. `2f02b94af5` - Add pytest execution with category-based configuration
5. `6d355a883a` - Add Tensile pytest test configuration
6. `310337ade9` - Add Tensilite pytest test configuration
7. `f2f1347256` - Support source-based pytest test execution
8. `d4e54c7eb2` - Add Tensilite as source-based pytest component

## Test Plan

**Local verification (Tensile):**
```bash
# Setup environment
export TEST_COMPONENT=tensile
export TEST_TYPE=standard
export AMDGPU_FAMILIES=gfx1151
export GITHUB_WORKSPACE=/home/jorobbin/git/rocm-libraries
export SHARD_INDEX=1
export TOTAL_SHARDS=4
export THEROCK_BIN_DIR=/opt/rocm/bin

# Test pytest_runner.py
cd /home/jorobbin/git/TheRock
python3 build_tools/github_actions/test_executable_scripts/pytest_runner.py

# Expected behavior:
# - Loads test_categories.yaml from shared/tensile/
# - Builds marker expression: "-m 'pre_checkin and not disabled and not skip-gfx1151 and not skip-gfx115X and not skip-gfx11X'"
# - Collects all tests matching markers
# - Filters to shard 1 of 4 (modulo arithmetic)
# - Runs pytest with --numprocesses=4 --timeout=1800
```

**Local verification (Tensilite):**
```bash
export TEST_COMPONENT=tensilite
export TEST_TYPE=quick
export GITHUB_WORKSPACE=/home/jorobbin/git/rocm-libraries

python3 build_tools/github_actions/test_executable_scripts/pytest_runner.py

# Expected: runs unit tests from projects/hipblaslt/tensilite/Tensile/Tests
```

**CI verification (after both PRs merge):**
1. TheRock CI will automatically detect Tensile/Tensilite in `projects_to_test`
2. Tests execute in correct categories via pytest markers
3. Sharding distributes tests across multiple runners
4. GPU skip markers prevent incompatible tests from running

## Test Result

- Pytest runner logic verified locally with dry-run test collection
- Marker expression building tested with various category configurations
- GPU skip marker hierarchical filtering confirmed (gfx1151 → skip-gfx1151, skip-gfx115X, skip-gfx11X)
- Source-based path resolution verified for both Tensile and Tensilite
- Sharding math verified (modulo distribution matches GTest behavior)
- Full CI validation pending rocm-libraries PR merge

## Submission Checklist

- [x] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.
