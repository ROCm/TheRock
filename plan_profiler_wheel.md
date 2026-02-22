# Plan: Create Separate Profiler Wheel for rocprof-sys and rocprof-compute

## Context

Currently, ROCm Python packaging creates these wheels:
- `rocm` (meta package, sdist only)
- `rocm-sdk-core` (runtime: HIP, compilers, rocprofiler-sdk with rocprofv3)
- `rocm-sdk-libraries-{target_family}` (math/ML libraries)
- `rocm-sdk-devel` (development files in tarball)

**Problem:** rocprofiler-systems (rocprof-sys) and rocprofiler-compute (rocprof-compute) are defined as artifacts but NOT included in any Python wheel, even though rocprofv3 (from rocprofiler-sdk) IS included.

**Requirement:** Create a new separate wheel `rocm-sdk-profiler` that ships both rocprofiler-systems and rocprofiler-compute binaries.

## Artifacts Analysis

From BUILD_TOPOLOGY.toml:

1. **rocprofiler-sdk** (already in core package)
   - artifact_group: profiler-core
   - Contains: rocprofv3, rocprof-attach, rocprofiler SDK libraries
   - Status: ✅ Already included in rocm-sdk-core

2. **rocprofiler-compute**
   - artifact_group: profiler-core
   - Contains: rocprof-compute binaries and tools
   - Status: ❌ NOT included in any wheel

3. **rocprofiler-systems**
   - artifact_group: profiler-apps
   - Contains: rocprof-sys-* binaries (rocprof-sys-avail, rocprof-sys-instrument, rocprof-sys-run, rocprof-sys-sample, rocprof-sys-causal)
   - Status: ❌ NOT included in any wheel

## Implementation Plan

### Phase 1: Define New Package Entry

**File:** `/root/TheRock/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py`

**Changes:**
1. Add new PackageEntry for profiler package (after line ~205):
```python
PackageEntry(
    "profiler",
    "rocm-sdk-profiler",
    pure_py_package_name="rocm_sdk_profiler",
    template_directory="rocm-sdk-profiler",
    required=False,
)
```

### Phase 2: Create Package Template

**Directory:** `/root/TheRock/build_tools/packaging/python/templates/rocm-sdk-profiler/`

**Structure:**
```
rocm-sdk-profiler/
├── MANIFEST.in           # Copy from rocm-sdk-core, adapt
├── pyproject.toml        # Copy from rocm-sdk-core, adapt
├── README.md             # New file describing profiler tools
├── setup.py              # New file with console_scripts for profiler tools
└── src/
    └── rocm_sdk_profiler/
        ├── __init__.py   # Empty or minimal
        └── _cli.py       # Console script trampolines
```

**Key Files to Create:**

1. **setup.py** - Similar to rocm-sdk-core but with profiler-specific entry points:
```python
setup(
    name="rocm-sdk-profiler",
    version=dist_info.__version__,
    # ... other fields ...
    entry_points={
        "console_scripts": [
            # Only rocprofiler-systems and rocprofiler-compute tools
            # Note: rocprofv3 and rocprof-attach stay in rocm-sdk-core
            "rocprof-compute=rocm_sdk_profiler._cli:rocprof_compute",
            "rocprof-sys-avail=rocm_sdk_profiler._cli:rocprof_sys_avail",
            "rocprof-sys-causal=rocm_sdk_profiler._cli:rocprof_sys_causal",
            "rocprof-sys-instrument=rocm_sdk_profiler._cli:rocprof_sys_instrument",
            "rocprof-sys-run=rocm_sdk_profiler._cli:rocprof_sys_run",
            "rocprof-sys-sample=rocm_sdk_profiler._cli:rocprof_sys_sample",
        ]
    },
    # NO install_requires - following convention!
    # Dependency managed by meta package (rocm)
)
```

2. **_cli.py** - Trampolines to platform binaries:
```python
def _exec(relpath: str):
    full_path = PLATFORM_PATH / (relpath + exe_suffix)
    os.execv(full_path, [str(full_path)] + sys.argv[1:])

def rocprof_compute():
    _exec("bin/rocprof-compute")

def rocprof_sys_avail():
    _exec("bin/rocprof-sys-avail")

def rocprof_sys_causal():
    _exec("bin/rocprof-sys-causal")

def rocprof_sys_instrument():
    _exec("bin/rocprof-sys-instrument")

def rocprof_sys_run():
    _exec("bin/rocprof-sys-run")

def rocprof_sys_sample():
    _exec("bin/rocprof-sys-sample")
```

### Phase 3: Update Build Script

**File:** `/root/TheRock/build_tools/build_python_packages.py`

**Changes:**

1. Add profiler_artifact_filter function (after line ~127):
```python
def profiler_artifact_filter(an: ArtifactName) -> bool:
    profiler = an.name in [
        "rocprofiler-compute",
        "rocprofiler-systems",
    ] and an.component in [
        "lib",
        "run",
    ]
    return profiler
```

2. Create and populate profiler package in run() function (after line ~63):
```python
# Populate profiler package (target-neutral)
profiler = PopulatedDistPackage(params, logical_name="profiler")
profiler.rpath_dep(core, "lib")
profiler.populate_runtime_files(
    params.filter_artifacts(
        profiler_artifact_filter,
    )
)
```

### Phase 4: Update Meta Package (rocm)

**File:** `/root/TheRock/build_tools/packaging/python/templates/rocm/setup.py`

**Changes:**
- No changes needed! The setup.py already dynamically generates extras_require from ALL_PACKAGES
- The new profiler package will automatically appear as `rocm[profiler]`

### Phase 5: Keep rocprofiler-sdk in Core Package

**USER DECISION:** Keep rocprofv3 in core, add rocprofiler-systems/compute to profiler

**Implementation:**
- Keep rocprofiler-sdk in core_artifact_filter (NO CHANGES to core filter)
- profiler package includes ONLY rocprofiler-systems and rocprofiler-compute
- rocprofv3, rocprof-attach remain in rocm-sdk-core
- No breaking changes for existing users
- NO install_requires dependency - follow existing convention (see Phase 6)

This means:
- `rocm-sdk-core` provides: rocprofv3, rocprof-attach, rocprofiler SDK libs, roctx API
- `rocm-sdk-profiler` provides: rocprof-sys-*, rocprof-compute

**Why rocprofiler-sdk stays in core:**
- Provides roctx API (user-facing annotation library used by applications like PyTorch)
- Root of the profiler dependency DAG (both compute and systems depend on it)
- Non-optional infrastructure component

### Phase 6: Follow Meta Package Convention

**DESIGN DECISION:** Do NOT add install_requires to rocm-sdk-profiler

**Rationale:**
- Existing packages (rocm-sdk-libraries, rocm-sdk-devel) do NOT declare install_requires
- All dependency management is handled by the meta package (`rocm`)
- Users install via `pip install rocm[profiler]`, NOT `pip install rocm-sdk-profiler` directly
- Meta package ensures rocm-sdk-core is always installed (via `required=True`)
- Packages use RPATH to find dependencies, expecting co-installation in same site-packages

**How it works:**
1. User runs: `pip install rocm[profiler]`
2. Meta package installs rocm-sdk-core (from install_requires, because core has required=True)
3. Meta package installs rocm-sdk-profiler (from extras_require["profiler"])
4. Both packages installed in same site-packages, RPATH resolves dependencies

**No changes needed to setup.py** - just omit install_requires like other packages do.

### Phase 7: Documentation Updates

**Files to update:**
1. `/root/TheRock/docs/packaging/python_packaging.md`
   - Document new `rocm-sdk-profiler` package
   - Update installation examples to show `rocm[profiler]`

2. `/root/TheRock/RELEASES.md` or changelog
   - Announce new profiler package

## Summary of Files to Modify/Create

### Files to Modify:
1. `/root/TheRock/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py`
   - Add PackageEntry for "profiler"

2. `/root/TheRock/build_tools/build_python_packages.py`
   - Add profiler_artifact_filter()
   - Add profiler package population in run()

### Files to Create:
1. `/root/TheRock/build_tools/packaging/python/templates/rocm-sdk-profiler/pyproject.toml`
2. `/root/TheRock/build_tools/packaging/python/templates/rocm-sdk-profiler/setup.py`
3. `/root/TheRock/build_tools/packaging/python/templates/rocm-sdk-profiler/MANIFEST.in`
4. `/root/TheRock/build_tools/packaging/python/templates/rocm-sdk-profiler/README.md`
5. `/root/TheRock/build_tools/packaging/python/templates/rocm-sdk-profiler/src/rocm_sdk_profiler/__init__.py`
6. `/root/TheRock/build_tools/packaging/python/templates/rocm-sdk-profiler/src/rocm_sdk_profiler/_cli.py`

### Files to Update (Documentation):
1. `/root/TheRock/docs/packaging/python_packaging.md`

## Testing Plan

After implementation:

1. **Build packages:**
```bash
python ./build_tools/build_python_packages.py \
    --artifact-dir ./output-linux-portable/build/artifacts \
    --dest-dir /tmp/test-packages
```

2. **Verify wheel created:**
```bash
ls /tmp/test-packages/dist/rocm_sdk_profiler-*.whl
```

3. **Install and test:**
```bash
python -m venv test-venv
source test-venv/bin/activate
pip install rocm[profiler] --find-links=/tmp/test-packages/dist
which rocprof-sys-run
which rocprof-compute
which rocprofv3
```

4. **Verify binaries work:**
```bash
rocprofv3 --help
rocprof-sys-run --help
rocprof-compute --help
```

## Decisions Made

1. ✅ **rocprofv3 location:** Keep in rocm-sdk-core (backward compatible)
   - Rationale: rocprofiler-sdk provides roctx API used by applications, must remain in core

2. ✅ **Dependencies:** NO install_requires in rocm-sdk-profiler
   - Rationale: Follow existing convention - meta package handles all dependencies

3. ✅ **Artifacts:** Include rocprofiler-systems and rocprofiler-compute only
   - rocprofv3, rocprof-attach stay in core with rocprofiler-sdk

## Installation Pattern

Users will install via:
```bash
pip install rocm[profiler]
# Or combined with other extras:
pip install rocm[libraries,devel,profiler]
```

NOT:
```bash
pip install rocm-sdk-profiler  # ❌ Not recommended - bypasses meta package
```
