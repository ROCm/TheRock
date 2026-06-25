# Build Infrastructure Components

## Purpose

The build infrastructure provides the CMake modules, Python build tools, and orchestration systems that enable TheRock's artifact-based super-project build system. It handles dependency management, artifact packaging, CI/CD integration, and topology-driven configuration.

## Location

- **Primary directory**: `cmake/` (CMake modules), `build_tools/` (Python scripts)
- **Key files**:
  - `BUILD_TOPOLOGY.toml`: Single source of truth for build structure (770 lines)
  - `CMakeLists.txt`: Top-level CMake orchestration
  - `cmake/therock_subproject.cmake`: Subproject management (81 KB - largest module)
  - `build_tools/topology_to_cmake.py`: Topology → CMake generator
  - `build_tools/_therock_utils/build_topology.py`: Topology parser

## Dependencies

### Used By
- All artifact groups depend on this infrastructure
- CI/CD workflows use `build_tools/github_actions/` scripts
- Packaging system uses `build_tools/packaging/` modules

### Depends On
- **Python 3.9+**: Required for build scripts
- **CMake 3.25+**: Minimum version for super-project
- **Git**: Submodule management
- **Ninja**: Recommended build system (faster than Make)

## CMake Infrastructure (`cmake/`)

### Core Modules (21 files)

**Global Configuration:**
- `therock_globals.cmake` - Global variables, paths, and definitions
- `therock_compiler_config.cmake` - Compiler detection and flag setup
- `therock_amdgpu_targets.cmake` - GPU architecture target handling
- `therock_features.cmake` - Feature flag system (THEROCK_ENABLE_* variables)
- `therock_flag_utils.cmake` - Compiler flag manipulation utilities

**Subproject Orchestration:**
- `therock_subproject.cmake` - **Core orchestration logic** (81 KB)
  - Implements `therock_add_subproject()` macro
  - Manages 4-phase build: configure → build → stage → dist
  - Handles ExternalProject setup with custom targets
  - Coordinates dependency resolution between components
- `therock_subproject_dep_provider.cmake` - CMake dependency provider for subprojects
- `therock_global_post_subproject.cmake` - Post-build processing

**Artifact System:**
- `therock_artifacts.cmake` - Artifact management and packaging
- `therock_default_targets.cmake` - Standard targets (expunge, archives, etc.)

**Testing:**
- `therock_testing.cmake` - Test infrastructure setup
- `therock_sanitizers.cmake` - AddressSanitizer, UBSan, etc.

**Build Optimization:**
- `therock_job_pools.cmake` - Ninja job pool configuration for parallel builds
- `therock_install_linux_build_id_files.cmake` - Build ID tracking

**External Integration:**
- `therock_external_source.cmake` - External source management
- `therock_explicit_finders.cmake` - Package finder overrides
- `therock_meson_env.cmake` - Meson build system integration

**Python Support:**
- `therock_detect_python_versions.cmake` - Python version detection for wheels

**Finder Modules:**
- `FindBLAS.cmake` - BLAS provider detection
- `FindLAPACK.cmake` - LAPACK provider detection

### CMake Subdirectories

- `cmake/finders/` - Custom Find modules for dependencies
- `cmake/modules/` - Additional utility modules
- `cmake/toolchains/` - Toolchain files (e.g., `linux-amdclang.cmake`)
- `cmake/templates/` - Template files for code generation

## Python Build Tools (`build_tools/`)

### Topology System

**Core Scripts:**
- `topology_to_cmake.py` - Reads BUILD_TOPOLOGY.toml, generates CMake targets
  - Creates `THEROCK_ENABLE_*` variables for each artifact
  - Generates `therock_topology.cmake` with dependency information
  - Produces artifact group targets
- `build_topology.py` - Parses topology for CI sharding
  - Computes artifact dependencies
  - Determines which artifacts to fetch before each build stage
- `configure_stage.py` - Stage configuration for CI/CD

### Artifact Management

**Build & Package:**
- `artifact_manager.py` - Core artifact operations
- `build_tarballs.py` - Package artifacts into `.tar.xz` archives
- `generate_therock_manifest.py` - Generate component manifest
- `generate_manifest_diff_report.py` - Generate diff reports
- `generate_s3_index.py` - Generate S3 indexes for artifact storage

**Fetch & Retrieve:**
- `fetch_artifacts.py` - Download pre-built artifacts from storage
- `fetch_sources.py` - Clone git submodules and apply patches
  - **Destructive**: Resets all submodules and reapplies patches
  - Supports partial checkouts via `--stage` flag
  - Use `git reflog` to recover lost work after running
- `find_artifacts_for_commit.py` - Locate artifacts by commit hash
- `find_latest_artifacts.py` - Find latest artifact versions

**Validation:**
- `validate_shared_library.py` - ELF library validation
- `patch_linux_so.py` - ELF patcher for RPATH/RUNPATH
- `patch_rocm_libraries.py` - ROCm-specific library patches

### Build Control

- `buildctl.py` - Main build control script
- `analyze_build_times.py` - Build performance analysis
- `memory_monitor.py` - Memory usage tracking during builds

### Git & Submodule Management

- `setup_git_mirrors.py` - Git mirror management for faster clones
- `bump_submodules.py` - Submodule version bumping utilities

### Python Packaging

- `build_python_packages.py` - Build Python wheel packages
- `fileset_tool.py` - File set management for packaging

### CI/CD Scripts (`build_tools/github_actions/`)

**Configuration:**
- `configure_ci.py` - CI pipeline configuration
- `configure_multi_arch_ci.py` - Multi-architecture CI setup
- `configure_ci_path_filters.py` - Path-based CI filtering
- `configure_target_run.py` - Target run configuration
- `determine_version.py` - Version determination from git

**Artifact Upload:**
- `post_build_upload.py` - Upload artifacts after build
- `post_stage_upload.py` - Upload artifacts after stage

**External Integration:**
- `generate_pytorch_manifest.py` - PyTorch manifest generation
- `generate_jax_manifest.py` - JAX manifest generation
- `detect_external_repo_config.py` - External repository detection
- Multiple manifest handlers for external dependencies

**Testing:**
- `fetch_test_configurations.py` - Fetch test configurations
- `tests/pytest_runner_test.py` - Pytest runner testing

### Utility Library (`build_tools/_therock_utils/`)

**Core Utilities (14 modules):**
- `build_topology.py` - Topology parsing and data structures
- `artifacts.py` - Artifact data structures and operations
- `artifact_builder.py` - Artifact building logic
- `artifact_backend.py` - Storage backend abstraction
- `storage_backend.py` - Storage operations (S3, local)
- `storage_location.py` - Storage location handling
- `cmake_amdgpu_targets.py` - GPU target list generation
- `git_mirrors.py` - Git mirror operations
- `hash_util.py` - Hash utilities for artifact verification
- `pattern_match.py` - Pattern matching utilities
- `py_packaging.py` - Python packaging utilities
- `workflow_outputs.py` - GitHub Actions workflow output management
- `exe_stub_gen.py` - Executable stub generation for Python packages

### Packaging Infrastructure (`build_tools/packaging/`)

- `linux/` - Linux packaging scripts (DEB/RPM)
- `python/` - Python wheel generation
- `tests/` - Packaging tests
- `generate_local_index.py` - Local package index generation

### Additional Tools

- `hack/env_check/` - Environment validation scripts
- `hack/ccache/` - ccache integration utilities
- `third_party/` - Third-party tools
  - `implib/` - Import library tools
  - `s3_management/` - S3 operations
  - `change_wheel_version/` - Wheel version manipulation

## Entry Points

### CMake Entry Points

```cmake
# Top-level CMakeLists.txt includes:
include(therock_globals)
include(therock_features)
include(therock_subproject)
include(therock_artifacts)
# ... 15+ modules total

# Components use:
therock_add_subproject(
  NAME component-name
  SOURCE_DIR path/to/source
  CMAKE_ARGS "-DOPTION=VALUE"
  ARTIFACT_GROUP group-name
)
```

### Python Entry Points

```bash
# Topology generation (run during CMake configure)
python3 build_tools/topology_to_cmake.py \
  --topology BUILD_TOPOLOGY.toml \
  --output build/cmake/therock_topology.cmake

# Fetch sources
python3 build_tools/fetch_sources.py  # All submodules
python3 build_tools/fetch_sources.py --stage math-libs  # Partial

# Artifact management
python3 build_tools/build_tarballs.py --component rocblas
python3 build_tools/fetch_artifacts.py --artifact rocblas --commit abc123

# CI configuration
python3 build_tools/github_actions/configure_ci.py
```

## BUILD_TOPOLOGY.toml Schema

### Structure

```toml
[metadata]
version = "2.0"
description = "TheRock artifact-based build topology"

[source_sets.<name>]
description = "Description"
submodules = ["submodule1", "submodule2"]
disable_platforms = ["windows"]  # Optional

[build_stages.<name>]
description = "Description"
artifact_groups = ["group1", "group2"]
type = "generic" | "per-arch"

[artifact_groups.<name>]
description = "Description"
type = "generic" | "per-arch"
artifact_group_deps = ["dep1", "dep2"]
source_sets = ["set1", "set2"]

[artifacts.<name>]
artifact_group = "group-name"
type = "target-neutral" | "target-specific"
artifact_deps = ["dep1", "dep2"]
feature_name = "FEATURE_NAME"  # Optional override
platform = "windows"  # Optional
disable_platforms = ["windows"]  # Optional
python_requires = ["-r requirements.txt", "pkg"]
split_databases = ["rocblas", "hipblaslt"]  # For kpack
```

### Naming Conventions

- **Entity names**: lowercase-with-hyphens (e.g., "core-runtime")
- **feature_name**: UPPERCASE_WITH_UNDERSCORES (e.g., "CORE_RUNTIME")
- **type**: lowercase (e.g., "generic", "per-arch", "target-neutral")
- **platform**: lowercase (e.g., "windows", "linux")

## Component Build Lifecycle

### 4-Phase Build Process

1. **Configure Phase**
   - CMake runs `ExternalProject_Add()` with component-specific args
   - `pre_hook_*.cmake` scripts run before configuration
   - Component's `CMakeLists.txt` is configured

2. **Build Phase**
   - Component source is compiled
   - Libraries and executables are built
   - Tests are built if `THEROCK_BUILD_TESTING=ON`

3. **Stage Phase**
   - Component artifacts installed to `build/component/stage/`
   - Isolated install tree (this component only)
   - `CMAKE_INSTALL_PREFIX` points to stage directory

4. **Dist Phase**
   - Artifacts copied to `build/component/dist/`
   - Dependencies merged from other component dist directories
   - `post_hook_*.cmake` scripts run after dist
   - Final output: `build/dist/rocm/` (unified ROCm installation)

### Target Naming Pattern

```
component              # Full build (all 4 phases)
component+build        # Re-run build phase only
component+stage        # Re-run stage phase
component+dist         # Re-run dist phase
component+expunge      # Clean all artifacts and rebuild
```

## Patterns

### Subproject Hook Pattern

Components can define pre/post hooks:

```cmake
# In component directory: pre_hook_rocblas.cmake
function(rocblas_pre_hook)
  set(ROCBLAS_CUSTOM_ARG "value" PARENT_SCOPE)
endfunction()

# In component directory: post_hook_rocblas.cmake
function(rocblas_post_hook)
  # Custom post-dist operations
  message(STATUS "rocBLAS dist complete")
endfunction()
```

### Artifact TOML Pattern

Each component defines artifact metadata:

```toml
# artifact-rocblas.toml
[artifact]
name = "rocblas"
version = "6.5.0"
description = "ROCm BLAS library"

[files]
patterns = [
  "lib/librocblas.so*",
  "include/rocblas/**",
  "bin/rocblas-test"
]

[dependencies]
runtime = ["hip-clr", "rocr-runtime"]
build = ["rocm-cmake"]
```

### Feature Flag Generation

topology_to_cmake.py generates:

```cmake
# From artifact named "rocblas" in group "math-libs"
therock_add_feature(
  FEATURE ROCBLAS
  GROUP MATH_LIBS
  DEFAULT ON
  DESCRIPTION "ROCm BLAS library"
)
# Creates: THEROCK_ENABLE_ROCBLAS cache variable
# Creates: THEROCK_ENABLE_MATH_LIBS group option
```

### Dependency Provider Pattern

```cmake
# therock_subproject_dep_provider.cmake
# Intercepts find_package() calls
# Redirects to component dist directories
set(CMAKE_FIND_PACKAGE_REDIRECTS_DIR
  "${THEROCK_DIST_DIR}/lib/cmake")
```

## Key Constraints

1. **Topology is authoritative**: All artifact structure must be in BUILD_TOPOLOGY.toml
2. **No in-source builds**: Build directory must be separate from source
3. **Python 3.9+**: Required for all build scripts
4. **CMake 3.25+**: Minimum version for super-project features
5. **Destructive fetch**: `fetch_sources.py` resets all submodules
6. **Build order**: Determined by topology dependencies, not directory order

## Testing

Build infrastructure has limited direct tests:

- `build_tools/github_actions/tests/pytest_runner_test.py` - Tests pytest runner
- Integration tests via full builds in CI
- Validation through successful component builds

## Future Enhancements

From RFCs and development docs:

- **Meson support**: Better integration with Meson-based subprojects
- **Native packaging**: DEB/RPM improvements, Windows MSI support
- **Partial topology**: Build only changed artifacts based on git diff
- **Remote caching**: Share build artifacts across developers
- **Monorepo gardening**: Automated submodule updates and patch management
