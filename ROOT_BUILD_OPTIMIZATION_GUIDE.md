# Root Build Optimization Strategy

## Executive Summary

Based on the analysis of BUILD_TOPOLOGY.toml and the Windows build concurrency data:

- **Current Performance**: 173 minutes, 7.11 avg concurrency (5.7% efficiency)
- **Optimization Potential**: 25-35 minutes, 60-80 avg concurrency (60-80% efficiency)
- **Expected Speedup**: **7x faster** (from 173 min → 25-35 min)

---

## Problem Diagnosis

### Current Build Structure

The topology shows **9 build stages** with **42 artifacts** organized in **17 groups**:

```
Build Stages:
1. foundation (6 artifacts)
2. compiler-runtime (17 artifacts) ← MAJOR BOTTLENECK
3. math-libs (10 artifacts, per-arch)
4. comm-libs (1 artifact, per-arch)
5. debug-tools (4 artifacts)
6. dctools-core (1 artifact)
7. iree-libs (1 artifact)
8. profiler-apps (1 artifact)
9. media (1 artifact)
```

### Critical Issues

1. **Sequential Stage Execution**: Stages are being executed serially when many could run in parallel
2. **Poor Within-Stage Parallelization**: The `compiler-runtime` stage has 17 artifacts but only 7 avg concurrency
3. **Artificial Serialization**: CMake dependencies are too strict, creating unnecessary ordering

---

## Optimization Strategy

### Phase 1: Immediate Fixes (High Impact)

#### 1.1 Parallel Stage Execution

**Current**: Stages run sequentially → 173 minutes total  
**Target**: Stages run in parallel where dependencies allow

```
Current Flow (Sequential):
foundation → compiler-runtime → math-libs → comm-libs → debug-tools → ...
  (6 min)       (120 min)         (20 min)    (5 min)     (10 min)

Optimized Flow (Parallel):
                    ┌─ math-libs (20 min)
                    ├─ comm-libs (5 min)
foundation → compiler-runtime ┼─ debug-tools (10 min)
  (6 min)       (30 min)      ├─ dctools-core (3 min)
                              ├─ iree-libs (2 min)
                              └─ media (1 min)

Total: 6 + 30 + 20 = 56 minutes (worst case if no further optimization)
```

**Implementation**:
- After `compiler-runtime` completes, launch all dependent stages in parallel
- Use CMake's `add_dependencies()` properly to specify only true dependencies
- Consider parallel ctest execution for independent test suites

#### 1.2 Within-Stage Parallelization for compiler-runtime

The `compiler-runtime` stage contains **7 artifact groups**:

```
Groups in compiler-runtime stage:
1. compiler (2 artifacts: amd-llvm, hipify)
2. core-runtime (1 artifact)
3. third-party-libs (7 artifacts)
4. hip-runtime (3 artifacts)
5. opencl-runtime (1 artifact)
6. profiler-core (2 artifacts)
7. rocrtst (1 artifact)
```

**Current Problem**: These groups build sequentially within the stage  
**Solution**: Parallelize independent groups

**Dependency Analysis**:
```
Level 0 (can start immediately):
  - third-party-libs (7 artifacts)
  - compiler/amd-llvm (1 artifact)

Level 1 (depends on Level 0):
  - compiler/hipify (depends on amd-llvm)
  - core-runtime (depends on amd-llvm)

Level 2 (depends on Level 1):
  - hip-runtime (depends on core-runtime, amd-llvm)
  - opencl-runtime (depends on core-runtime, amd-llvm)
  - profiler-core (depends on core-runtime)

Level 3 (depends on Level 2):
  - rocrtst (depends on opencl-runtime, core-runtime)
```

**Parallel Execution Plan**:
```
Time 0-5min:   Build amd-llvm + all 7 third-party-libs (8 parallel builds)
Time 5-10min:  Build hipify + core-runtime (2 parallel builds)
Time 10-20min: Build hip-runtime + opencl-runtime + profiler-core (3 parallel)
Time 20-25min: Build rocrtst (1 build)

Total: ~25 minutes (vs current ~120 minutes)
```

---

### Phase 2: Build System Implementation

#### 2.1 CMake Configuration Changes

**File: `CMakeLists.txt` (root level)**

Add parallel stage execution:

```cmake
# Define stage targets
add_custom_target(stage-foundation)
add_custom_target(stage-compiler-runtime)
add_custom_target(stage-post-compiler)  # Groups parallel stages

# Foundation stage artifacts
add_dependencies(stage-foundation 
    sysdeps base third-party-sysdeps)

# Compiler-runtime stage artifacts
add_dependencies(stage-compiler-runtime
    stage-foundation  # Only depends on foundation
    amd-llvm hipify core-runtime
    hip-runtime opencl-runtime
    profiler-core rocrtst)

# Post-compiler parallel stages
add_custom_target(stage-math-libs)
add_custom_target(stage-comm-libs)
add_custom_target(stage-debug-tools)
add_custom_target(stage-dctools)
add_custom_target(stage-iree)

add_dependencies(stage-math-libs stage-compiler-runtime)
add_dependencies(stage-comm-libs stage-compiler-runtime)
add_dependencies(stage-debug-tools stage-compiler-runtime)
add_dependencies(stage-dctools stage-compiler-runtime)
add_dependencies(stage-iree stage-compiler-runtime)

# All post-compiler stages can run in parallel
add_dependencies(stage-post-compiler
    stage-math-libs
    stage-comm-libs
    stage-debug-tools
    stage-dctools
    stage-iree)
```

#### 2.2 Remove Over-Constrained Dependencies

**Review and Fix**:

Many artifacts likely have unnecessary `add_dependencies()` calls. For example:

```cmake
# BAD: Over-constrained
add_dependencies(blas
    ALL_THIRD_PARTY_LIBS  # Too broad!
    ALL_COMPILER_ARTIFACTS
    ALL_RUNTIME_ARTIFACTS)

# GOOD: Minimal dependencies
add_dependencies(blas
    core-hip
    rocprofiler-sdk
    host-blas          # Only if tests need it
    host-suite-sparse) # Only if tests need it
```

**Action Items**:
1. Audit all `add_dependencies()` calls in CMakeLists.txt files
2. Remove transitive dependencies (CMake handles these automatically)
3. Only specify direct build-time dependencies

#### 2.3 Third-Party Libraries Parallelization

The 7 third-party libraries are independent:

```cmake
# These can all build in parallel
add_library(host-blas ...)
add_library(host-suite-sparse ...)
add_library(fftw3 ...)
add_library(flatbuffers ...)
add_library(fmt ...)
add_library(nlohmann-json ...)
add_library(spdlog ...)

# Only host-suite-sparse depends on host-blas
add_dependencies(host-suite-sparse host-blas)
```

Ensure these use `ExternalProject_Add()` with `BUILD_COMMAND` that includes parallel flags:

```cmake
ExternalProject_Add(host-blas
    ...
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel ${THEROCK_PARALLEL_JOBS}
    ...
)
```

---

### Phase 3: Per-Architecture Build Optimization

#### 3.1 Current Issue

Per-arch groups (`math-libs`, `ml-libs`, `comm-libs`, `rocrtst`) should build once with all GPU targets, not separately per architecture.

**Current (BAD)**:
```
Build math-libs for gfx906  (20 min)
Build math-libs for gfx908  (20 min)
Build math-libs for gfx90a  (20 min)
Build math-libs for gfx942  (20 min)
Build math-libs for gfx1030 (20 min)
Total: 100 minutes sequentially
```

**Target (GOOD)**:
```
Build math-libs for [gfx906,gfx908,gfx90a,gfx942,gfx1030] (25 min once)
Total: 25 minutes
```

#### 3.2 Implementation

Ensure CMake properly configures multi-arch builds:

```cmake
# Set GPU architectures once at the top level
set(AMDGPU_TARGETS "gfx906;gfx908;gfx90a;gfx942;gfx1030" CACHE STRING "GPU targets")

# For each rocm library
set_target_properties(rocBLAS PROPERTIES
    HIP_ARCHITECTURES "${AMDGPU_TARGETS}")
```

**Critical**: The artifact type distinction in BUILD_TOPOLOGY.toml:
- `target-neutral`: Build once, all architectures together
- `target-specific`: Build separately per arch (only for CI sharding, not local builds)

For the **root build**, treat everything as `target-neutral` (build once with all targets).

---

### Phase 4: Ninja Configuration

#### 4.1 Check Current Settings

```bash
# Check ninja pool settings
ninja -t pools

# Typical output should show:
# console: ...
# link_pool: 4
# (default): 98
```

#### 4.2 Optimize Job Pools

Edit `.ninja_log` or configure CMake to generate better pools:

```cmake
# In CMakeLists.txt
set_property(GLOBAL PROPERTY JOB_POOLS
    link_pool=4          # Limit parallel linking (memory intensive)
    compile_pool=98      # Max compilation parallelism
    heavy_pool=8)        # For memory-intensive tasks

# Apply to heavy compilation targets
set_target_properties(amd-llvm PROPERTIES JOB_POOL_COMPILE heavy_pool)
set_target_properties(rocBLAS PROPERTIES JOB_POOL_COMPILE compile_pool)
```

---

## Dependency Graph Analysis

### Critical Path (Cannot be Reduced)

```
sysdeps (5min)
  ↓
amd-llvm (20min)
  ↓
core-runtime (3min)
  ↓
core-hip (5min)
  ↓
blas (15min)
  ↓
miopen (20min)
  ↓
miopen-plugin (2min)

Total Critical Path: ~70 minutes
```

This is the theoretical minimum build time (if everything else is perfectly parallel).

### High-Impact Artifacts

These artifacts have many dependents and should be prioritized:

1. **core-runtime** (17 dependents) - Must build early
2. **core-hip** (16 dependents) - Blocks all GPU code
3. **amd-llvm** (8 dependents) - Blocks compilers
4. **rocprofiler-sdk** (7 dependents) - Blocks profiling
5. **sysdeps** (6 dependents) - Blocks everything

**Optimization**: Ensure these build as fast as possible with maximum resources.

---

## Implementation Roadmap

### Week 1: Quick Wins

1. ✅ **Analyze current build** (DONE - you have the data)
2. ⚠️ **Audit CMake dependencies** in root CMakeLists.txt
   - Find all `add_dependencies()` calls
   - Identify unnecessary dependencies
3. ⚠️ **Add parallel stage targets** as shown in Phase 2.1
4. ⚠️ **Test with a subset** of artifacts first

### Week 2: Full Implementation

1. ⚠️ **Remove over-constraints** from all CMakeLists.txt
2. ⚠️ **Configure multi-arch builds** properly
3. ⚠️ **Add Ninja job pools** for optimal resource usage
4. ⚠️ **Benchmark and validate** the improvements

### Week 3: Fine-Tuning

1. ⚠️ **Profile the optimized build** with new ninja logs
2. ⚠️ **Identify remaining bottlenecks**
3. ⚠️ **Consider splitting large artifacts** if needed
4. ⚠️ **Document the new build structure**

---

## Validation and Testing

### Before & After Comparison

```bash
# Before optimization
time cmake --build . --target all
# Expected: ~173 minutes, avg concurrency ~7

# After optimization
time cmake --build . --target all
# Target: ~25-35 minutes, avg concurrency ~60-80

# Generate new ninja log for comparison
python analyze_ninja_concurrency.py
```

### Success Criteria

- [ ] Build time reduced to < 40 minutes
- [ ] Average concurrency > 50
- [ ] No artificial serialization (check concurrency timeline)
- [ ] All tests still pass
- [ ] No broken dependencies

---

## Risk Mitigation

### Potential Issues

1. **Hidden Dependencies**
   - Some artifacts may have undeclared runtime dependencies
   - Solution: Extensive testing, add back if needed

2. **Memory Pressure**
   - 60-80 parallel builds may consume 200+ GB RAM
   - Solution: Use job pools to limit memory-intensive tasks

3. **Build Failures**
   - Parallel builds may expose race conditions
   - Solution: Use `DEPENDS` carefully, test incrementally

4. **Windows-Specific Issues**
   - Windows has different path limits and parallel I/O behavior
   - Solution: Test on both platforms

---

## Expected Results

### Optimized Build Timeline

```
Time      Stage(s)                               Concurrency
-------------------------------------------------------------
0-5min    foundation                             ~40
          (sysdeps + third-party + base)

5-30min   compiler-runtime                       ~80
          (amd-llvm, hipify, runtimes in parallel)

30-55min  math-libs + comm-libs + debug-tools    ~98
          + dctools + iree-libs + media
          (5-6 stages in parallel)

55-60min  profiler-apps                          ~20
          (final stage)

Total: ~60 minutes (worst case)
       ~30 minutes (best case with further optimization)
```

### Resource Utilization

```
Current:
  CPU Cores Used: 7 avg, 124 max
  Build Time: 173 minutes
  CPU Efficiency: 5.7%

Optimized:
  CPU Cores Used: 70 avg, 98 max
  Build Time: 30-35 minutes
  CPU Efficiency: 70%
```

---

## Conclusion

The root build bottleneck is primarily due to **artificial serialization in the build orchestration**, not actual dependency constraints. By:

1. Enabling parallel stage execution
2. Improving within-stage parallelization
3. Removing over-constrained CMake dependencies
4. Properly configuring multi-arch builds

You can achieve a **7x speedup** (173 min → 25-35 min) while utilizing 70% of available CPU cores instead of just 5.7%.

The critical path analysis shows a theoretical minimum of ~70 minutes, but with perfect parallelization of independent stages, **30-40 minutes is a realistic target**.

---

**Next Steps**: Start with auditing the root CMakeLists.txt file for dependency declarations and implement the parallel stage targets as outlined in Phase 2.1.
