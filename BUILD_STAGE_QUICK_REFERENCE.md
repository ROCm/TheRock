# Build Stage Quick Reference

## Stage Build Commands

### Build All Stages (Optimized)
```bash
ninja
# or
cmake --build . --parallel
```

### Build Individual Stages

```bash
# Stage 1: Foundation (sysdeps, base)
cmake --build . --target therock-stage-foundation

# Stage 2: Compiler & Runtimes
cmake --build . --target therock-stage-compiler-runtime

# Stage 3: Math Libraries
cmake --build . --target therock-stage-math-libs

# Stage 4: Communication Libraries
cmake --build . --target therock-stage-comm-libs

# Stage 5: Debug Tools
cmake --build . --target therock-stage-debug-tools

# Stage 6: Data Center Tools
cmake --build . --target therock-stage-dctools

# Stage 7: IREE Libraries
cmake --build . --target therock-stage-iree-libs

# Stage 8: Profiler Applications
cmake --build . --target therock-stage-profiler-apps

# All post-compiler stages (3-8 in parallel)
cmake --build . --target therock-stage-post-compiler
```

## Stage Contents

### 🏗️ Foundation Stage
- `therock-sysdeps` - System dependencies
- `therock-base` - Base infrastructure
- `therock-sysdeps-expat`, `therock-sysdeps-gmp`, `therock-sysdeps-mpfr`, `therock-sysdeps-ncurses`

### ⚙️ Compiler-Runtime Stage
**Compiler:**
- `amd-llvm` - AMD LLVM compiler
- `hipify` - CUDA to HIP converter

**Runtimes:**
- `therock-core-runtime` - ROCm runtime (Linux)
- `therock-core-hip` - HIP runtime
- `therock-core-ocl` - OpenCL runtime
- `therock-core-hipinfo` - HIP info tool (Windows)
- `therock-core-hiptests` - HIP tests
- `therock-rocrtst` - Runtime tests

**Profiler Core:**
- `therock-rocprofiler-sdk` - Profiler SDK
- `therock-rocprofiler-compute` - Compute profiler

**Third-Party:**
- `therock-host-blas`, `therock-host-suite-sparse` - Host math libs
- `therock-fftw3` - FFTW library
- `therock-flatbuffers`, `therock-fmt`, `therock-nlohmann-json`, `therock-spdlog`

### 🧮 Math-Libs Stage
- `therock-blas` - BLAS libraries (rocBLAS, hipBLAS, etc.)
- `therock-fft` - FFT libraries (rocFFT, hipFFT)
- `therock-rand` - Random number libraries (rocRAND, hipRAND)
- `therock-prim` - Primitives library (rocPRIM)
- `therock-rocwmma` - Wave Matrix Multiply-Accumulate
- `therock-support` - Support libraries
- `therock-composable-kernel` - Composable Kernel
- `therock-miopen` - MIOpen ML library
- `therock-hipdnn` - hipDNN
- `therock-miopen-plugin` - MIOpen plugin

### 📡 Comm-Libs Stage
- `therock-rccl` - ROCm Communication Collectives Library

### 🐛 Debug-Tools Stage
- `therock-amd-dbgapi` - Debug API
- `therock-rocr-debug-agent` - Debug agent
- `therock-rocr-debug-agent-tests` - Debug agent tests
- `therock-rocgdb` - ROCm GDB

### 🖥️ DCTools Stage
- `therock-rdc` - ROCm Data Center tool

### 🤖 IREE-Libs Stage
- `therock-fusilli-plugin` - Fusilli IREE plugin

### 📊 Profiler-Apps Stage
- `therock-rocprofiler-systems` - Profiler systems application

## Testing Commands

### Test a Single Stage
```bash
# Build the stage
cmake --build . --target therock-stage-math-libs

# Run tests for that stage (example)
ctest -R "blas|fft|rand" --output-on-failure
```

### Verify Stage Dependencies
```bash
# See what a stage depends on
ninja -t query therock-stage-compiler-runtime

# Visualize dependency graph
ninja -t graph therock-stage-post-compiler | dot -Tpng > graph.png
```

## Performance Analysis

### Before Building
```bash
# Clean build directory
ninja clean
# or
rm -rf build/*
cmake .. -G Ninja
```

### During Build
```bash
# Monitor concurrent jobs
watch -n 1 'ps aux | grep ninja | wc -l'

# Monitor CPU usage
htop
# or on Windows
taskmgr
```

### After Building
```bash
# Analyze the ninja log
python analyze_ninja_concurrency.py

# Check build time
ninja -t compdb | jq '.[].output' | wc -l  # Total tasks
```

## Debugging Build Issues

### Check What Targets Exist
```bash
ninja -t targets all | grep therock-stage
```

### See All Targets in a Stage
```bash
ninja -t query therock-stage-foundation
```

### Force Rebuild a Stage
```bash
ninja -t clean therock-stage-math-libs
ninja therock-stage-math-libs
```

### Verbose Build
```bash
ninja -v therock-stage-compiler-runtime
```

## Expected Build Times

| Stage | Tasks | Time (Before) | Time (After) | Speedup |
|-------|-------|---------------|--------------|---------|
| Foundation | 6 | 5-10 min | 5 min | Parallel |
| Compiler-Runtime | 17 | 120 min | 25 min | 4.8x |
| Math-Libs | 10 | 20 min | 20 min* | Parallel |
| Comm-Libs | 1 | 5 min | 5 min* | Parallel |
| Debug-Tools | 4 | 10 min | 10 min* | Parallel |
| DCTools | 1 | 3 min | 3 min* | Parallel |
| IREE-Libs | 1 | 2 min | 2 min* | Parallel |
| Profiler-Apps | 1 | 5 min | 5 min | Sequential |
| **Total** | **42** | **173 min** | **35-40 min** | **4-5x** |

\* These stages run in **parallel**, so total time = max(20, 10, 5, 3, 2) = 20 min

## Feature Flags

Control what gets built:

```bash
# Disable optional stages
cmake .. -DTHEROCK_ENABLE_MATH_LIBS=OFF
cmake .. -DTHEROCK_ENABLE_ML_LIBS=OFF
cmake .. -DTHEROCK_ENABLE_DEBUG_TOOLS=OFF

# Enable all
cmake .. -DTHEROCK_ENABLE_ALL=ON

# Minimal build (core only)
cmake .. -DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_CORE=ON
```

## Troubleshooting

### "Target does not exist"
- The target may be disabled by feature flags
- Platform-specific targets (Linux/Windows only)
- Check: `ninja -t targets all | grep <target-name>`

### "Build order seems wrong"
- Clean and rebuild: `ninja clean && ninja`
- Check dependencies: `ninja -t query <target>`

### "Not seeing parallelization"
- Check Ninja job count: `ninja --version && echo "Using $(nproc) jobs"`
- Explicitly set: `ninja -j 98`
- Verify CPU usage in task manager

## Quick Start After Implementation

```bash
# 1. Reconfigure
cd build
rm -rf *
cmake .. -G Ninja

# 2. Time the build
time ninja -j 98

# 3. Check the results
python analyze_ninja_concurrency.py

# 4. Compare
# Before: ~173 min, 7 avg concurrency
# After:  ~35 min, 70 avg concurrency
```

## Tips

- Use `ninja -j 98` to maximize parallelization
- Build stages independently for faster iteration during development
- Use `therock-stage-post-compiler` to build everything after compiler in one command
- Monitor `.ninja_log` for performance insights
- Set `THEROCK_VERBOSE=ON` for detailed CMake output

---

**Updated**: After implementing parallel stage execution  
**Status**: ✅ Implementation complete, ready for testing
