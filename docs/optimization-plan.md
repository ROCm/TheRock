# ROCm TheRock Optimization Plan

## Current Baseline Build

- **Location**: `/home/hashcat/TheRock`
- **Target**: gfx103X-all family (gfx1030, gfx1032, gfx1035, gfx1036)
- **Build Type**: Release with `-O3`
- **Purpose**: Baseline/stable build for testing

## Future Optimized Build Plan

### Proposed Location

```bash
/home/hashcat/TheRock-gfx103X-optimized
```

### Phase 1: CPU-Specific Optimizations (Safest)

**Build command:**

```bash
cd /home/hashcat/TheRock-gfx103X-optimized
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEROCK_AMDGPU_FAMILIES=gfx103X-all \
  -DTHEROCK_AMDGPU_TARGETS= \
  -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=gfx103X-all \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
  -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native"
```

**Expected improvement:** 5-15% faster on your specific CPU
**Risk:** Low (only loses portability)

### Phase 2: Link-Time Optimization (LTO)

**Build command:**

```bash
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEROCK_AMDGPU_FAMILIES=gfx103X-all \
  -DTHEROCK_AMDGPU_TARGETS= \
  -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=gfx103X-all \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto=thin" \
  -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -flto=thin"
```

**Expected improvement:** Additional 10-20% from cross-module optimization
**Risk:** Medium (longer build time, more RAM needed)
**Note:** ThinLTO is faster than full LTO with similar results

### Phase 3: Aggressive Optimizations (Test Carefully!)

**Build command:**

```bash
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEROCK_AMDGPU_FAMILIES=gfx103X-all \
  -DTHEROCK_AMDGPU_TARGETS= \
  -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=gfx103X-all \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto=thin -ffast-math -funroll-loops" \
  -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -flto=thin -ffast-math -funroll-loops"
```

**Flags explained:**

- `-ffast-math`: Relaxes IEEE floating-point rules for speed
  - ⚠️ **WARNING**: May affect numerical accuracy in ML models!
- `-funroll-loops`: Unrolls loops for better instruction pipelining

**Expected improvement:** Additional 5-10%
**Risk:** High (may break numerical stability, test thoroughly!)

### Phase 4: Profile-Guided Optimization (PGO) - Advanced

**Step 1: Build with instrumentation**

```bash
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEROCK_AMDGPU_FAMILIES=gfx103X-all \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -fprofile-generate" \
  -DCMAKE_C_FLAGS="-O3 -march=native -fprofile-generate"
cmake --build build
```

**Step 2: Run representative workloads**

```bash
# Install instrumented build to /opt/rocm-profile
# Run Ollama workloads
# Run LMStudio workloads
# Run llama-server benchmarks
# Profile data collected in *.profraw files
```

**Step 3: Rebuild with profile data**

```bash
llvm-profdata merge -output=default.profdata *.profraw

cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEROCK_AMDGPU_FAMILIES=gfx103X-all \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -fprofile-use=default.profdata" \
  -DCMAKE_C_FLAGS="-O3 -march=native -fprofile-use=default.profdata"
cmake --build build
```

**Expected improvement:** 10-20% on profiled workloads
**Risk:** High complexity, time-consuming
**Best for:** Production deployments after thorough testing

## GPU-Specific Optimizations

### AMD GPU Compiler Flags

You can also tune the GPU kernel compilation:

```bash
-DAMDGPU_TARGETS="gfx1030;gfx1032;gfx1035;gfx1036" \
-DCMAKE_HIP_FLAGS="-O3 -ffast-math"
```

## Testing Checklist

After each optimization phase, test:

- [ ] Ollama model loading and inference
- [ ] LMStudio model loading and inference
- [ ] llama-server benchmarks
- [ ] rocm-smi GPU detection
- [ ] hipBLAS/rocBLAS accuracy tests
- [ ] MIOpen convolution tests
- [ ] Memory leak checks (long-running inference)
- [ ] Multi-GPU tests (if applicable)

## Recommended Workflow

1. **Complete baseline build** in `/home/hashcat/TheRock`
1. **Test thoroughly** for 1-2 weeks with real workloads
1. **Clone to optimized directory**:
   ```bash
   cp -a /home/hashcat/TheRock /home/hashcat/TheRock-gfx103X-optimized
   cd /home/hashcat/TheRock-gfx103X-optimized
   rm -rf build
   ```
1. **Start with Phase 1** (march=native)
1. **Test and benchmark** each phase
1. **Keep baseline build** as fallback!

## Benchmark Commands

### Ollama

```bash
time curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

### LMStudio

```bash
time curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "local-model",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7
}'
```

### rocBLAS Direct

```bash
cd /opt/rocm/bin
./rocblas-bench -f gemm -r f32 -m 1024 -n 1024 -k 1024
```

## Notes

- **Backup baseline build** before experimenting!
- **Don't use aggressive optimizations in production** until thoroughly validated
- **Keep both builds** - baseline for stability, optimized for performance
- **Document any issues** encountered with specific flags
- **GPU kernels** (rocBLAS, MIOpen) may benefit less from CPU flags
- **LLVM compilation** is the most time-consuming part - LTO will extend this significantly

## Estimated Build Times

- Baseline (-O3): ~2-4 hours (your current build)
- - march=native: ~2-4 hours (similar)
- - ThinLTO: ~4-6 hours (longer linking)
- - PGO: ~6-8 hours (two full builds + profiling)

## Resource Requirements

- **RAM**: 32GB minimum (64GB recommended for LTO)
- **Swap**: 32GB+ (you already have this configured)
- **Disk**: ~50GB per build directory
- **Parallel jobs**: Consider `-j$(nproc)` or `-j8` based on RAM

______________________________________________________________________

**Created**: 2025-11-28
**For**: gfx103X family optimization after baseline validation
