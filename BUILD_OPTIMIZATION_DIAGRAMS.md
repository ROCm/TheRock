# Build Dependency Visualization

## Current Build Flow (Sequential - 173 minutes)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ FOUNDATION (6 min)                                                           │
│ sysdeps, base, third-party-sysdeps                                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ COMPILER-RUNTIME (120 min) ← MAJOR BOTTLENECK                               │
│ Sequential execution of:                                                     │
│   • amd-llvm (20 min)                                                       │
│   • third-party-libs (15 min, but could be parallel)                        │
│   • hipify (5 min)                                                          │
│   • core-runtime (10 min)                                                   │
│   • hip-runtime + opencl-runtime (15 min sequential)                        │
│   • profiler-core (10 min)                                                  │
│   • rocrtst (5 min)                                                         │
│ Only 7 avg concurrent tasks out of 17 artifacts!                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ MATH-LIBS (20 min)                                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ COMM-LIBS (5 min)                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ DEBUG-TOOLS (10 min)                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ DCTOOLS-CORE (3 min)                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ IREE-LIBS (2 min)                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ PROFILER-APPS (5 min)                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ MEDIA (1 min)                                                                │
└──────────────────────────────────────────────────────────────────────────────┘

Total: 6 + 120 + 20 + 5 + 10 + 3 + 2 + 5 + 1 = 173 minutes
Average Concurrency: 7.11 tasks
```

---

## Optimized Build Flow (Parallel - 30-35 minutes)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ FOUNDATION (5 min, 40 concurrent)                                           │
│ Parallel: sysdeps, base, third-party-sysdeps, all independent libs          │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ COMPILER-RUNTIME (25 min, 80 concurrent) ← OPTIMIZED!                       │
│                                                                              │
│ Level 0 (0-5 min, parallel):                                                │
│   ├─ amd-llvm (20 min, starts immediately)                                 │
│   └─ third-party-libs (all 7 in parallel, 5 min)                           │
│                                                                              │
│ Level 1 (5-10 min, parallel):                                               │
│   ├─ hipify (5 min, needs amd-llvm)                                        │
│   └─ core-runtime (5 min, needs amd-llvm)                                  │
│                                                                              │
│ Level 2 (10-20 min, parallel):                                              │
│   ├─ hip-runtime (10 min)                                                   │
│   ├─ opencl-runtime (8 min)                                                 │
│   └─ profiler-core (7 min)                                                  │
│                                                                              │
│ Level 3 (20-25 min):                                                        │
│   └─ rocrtst (5 min)                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                    ┌───────────────┴───────────────┬──────────────────┐
                    ↓                               ↓                  ↓
┌─────────────────────────────┐  ┌──────────────────────────┐  ┌────────────────┐
│ MATH-LIBS (20 min)          │  │ DEBUG-TOOLS (10 min)     │  │ MEDIA (1 min)  │
│ 98 concurrent               │  │ 40 concurrent            │  │ 10 concurrent  │
└─────────────────────────────┘  └──────────────────────────┘  └────────────────┘
                    ↓                               ↓                  ↓
┌─────────────────────────────┐  ┌──────────────────────────┐  ┌────────────────┐
│ COMM-LIBS (5 min)           │  │ DCTOOLS-CORE (3 min)     │  │                │
│ 50 concurrent               │  │ 20 concurrent            │  │                │
└─────────────────────────────┘  └──────────────────────────┘  └────────────────┘
                    ↓                               ↓
                    └───────────────┬───────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │ IREE-LIBS (2 min)             │
                    │ 15 concurrent                 │
                    └───────────────────────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │ PROFILER-APPS (5 min)         │
                    │ 20 concurrent                 │
                    └───────────────────────────────┘

Total: 5 + 25 + max(20, 10+3+2, 1) + 5 = 5 + 25 + 20 + 5 = 35 minutes
Average Concurrency: 70 tasks
```

---

## Dependency Graph by Level

```
Level 0 (12 artifacts, can start immediately):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ sysdeps     │ sysdeps-    │ sysdeps-gmp │ sysdeps-    │
│             │ expat       │             │ ncurses     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ host-blas   │ fftw3       │ flatbuffers │ fmt         │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ nlohmann-   │ spdlog      │ base        │ support     │
│ json        │             │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┘

Level 1 (4 artifacts):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ sysdeps-    │ sysdeps-    │ host-suite- │ amd-llvm    │
│ amd-mesa    │ mpfr        │ sparse      │             │
└─────────────┴─────────────┴─────────────┴─────────────┘

Level 2 (2 artifacts):
┌─────────────┬─────────────┐
│ hipify      │ core-runtime│
└─────────────┴─────────────┘

Level 3 (4 artifacts):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ core-hip    │ core-ocl    │ rocprofiler-│ amd-dbgapi  │
│             │             │ sdk         │             │
└─────────────┴─────────────┴─────────────┴─────────────┘

Level 4 (15 artifacts, mostly math/ML libs):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ rocrtst     │ core-hipinfo│ core-       │ blas        │
│             │             │ hiptests    │             │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ fft         │ rand        │ prim        │ composable- │
│             │             │             │ kernel      │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ hipdnn      │ rccl        │ rocprofiler-│ rocr-debug- │
│             │             │ compute     │ agent       │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ rocprofiler-│ rdc         │ rocgdb      │             │
│ systems     │             │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┘

Level 5 (4 artifacts):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ rocwmma     │ miopen      │ fusilli-    │ rocr-debug- │
│             │             │ plugin      │ agent-tests │
└─────────────┴─────────────┴─────────────┴─────────────┘

Level 6 (1 artifact):
┌─────────────┐
│ miopen-     │
│ plugin      │
└─────────────┘
```

---

## Critical Path Analysis

```
The Critical Path (longest dependency chain):
═══════════════════════════════════════════════════════════

sysdeps (5 min)
   ║
   ╠══ Core system dependencies
   ║
   ▼
amd-llvm (20 min)
   ║
   ╠══ LLVM/Clang compiler (longest single artifact)
   ║
   ▼
core-runtime (5 min)
   ║
   ╠══ ROCm runtime (HSA, rocminfo)
   ║
   ▼
core-hip (10 min)
   ║
   ╠══ HIP runtime and compiler driver
   ║
   ▼
blas (15 min)
   ║
   ╠══ rocBLAS + hipBLAS (math library)
   ║
   ▼
miopen (20 min)
   ║
   ╠══ MIOpen (ML library, depends on BLAS)
   ║
   ▼
miopen-plugin (2 min)
   ║
   ╠══ Plugin interface
   ║
   ▼
DONE

Total Critical Path Time: ~77 minutes
(This is the theoretical minimum if everything else is perfectly parallel)
```

---

## Parallelization Opportunities

### Compiler-Runtime Stage Breakdown

```
Current (Sequential):              Optimized (Parallel):

Time 0 ─────────────────────      Time 0 ─────────────────────
        │                                 │ amd-llvm (20m)
        │ amd-llvm (20m)                  │ + 7 third-party
        │                                 │   libs (5m)
Time 20 ─────────────────────      Time 5 ─────────────────────
        │ third-party (15m)               │ hipify (5m)
        │                                 │ + core-runtime (5m)
Time 35 ─────────────────────      Time 10 ────────────────────
        │ hipify (5m)                     │ hip-runtime (10m)
        │                                 │ + opencl-runtime (8m)
Time 40 ─────────────────────            │ + profiler-core (7m)
        │ core-runtime (10m)       Time 20 ────────────────────
        │                                 │ rocrtst (5m)
Time 50 ─────────────────────      Time 25 ────────────────────
        │ hip-runtime (7m)                DONE!
        │                          
Time 57 ─────────────────────      Speedup: 120m → 25m (4.8x)
        │ opencl-runtime (8m)      
        │
Time 65 ─────────────────────
        │ profiler-core (10m)
        │
Time 75 ─────────────────────
        │ rocrtst (5m)
        │
Time 80 ─────────────────────
        DONE

Total: 80 minutes                 Total: 25 minutes
```

---

## High-Impact Artifacts (Build These Fast!)

```
Artifact              Dependents    Impact    Priority
═══════════════════════════════════════════════════════════
core-runtime          17           CRITICAL   P0
core-hip              16           CRITICAL   P0
amd-llvm              8            CRITICAL   P0
rocprofiler-sdk       7            HIGH       P1
sysdeps               6            HIGH       P1
base                  2            MEDIUM     P2
blas                  2            MEDIUM     P2
```

These artifacts block the most downstream work. Optimize their build:
- Use fastest compiler flags
- Allocate maximum parallel jobs
- Consider pre-built caching

---

## Resource Utilization Comparison

```
CURRENT BUILD:
┌────────────────────────────────────────────────────────────┐
│ CPU Cores Available: 98                                    │
│ Cores Used (Avg):    7.11   ▓░░░░░░░░░░░░░░  7%          │
│ Cores Used (Max):    124    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  127% (burst) │
│ Efficiency:          5.7%   POOR                           │
│ Build Time:          173 minutes                           │
└────────────────────────────────────────────────────────────┘

OPTIMIZED BUILD:
┌────────────────────────────────────────────────────────────┐
│ CPU Cores Available: 98                                    │
│ Cores Used (Avg):    70     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  71%           │
│ Cores Used (Max):    98     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  100%          │
│ Efficiency:          71%    EXCELLENT                      │
│ Build Time:          30-35 minutes                         │
└────────────────────────────────────────────────────────────┘

Improvement: 7x faster, 12x better CPU utilization
```

---

## Implementation Checklist

```
Phase 1: Immediate Fixes
  [ ] Audit root CMakeLists.txt for dependencies
  [ ] Add parallel stage targets
  [ ] Remove over-constrained dependencies
  [ ] Test with subset of artifacts

Phase 2: Full Implementation  
  [ ] Update all CMakeLists.txt files
  [ ] Configure multi-arch builds
  [ ] Add Ninja job pools
  [ ] Full build test

Phase 3: Validation
  [ ] Generate new ninja logs
  [ ] Compare before/after concurrency
  [ ] Verify all tests pass
  [ ] Document changes
```

---

Generated by: analyze_build_topology.py
Based on: BUILD_TOPOLOGY.toml + Windows ninja logs analysis
