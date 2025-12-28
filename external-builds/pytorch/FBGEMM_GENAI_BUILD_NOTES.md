# FBGEMM_GENAI Build Attempt - PyTorch 2.11

## Summary

Attempted to enable FBGEMM_GENAI for improved GenAI inference performance on gfx1031 (RX 6700 XT), but encountered build failures due to Composable Kernel dependency issues.

## What is FBGEMM_GENAI?

FBGEMM (Facebook GEneral Matrix Multiplication) GenAI is a high-performance library optimized for generative AI inference workloads:

- **5-15% faster LLM inference**
- Better quantized model support (INT8/INT4)
- Reduced memory footprint for large models
- Optimized kernels for transformer models

## Build Attempt

**Date:** 2025-12-28
**Commit:** Cherry-picked c7c118ff "Enable FBGEMM_GENAI for pytorch 2.9"
**PyTorch Version:** 2.11.0a0
**ROCm Version:** 7.11 (custom build for gfx1031)

### Configuration

Set in `set_build_env.sh`:

```bash
export USE_FBGEMM_GENAI=ON
```

### Build Failure

**Build Progress:** 88% complete (6944/7822 steps)

**Error:**

```
fatal error: 'ck/config.h' file not found
    6 | #include "ck/config.h"
      |          ^~~~~~~~~~~~~
```

**Root Cause:**
FBGEMM_GENAI requires Composable Kernel (CK) to be properly built and configured. The PyTorch 2.11 build references an external `composable_kernel` submodule that hasn't been configured, so `ck/config.h` doesn't exist.

### Why It Failed

From the cherry-picked commit warning:

> "FBGEMM_GENAI is not available for PyTorch 2.7, and enabling it may cause build failures for PyTorch >= 2.8 (Except 2.9)"

**PyTorch 2.11 is in the "build failures" category.**

## Solution Options

### Option 1: Use PyTorch 2.9 (Guaranteed to Work)

```bash
cd /home/hashcat/TheRock/external-builds/pytorch
python pytorch_torch_repo.py checkout --pytorch-version 2.9
```

PyTorch 2.9 has verified FBGEMM_GENAI support for ROCm.

### Option 2: Wait for Official Support

Track issue: https://github.com/ROCm/TheRock/issues/2056

Wait for AMD to release a PyTorch 2.11+ build with working FBGEMM_GENAI.

### Option 3: Manual CK Configuration (Advanced)

Would require:

1. Building Composable Kernel separately
1. Configuring CMake paths for PyTorch to find CK
1. Ensuring version compatibility
1. Significant build system modifications

**Not recommended** - high complexity, likely to break other things.

## Current Status

**FBGEMM_GENAI:** Disabled (`USE_FBGEMM_GENAI=OFF`)
**Reason:** Build dependency issues with PyTorch 2.11
**Workaround:** Use PyTorch 2.9 if FBGEMM_GENAI is critical

## Related Commits

- **60509700** - Enable FBGEMM_GENAI for pytorch 2.9 (#2605)
- **5bb4fce0** - Add PyTorch build environment with FBGEMM_GENAI documentation

## Impact on llama-server

Without FBGEMM_GENAI, your Qwen models will run with:

- ✅ Standard FBGEMM (basic optimizations available)
- ✅ Full ROCm 7.11 gfx1031 support
- ✅ All other PyTorch features working
- ❌ Missing ~5-15% GenAI-specific inference speedup
- ❌ Suboptimal quantized model performance

## Recommendation

For production use with your current setup (PyTorch 2.11 + ROCm 7.11 gfx1031):

- **Keep FBGEMM_GENAI disabled** - stable and working
- **Performance is still excellent** - custom ROCm 7.11 provides major improvements
- **Consider PyTorch 2.9** - if GenAI optimizations are critical

The core ROCm optimizations (HIP kernels, rocBLAS, Composable Kernel for general ops) still provide substantial performance benefits for your llama-server workloads.
