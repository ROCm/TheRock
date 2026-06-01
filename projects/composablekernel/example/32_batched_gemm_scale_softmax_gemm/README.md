# Batched GEMM-Scale-Softmax-GEMM: Fused Attention

## Theory

This example demonstrates the **fused attention mechanism** used in transformer models, implementing the sequence: batched Q×K^T → scaling → softmax → ×V in a single kernel. This pattern is critical for efficient transformer inference and training.

**Mathematical Formulation:**
- Attention: $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- $Q$: [B, H, N, d_k] queries
- $K$: [B, H, N, d_k] keys
- $V$: [B, H, N, d_v] values
- $O$: [B, H, N, d_v] output

**Algorithmic Background:**
- Computes Q×K^T, scales by $1/\sqrt{d_k}$, applies softmax, then multiplies by V.
- Uses numerically stable softmax and memory-efficient tiling.
- Used in multi-head attention and transformer blocks.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/32_batched_gemm_scale_softmax_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./batched_gemm_scale_softmax_gemm_xdl --batch=32 --heads=12 --seq_len=512 --head_dim=64 --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/32_batched_gemm_scale_softmax_gemm/
├── batched_gemm_scale_softmax_gemm_xdl.cpp         # Main example: sets up, runs, and verifies fused attention
include/ck/tensor_operation/gpu/device/
│   └── device_batched_gemm_scale_softmax_gemm.hpp       # Device-level fused attention API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_batched_attention_impl.hpp                # Attention-specific implementation
│   └── device_online_softmax_impl.hpp                   # Online softmax implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_batched_gemm_softmax.hpp                # Grid-level fused attention kernel
│   └── gridwise_online_softmax.hpp                      # Grid-level online softmax
```

### Key Classes and Functions

- **DeviceBatchedGemmScaleSoftmaxGemm** (in `device_batched_gemm_scale_softmax_gemm.hpp`):  
  Device API for fused attention.
- **gridwise_batched_gemm_softmax** (in `gridwise_batched_gemm_softmax.hpp`):  
  Implements the tiled/blocking fused attention kernel.
- **gridwise_online_softmax** (in `gridwise_online_softmax.hpp`):  
  Implements numerically stable, memory-efficient softmax.

This example demonstrates how Composable Kernel implements efficient, fused attention for transformer and large language models.
