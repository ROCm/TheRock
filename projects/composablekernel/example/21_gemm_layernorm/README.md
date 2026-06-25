# GEMM with LayerNorm Fusion

## Theory

This example demonstrates **GEMM fused with layer normalization**. This pattern is used in transformer feed-forward networks and other architectures where a linear transformation is followed by normalization for improved training stability.

**Mathematical Formulation:**
- GEMM: $Y = A \times B$
- LayerNorm: $\text{LayerNorm}(Y) = \gamma \cdot \frac{Y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$
  - $\mu$: mean of $Y$ over the normalization axis
  - $\sigma^2$: variance of $Y$ over the normalization axis
  - $\gamma$, $\beta$: learnable scale and shift parameters

**Algorithmic Background:**
- The GEMM result is kept in registers, and layer normalization is applied before writing to global memory.
- LayerNorm is typically applied over the last dimension (features).
- This fusion reduces memory traffic and is common in transformer MLP blocks.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/21_gemm_layernorm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm_layernorm_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/21_gemm_layernorm/
├── gemm_layernorm_xdl.cpp         # Main example: sets up, runs, and verifies GEMM+LayerNorm
include/ck/tensor_operation/gpu/device/
│   └── device_gemm_layernorm.hpp       # Device-level GEMM+LayerNorm API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_layernorm_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
    └── gridwise_gemm_layernorm.hpp     # Grid-level kernel
```

### Key Classes and Functions

- **DeviceGemmLayerNorm** (in `device_gemm_layernorm.hpp`):  
  Device API for GEMM fused with layer normalization.
- **gridwise_gemm_layernorm** (in `gridwise_gemm_layernorm.hpp`):  
  Implements the tiled/blocking GEMM kernel with layer normalization epilogue.

This example demonstrates how Composable Kernel supports efficient fusion of linear and normalization layers for transformer and deep learning models.
