# GEMM with Multiple D and Multiple Reductions

## Theory

This example demonstrates **GEMM with multiple auxiliary tensors (D) and multiple reduction operations**. This pattern is used in advanced neural network layers that require additional outputs or statistics (such as sums, means, or other reductions) alongside the main GEMM result.

**Mathematical Formulation:**
- For each GEMM: $C = A \times B$
- Auxiliary tensors: $D_0, D_1, ...$ (various shapes)
- Reductions: e.g., sum, mean, max over specified axes or outputs

The kernel computes the main GEMM output and additional reductions or statistics in a single pass.

**Algorithmic Background:**
- The GEMM result is kept in registers, auxiliary tensors are fused in the epilogue, and reductions are computed as part of the output.
- Useful for multi-task learning, attention statistics, and custom neural network layers.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/16_gemm_multi_d_multi_reduces
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm_multi_d_multi_reduces_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/16_gemm_multi_d_multi_reduces/
├── gemm_multi_d_multi_reduces_xdl.cpp         # Main example: sets up, runs, and verifies GEMM with multi-D/multi-reduce
include/ck/tensor_operation/gpu/device/
│   └── device_gemm_multi_d_multi_reduces.hpp       # Device-level API for multi-D/multi-reduce GEMM
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_multi_d_multi_reduces_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
    └── gridwise_gemm_multi_d_multi_reduces.hpp     # Grid-level kernel
```

### Key Classes and Functions

- **DeviceGemmMultiDMultiReduces** (in `device_gemm_multi_d_multi_reduces.hpp`):  
  Device API for GEMM with multiple outputs and reductions.
- **gridwise_gemm_multi_d_multi_reduces** (in `gridwise_gemm_multi_d_multi_reduces.hpp`):  
  Implements the tiled/blocking GEMM kernel with multi-output/reduce epilogue.

This example demonstrates how Composable Kernel supports advanced GEMM patterns with multiple outputs and reductions in a single efficient kernel.
