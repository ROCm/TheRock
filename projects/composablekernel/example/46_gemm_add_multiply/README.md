# GEMM with Add and Multiply Fusion

## Theory

This example demonstrates **GEMM fused with addition and multiplication operations**. This pattern is used in neural networks for bias addition, scaling, gating, and other elementwise transformations after a linear layer.

**Mathematical Formulation:**
- GEMM: $Y = A \times B$
- Add: $Z = Y + D_0$
- Multiply: $E = Z \odot D_1$
  - $D_0$, $D_1$: auxiliary tensors (e.g., bias, scale, gate)

**Algorithmic Background:**
- The GEMM result is kept in registers, addition and multiplication are fused in the epilogue.
- No intermediate results are written to global memory.
- Used for bias+scale, gating, and other fused epilogue patterns.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/46_gemm_add_multiply
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

```

### Run ```example_gemm_add_multiply_dl_fp16```

```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
#arg4 to 11: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD0, StrideD1, StrideE"
./bin/example_gemm_add_multiply_dl_fp16 1 1 1
```

## Source Code Structure

### Directory Layout
```
example/46_gemm_add_multiply/
├── gemm_add_multiply_xdl.cpp         # Main example: sets up, runs, and verifies GEMM+Add+Multiply
include/ck/tensor_operation/gpu/device/
│   └── device_gemm_multiple_d.hpp       # Device-level API for multi-tensor GEMM
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_add_multiply_impl.hpp # Add+Multiply implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_gemm_multiple_d_xdl.hpp # Grid-level multi-stage GEMM
include/ck/tensor_operation/gpu/element/
    └── element_wise_operation.hpp       # Elementwise operation definitions
```

### Key Classes and Functions

- **DeviceGemmMultipleD** (in `device_gemm_multiple_d.hpp`):  
  Device API for GEMM with multiple auxiliary tensors and fused epilogues.
- **gridwise_gemm_multiple_d_xdl** (in `gridwise_gemm_multiple_d_xdl.hpp`):  
  Implements the tiled/blocking GEMM kernel with multi-stage epilogue.
- **element_wise_operation** (in `element_wise_operation.hpp`):  
  Defines addition, multiplication, and other elementwise operations.

This example demonstrates how Composable Kernel supports efficient fusion of addition and multiplication with GEMM for deep learning and scientific computing.
