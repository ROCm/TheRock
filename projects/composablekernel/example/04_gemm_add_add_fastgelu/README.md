# GEMM with Add, Add, and FastGELU Activation

## Theory

This example demonstrates a **GEMM operation fused with two addition operations and FastGELU activation**. This pattern is used in transformer feed-forward networks and other neural architectures where a linear transformation is followed by bias addition, residual addition, and a non-linear activation.

**Mathematical Formulation:**
$$
E = \text{FastGELU}((A \times B) + D_0 + D_1)
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $D_0$: [N] bias vector (broadcasted)
- $D_1$: [M, N] residual tensor
- $E$: [M, N] output

FastGELU is an efficient approximation of GELU:
$$
\text{FastGELU}(x) = x \cdot \sigma(1.702 \cdot x)
$$
where $\sigma$ is the sigmoid function.

**Algorithmic Background:**
- The GEMM result is kept in registers, bias and residual are added, and FastGELU is applied before writing to global memory.
- No intermediate results are written to global memory.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/04_gemm_add_add_fastgelu
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm_add_add_fastgelu_xdl -M 2048 -N 8192 -K 2048 --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/04_gemm_add_add_fastgelu/
├── gemm_add_add_fastgelu_xdl.cpp         # Main example: sets up, runs, and verifies GEMM+Add+Add+FastGELU
include/ck/tensor_operation/gpu/device/
│   └── device_gemm_multiple_d.hpp         # Device-level API for multi-tensor GEMM
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_xdl_cshuffle_v3.hpp    # XDL with C-Shuffle epilogue
│   └── device_gemm_fastgelu_impl.hpp      # FastGELU-specific implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_gemm_multiple_d_xdl.hpp   # Grid-level multi-stage GEMM
include/ck/tensor_operation/gpu/element/
    └── element_wise_operation.hpp         # Elementwise operation definitions
```

### Key Classes and Functions

- **DeviceGemmMultipleD** (in `device_gemm_multiple_d.hpp`):  
  Device API for GEMM with multiple auxiliary tensors and fused epilogues.
- **gridwise_gemm_multiple_d_xdl** (in `gridwise_gemm_multiple_d_xdl.hpp`):  
  Implements the tiled/blocking GEMM kernel with multi-stage epilogue.
- **element_wise_operation** (in `element_wise_operation.hpp`):  
  Defines FastGELU and other elementwise operations.

This example demonstrates how Composable Kernel supports complex multi-stage epilogue fusion for advanced neural network architectures.
