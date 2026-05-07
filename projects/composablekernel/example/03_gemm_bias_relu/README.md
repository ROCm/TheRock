# GEMM with Bias and ReLU Activation Fusion

## Theory

This example demonstrates **GEMM fused with bias addition and ReLU activation**. This is the core pattern for fully connected (dense) neural network layers and the feed-forward blocks in transformers.

**Mathematical Formulation:**
$$
E = \text{ReLU}(A \times B + \text{bias})
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $\text{bias}$: [N] bias vector (broadcasted)
- $E$: [M, N] output

**Algorithmic Background:**
- The GEMM result is kept in registers, bias is added, and ReLU is applied before writing to global memory.
- This fusion eliminates intermediate memory traffic and is a standard optimization in deep learning frameworks.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/03_gemm_bias_relu
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm_bias_relu_xdl -M 2048 -N 8192 -K 2048 --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/03_gemm_bias_relu/
├── gemm_bias_relu_xdl.cpp         # Main example: sets up, runs, and verifies GEMM+Bias+ReLU
include/ck/tensor_operation/gpu/device/
│   └── device_gemm_multiple_d.hpp         # Device-level API for multi-tensor GEMM
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_xdl_cshuffle_v3.hpp    # XDL with C-Shuffle epilogue
│   └── device_gemm_bias_relu_impl.hpp     # Specialized bias+ReLU implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_gemm_xdl_cshuffle.hpp     # Grid-level GEMM with epilogue
include/ck/tensor_operation/gpu/element/
    └── element_wise_operation.hpp         # Elementwise operation definitions
```

### Key Classes and Functions

- **DeviceGemmMultipleD** (in `device_gemm_multiple_d.hpp`):  
  Device API for GEMM with auxiliary tensors and fused epilogues.
- **gridwise_gemm_xdl_cshuffle** (in `gridwise_gemm_xdl_cshuffle.hpp`):  
  Implements the tiled/blocking GEMM kernel with fused epilogue.
- **element_wise_operation** (in `element_wise_operation.hpp`):  
  Defines bias addition and ReLU activation.

This example demonstrates the standard epilogue fusion concept that enables efficient neural network layers in modern deep learning.
