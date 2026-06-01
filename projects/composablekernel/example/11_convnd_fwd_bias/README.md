# N-Dimensional Convolution Forward with Bias

## Theory

This example demonstrates **N-dimensional convolution forward** with bias addition. This is a common pattern in convolutional neural networks (CNNs), where a bias term is added to each output channel after the convolution operation.

**Mathematical Formulation:**
$$
Y[n, c_{out}, o_1, ..., o_n] = \sum_{c_{in}} \sum_{k_1} ... \sum_{k_n} X[n, c_{in}, o_1 + k_1, ..., o_n + k_n] \cdot W[c_{out}, c_{in}, k_1, ..., k_n] + B[c_{out}]
$$
- $X$: [N, C_in, D1, D2, ..., Dn] input tensor
- $W$: [C_out, C_in, K1, K2, ..., Kn] weight tensor
- $B$: [C_out] bias tensor
- $Y$: [N, C_out, O1, O2, ..., On] output tensor

**Algorithmic Background:**
- Composable Kernel implements convolution as an implicit GEMM, with bias addition fused in the epilogue for efficiency.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/11_convnd_fwd_bias
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./convnd_fwd_bias_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/11_convnd_fwd_bias/
├── convnd_fwd_bias_xdl.cpp         # Main example: sets up, runs, and verifies N-D convolution with bias
include/ck/tensor_operation/gpu/device/
│   └── device_convnd_fwd_bias.hpp       # Device-level convolution API with bias
include/ck/tensor_operation/gpu/device/impl/
│   └── device_convnd_fwd_bias_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
    └── gridwise_convnd_fwd_bias.hpp     # Grid-level kernel
```

### Key Classes and Functions

- **DeviceConvNdFwdBias** (in `device_convnd_fwd_bias.hpp`):  
  Device API for N-dimensional convolution with bias.
- **gridwise_convnd_fwd_bias** (in `gridwise_convnd_fwd_bias.hpp`):  
  Implements the tiled/blocking convolution kernel with bias epilogue.

This example demonstrates how Composable Kernel fuses bias addition into the convolution forward pass for efficient CNN layer implementation.
