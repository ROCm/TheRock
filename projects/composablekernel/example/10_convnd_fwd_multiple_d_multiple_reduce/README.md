# N-Dimensional Convolution with Multiple D and Multiple Reduce

## Theory

This example demonstrates **N-dimensional convolution forward** with support for multiple auxiliary tensors (D) and multiple reduction operations. This is useful for advanced neural network layers that require additional outputs or statistics alongside the main convolution result.

**Mathematical Formulation:**
- Input tensor: $X[N, C_{in}, D_1, D_2, ..., D_n]$
- Weight tensor: $W[C_{out}, C_{in}, K_1, K_2, ..., K_n]$
- Auxiliary tensors: $D_0, D_1, ...$ (various shapes)
- Output tensor: $Y[N, C_{out}, O_1, O_2, ..., O_n]$
- Reduction operations: e.g., sum, mean, max over specified axes

The convolution computes the standard output as well as additional outputs or statistics by applying reduction operations to the convolution result or auxiliary tensors.

**Algorithmic Background:**
- Composable Kernel implements this as an implicit GEMM with support for multiple auxiliary tensors and reductions in the epilogue.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/10_convnd_fwd_multiple_d_multiple_reduce
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./convnd_fwd_multiple_d_multiple_reduce_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/10_convnd_fwd_multiple_d_multiple_reduce/
├── convnd_fwd_multiple_d_multiple_reduce_xdl.cpp   # Main example: sets up, runs, and verifies N-D convolution with multiple D/reduce
include/ck/tensor_operation/gpu/device/
│   └── device_convnd_fwd_multiple_d_multiple_reduce.hpp   # Device-level API for multi-D/multi-reduce convolution
include/ck/tensor_operation/gpu/device/impl/
│   └── device_convnd_fwd_multiple_d_multiple_reduce_impl.hpp # Implementation
include/ck/tensor_operation/gpu/grid/
    └── gridwise_convnd_fwd_multiple_d_multiple_reduce.hpp # Grid-level kernel
```

### Key Classes and Functions

- **DeviceConvNdFwdMultipleDMultipleReduce** (in `device_convnd_fwd_multiple_d_multiple_reduce.hpp`):  
  Device API for N-dimensional convolution with multiple outputs and reductions.
- **gridwise_convnd_fwd_multiple_d_multiple_reduce** (in `gridwise_convnd_fwd_multiple_d_multiple_reduce.hpp`):  
  Implements the tiled/blocking convolution kernel with multi-output/reduce epilogue.

This example demonstrates how Composable Kernel supports advanced convolution patterns with multiple outputs and reductions in a single efficient kernel.
