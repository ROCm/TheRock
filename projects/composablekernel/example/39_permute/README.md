# Tensor Permutation (Dimension Reordering)

## Theory

This example demonstrates **tensor permutation operations**, which reorder the dimensions of tensors according to a specified permutation pattern. Permutation is fundamental for many machine learning operations, including tensor layout transformations, data format conversions, and implementing complex tensor operations.

**Mathematical Formulation:**
Given an input tensor $X$ with shape $[D_0, D_1, ..., D_{n-1}]$ and a permutation pattern $P = [p_0, p_1, ..., p_{n-1}]$, the permutation operation produces an output tensor $Y$ with shape $[D_{p_0}, D_{p_1}, ..., D_{p_{n-1}}]$ such that:
$$
Y_{i_{p_0}, i_{p_1}, ..., i_{p_{n-1}}} = X_{i_0, i_1, ..., i_{n-1}}
$$

**Algorithmic Background:**
- Permutation is used for matrix transpose, NCHW/NHWC layout conversion, attention head reshaping, and more.
- Efficient permutation requires optimizing memory access patterns for coalescing and bandwidth.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/39_permute
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (matrix transpose)
./permute_xdl --input_shape=4096,4096 --permutation=1,0 --verify=1 --time=1

# Example run (NCHW to NHWC)
./permute_xdl --input_shape=32,256,56,56 --permutation=0,2,3,1 --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/39_permute/
├── permute_xdl.cpp         # Main example: sets up, runs, and verifies tensor permutation
include/ck/tensor_operation/gpu/device/
│   └── device_permute.hpp       # Device-level permutation API
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_permute.hpp     # Grid-level permutation kernel
```

### Key Classes and Functions

- **DevicePermute** (in `device_permute.hpp`):  
  Device API for tensor permutation.
- **gridwise_permute** (in `gridwise_permute.hpp`):  
  Implements the tiled/blocking permutation kernel.

This example demonstrates how Composable Kernel implements efficient tensor dimension reordering for layout transformations and deep learning operations.
