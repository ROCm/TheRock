# Elementwise Operation with Permutation Fusion

## Theory

This example demonstrates **elementwise operations fused with tensor permutation**. This pattern is used in deep learning for applying activation functions or scaling while simultaneously reordering tensor dimensions (e.g., NCHW to NHWC).

**Mathematical Formulation:**
- Elementwise: $Z = f(X)$ or $Z = f(X, Y)$
- Permute: $Y_{i_{p_0}, i_{p_1}, ..., i_{p_{n-1}}} = Z_{i_0, i_1, ..., i_{n-1}}$
  - $P = [p_0, p_1, ..., p_{n-1}]$ is the permutation pattern

**Algorithmic Background:**
- The elementwise operation and permutation are fused in a single kernel.
- Intermediate results are kept in registers, not written to global memory.
- Used for layout conversion with activation, attention head reshaping, and more.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/44_elementwise_permute
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (ReLU + NCHW to NHWC)
./elementwise_permute_xdl --input_shape=32,128,56,56 --permutation=0,2,3,1 --operation=relu --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/44_elementwise_permute/
├── elementwise_permute_xdl.cpp         # Main example: sets up, runs, and verifies elementwise+permute
include/ck/tensor_operation/gpu/device/
│   └── device_elementwise_permute.hpp       # Device-level API for fused elementwise+permute
include/ck/tensor_operation/gpu/device/impl/
│   └── device_elementwise_permute_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_elementwise_permute.hpp     # Grid-level kernel
include/ck/tensor_operation/gpu/element/
    └── element_wise_operation.hpp           # Elementwise operation definitions
```

### Key Classes and Functions

- **DeviceElementwisePermute** (in `device_elementwise_permute.hpp`):  
  Device API for fused elementwise and permutation.
- **gridwise_elementwise_permute** (in `gridwise_elementwise_permute.hpp`):  
  Implements the tiled/blocking elementwise+permute kernel.
- **element_wise_operation** (in `element_wise_operation.hpp`):  
  Defines elementwise operations (e.g., relu, scale).

This example demonstrates how Composable Kernel supports efficient fusion of elementwise operations and tensor permutation for deep learning and data layout transformations.
