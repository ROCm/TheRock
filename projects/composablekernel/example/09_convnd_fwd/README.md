# N-Dimensional Convolution Forward

## Theory

This example demonstrates the **N-dimensional convolution forward pass** using Composable Kernel. Convolution is a fundamental operation in deep learning, especially in convolutional neural networks (CNNs) for images, audio, and volumetric data.

**Mathematical Formulation:**
Given:
- Input tensor: $X[N, C_{in}, D_1, D_2, ..., D_n]$
- Weight tensor: $W[C_{out}, C_{in}, K_1, K_2, ..., K_n]$
- Output tensor: $Y[N, C_{out}, O_1, O_2, ..., O_n]$

The convolution computes:
$$
Y[n, c_{out}, o_1, ..., o_n] = \sum_{c_{in}} \sum_{k_1} ... \sum_{k_n} X[n, c_{in}, o_1 + k_1, ..., o_n + k_n] \cdot W[c_{out}, c_{in}, k_1, ..., k_n]
$$

Stride, padding, and dilation parameters control the mapping between input and output indices.

**Algorithmic Background:**
- Composable Kernel implements convolution as an implicit GEMM (matrix multiplication) for efficiency.
- The input and weight tensors are transformed into matrices, and the convolution is performed as a GEMM.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/09_convnd_fwd
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j
```

### Run ```example_convnd_fwd_xdl```

```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4: N spatial dimensions (default 2)
#Following arguments (depending on number of spatial dims):
# N, K, C, 
# <filter spatial dimensions>, (ie Y, X for 2D)
# <input image spatial dimensions>, (ie Hi, Wi for 2D)
# <strides>, (ie Sy, Sx for 2D)
# <dilations>, (ie Dy, Dx for 2D)
# <left padding>, (ie LeftPy, LeftPx for 2D)
# <right padding>, (ie RightPy, RightPx for 2D)
./bin/example_convnd_fwd_xdl 0 1 100
```
## Source Code Structure

### Directory Layout
```
example/09_convnd_fwd/
├── convnd_fwd_xdl.cpp         # Main example: sets up, runs, and verifies N-D convolution
include/ck/tensor_operation/gpu/device/
│   └── device_convnd_fwd.hpp       # Device-level convolution API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_convnd_fwd_xdl.hpp   # XDL-based convolution implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_convnd_fwd_xdl.hpp # Grid-level convolution kernel
include/ck/tensor_operation/gpu/block/
    └── blockwise_convnd_fwd_xdl.hpp # Block-level convolution
```

### Key Classes and Functions

- **DeviceConvNdFwd** (in `device_convnd_fwd.hpp`):  
  Device API for N-dimensional convolution.
- **gridwise_convnd_fwd_xdl** (in `gridwise_convnd_fwd_xdl.hpp`):  
  Implements the tiled/blocking convolution kernel.
- **blockwise_convnd_fwd_xdl** (in `blockwise_convnd_fwd_xdl.hpp`):  
  Handles block-level computation and shared memory tiling.

This example demonstrates how Composable Kernel implements efficient N-dimensional convolution using implicit GEMM, supporting a wide range of deep learning applications.
