# Client Example: N-Dimensional Convolution Forward

## Theory

This client example demonstrates **N-dimensional convolution forward** for 3D inputs, supporting multiple data types (FP16, FP32, FP8 composite). Convolution is a fundamental operation in deep learning, especially in convolutional neural networks (CNNs) for images, audio, and volumetric data.

**Mathematical Formulation:**
Given input $X$, weights $W$:
$$
Y = \text{Conv}(X, W)
$$

- Supports 3D convolution (ND can be extended).
- Utilizes implicit GEMM for efficient computation.

**Algorithmic Background:**
- The forward convolution operation is implemented as a convolution with transformed coordinates.
- Used in inference and training pipelines for 3D CNNs, medical imaging, and volumetric data.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/16_convnd_fwd
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (3D forward, FP16)
./conv3d_fwd_fp16

# Example run (3D forward, FP32)
./conv3d_fwd_fp32

# Example run (3D forward, FP16 compute with FP8)
./conv3d_fwd_fp16_comp_fp8
```

## Source Code Structure

### Directory Layout
```
client_example/16_convnd_fwd/
├── conv3d_fwd_fp16.cpp         # 3D convolution forward (FP16)
├── conv3d_fwd_fp32.cpp         # 3D convolution forward (FP32)
├── conv3d_fwd_fp16_comp_fp8.cpp # 3D convolution forward (FP16 compute, FP8)
├── common.hpp                  # Common utilities for convolution
├── CMakeLists.txt              # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input/output tensors, configures convolution parameters, launches the forward kernel, and verifies the result.
- **Forward convolution kernel invocation**:  
  Uses the Composable Kernel device API to launch convolution forward for different data types.

---

## Additional Details

- Supports FP16, FP32, and FP8 composite for 3D convolution.
- Parameters can be adjusted in the source files for different workloads. The following parameters are configurable:
  - `NumDimSpatial`: Number of spatial dimensions (default: 3 for 3D convolution)
  - `G`: Number of groups (default: 1)
  - `N`: Batch size (default: 64)
  - `K`: Number of output channels (default: 128)
  - `C`: Number of input channels (default: 64)
  - `Z`, `Y`, `X`: Filter/kernel dimensions (default: 3x3x3)
  - `Di`, `Hi`, `Wi`: Input dimensions - depth, height, width (default: 28x28x3)
  - `Do`, `Ho`, `Wo`: Output dimensions - depth, height, width (default: 28x28x3)

---

## Related Examples

- [09_convnd_fwd](../../example/09_convnd_fwd/README.md): N-dimensional convolution in the main example directory
- [30_grouped_conv_fwd_multiple_d](../../example/30_grouped_conv_fwd_multiple_d/README.md): Grouped convolution forward with multiple D

---
[Back to Client Examples](../README.md)
