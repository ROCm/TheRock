# Client Example: N-Dimensional Convolution Backward Data

## Theory

This client example demonstrates **N-dimensional convolution backward data** for 3D inputs, supporting multiple data types (FP16, FP32). The backward data operation computes the gradient of the input tensor with respect to the loss, given the output gradient and the weights. This is essential for training CNNs and 3D vision models.

**Mathematical Formulation:**
For input $X$, weights $W$, and output gradient $dY$:
$$
dX = \text{ConvBwdData}(dY, W)
$$

- Supports 3D convolution (ND can be extended).
- Utilizes implicit GEMM for efficient computation.

**Algorithmic Background:**
- The backward data operation is implemented as a convolution with transformed coordinates.
- Used in training pipelines for 3D CNNs, medical imaging, and volumetric data.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/15_convnd_bwd_data
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (3D backward data, FP16)
./conv3d_bwd_data_fp16

# Example run (3D backward data, FP32)
./conv3d_bwd_data_fp32
```

## Source Code Structure

### Directory Layout
```
client_example/15_convnd_bwd_data/
├── conv3d_bwd_data_fp16.cpp         # 3D convolution backward data (FP16)
├── conv3d_bwd_data_fp32.cpp         # 3D convolution backward data (FP32)
├── common.hpp                       # Common utilities for convolution
├── CMakeLists.txt                   # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input/output tensors, configures convolution parameters, launches the backward data kernel, and verifies the result.
- **Backward data kernel invocation**:  
  Uses the Composable Kernel device API to launch convolution backward data for different data types.

---

## Additional Details

- Supports FP16 and FP32 for 3D convolution.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [10_grouped_convnd_bwd_data](../10_grouped_convnd_bwd_data/README.md): Grouped convolution backward data
- [17_convnd_bwd_data](../../example/17_convnd_bwd_data/README.md): Convolution backward data in the main example directory

---
[Back to Client Examples](../README.md)
