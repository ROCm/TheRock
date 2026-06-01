# Client Example: Pooling Operations (2D Max, 3D Avg)

## Theory

This client example demonstrates **pooling operations** for 2D max pooling and 3D average pooling, including both forward and backward passes. Pooling is used in convolutional neural networks (CNNs) for spatial downsampling, translation invariance, and reducing computation.

**Mathematical Formulation:**
- **Max Pooling (2D):** $Y_{n,c,h,w} = \max_{i,j} X_{n,c,h \cdot s_H + i, w \cdot s_W + j}$
- **Average Pooling (3D):** $Y_{n,c,d,h,w} = \frac{1}{k_D k_H k_W} \sum_{i,j,k} X_{n,c,d \cdot s_D + i, h \cdot s_H + j, w \cdot s_W + k}$

Where $s_H, s_W, s_D$ are strides, $k_H, k_W, k_D$ are kernel sizes.

**Algorithmic Background:**
- Forward pass computes the pooled output.
- Backward pass computes the gradient with respect to the input.
- Handles padding and boundary conditions.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/19_pool
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (2D max pool forward)
./max_pool2d_fwd

# Example run (2D max pool backward)
./max_pool2d_bwd

# Example run (3D avg pool forward)
./avg_pool3d_fwd

# Example run (3D avg pool backward)
./avg_pool3d_bwd
```

## Source Code Structure

### Directory Layout
```
client_example/19_pool/
├── max_pool2d_fwd.cpp         # 2D max pooling forward
├── max_pool2d_bwd.cpp         # 2D max pooling backward
├── avg_pool3d_fwd.cpp         # 3D average pooling forward
├── avg_pool3d_bwd.cpp         # 3D average pooling backward
├── CMakeLists.txt             # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures pooling parameters, launches the forward or backward kernel, and verifies the result.
- **Pooling kernel invocation**:  
  Uses the Composable Kernel device API to launch pooling operations for different modes.

---

## Additional Details

- Supports both max and average pooling, forward and backward.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [13_pool2d_fwd](../../example/13_pool2d_fwd/README.md): 2D pooling in the main example directory
- [48_pool3d_fwd](../../example/48_pool3d_fwd/README.md): 3D pooling in the main example directory
- [49_maxpool2d_bwd](../../example/49_maxpool2d_bwd/README.md): 2D max pool backward in the main example directory
- [51_avgpool3d_bwd](../../example/51_avgpool3d_bwd/README.md): 3D avg pool backward in the main example directory

---
[Back to Client Examples](../README.md)
