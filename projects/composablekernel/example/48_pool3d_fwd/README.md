# 3D Pooling Forward

This example demonstrates a **3D pooling forward operation**. Pooling is a fundamental operation in convolutional neural networks that reduces the spatial dimensions of feature maps while retaining important information. 3D pooling extends this concept to three-dimensional data, commonly used in video analysis, medical imaging, and 3D computer vision applications.

## Mathematical Formulation

3D pooling operates on 5D tensors with shape `[N, C, D, H, W]` where:
- `N` is the batch size
- `C` is the number of channels
- `D`, `H`, `W` are the depth, height, and width dimensions

The operation applies a pooling function over 3D windows of the input tensor.

For each output position `(n, c, d_out, h_out, w_out)`:
$\text{Out}_{ncd_{out}h_{out}w_{out}} = \text{Pool}(\{X_{ncd'h'w'} : d' \in W_d, h' \in W_h, w' \in W_w\})$

Where:
- $W_d$, $W_h$, $W_w$ define the 3D pooling window
- `Pool` is the pooling function (e.g., max or average)

**Max Pooling**: $\text{Pool}(S) = \max(S)$
**Average Pooling**: $\text{Pool}(S) = \frac{1}{|S|} \sum_{x \in S} x$

The window positions are determined by:
- **Window size**: `(pool_d, pool_h, pool_w)`
- **Stride**: `(stride_d, stride_h, stride_w)`
- **Padding**: `(pad_d, pad_h, pad_w)`

## Algorithmic Strategy: Parallel Window-based Computation

3D pooling is implemented as a parallel algorithm where each thread computes one output element.

1.  **Grid Scheduling**: The output tensor elements are distributed across GPU threads. Each thread is assigned to compute one element of the output tensor.

2.  **Window Processing**: For each output position, a thread:
    -   **Calculate Input Window**: Determines the 3D input window corresponding to the current output position based on stride, padding, and window size.
    -   **Boundary Handling**: Checks for boundary conditions and padding, ensuring that only valid input positions are processed.
    -   **Apply Pooling Function**: 
        -   **Max Pooling**: Iterates through the window and finds the maximum value.
        -   **Average Pooling**: Iterates through the window, accumulates values, and computes the average.
    -   **Store Result**: Writes the computed result to the output tensor.

3.  **Memory Access Optimization**: The kernel is optimized for memory access patterns, using techniques like:
    -   Coalesced memory access where possible
    -   Shared memory for frequently accessed data
    -   Efficient handling of boundary conditions

## Source Code Organization

-   [`pool3d_fwd_xdl.cpp`](./pool3d_fwd_xdl.cpp): The main example file. It sets up a 3D input tensor, defines pooling parameters (window size, stride, padding), and instantiates the `DevicePool3dFwd` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_pool3d_fwd.hpp`](../../include/ck/tensor_operation/gpu/device/device_pool3d_fwd.hpp): The high-level device interface for 3D pooling operations.
-   [`../../include/ck/tensor_operation/gpu/grid/gridwise_pool3d_fwd.hpp`](../../include/ck/tensor_operation/gpu/grid/gridwise_pool3d_fwd.hpp): The grid-wise kernel implementing the parallel 3D pooling algorithm.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/48_pool3d_fwd
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run the Example
```bash
# Run the example with default settings
./pool3d_fwd_xdl

# Run with verification, data initialization, and timing
./pool3d_fwd_xdl 1 2 1
```

## Applications

3D pooling is essential in several domains that process volumetric or temporal data.

-   **Video Analysis**: In video understanding tasks, 3D CNNs use 3D pooling to reduce temporal and spatial dimensions while preserving important motion and appearance features.
-   **Medical Imaging**: 3D medical images (CT scans, MRI) require 3D pooling for feature extraction while maintaining spatial relationships in all three dimensions.
-   **3D Computer Vision**: Object detection and segmentation in 3D point clouds or voxel grids use 3D pooling for hierarchical feature learning.
-   **Action Recognition**: Video action recognition models use 3D pooling to aggregate features across temporal and spatial dimensions.
-   **Volumetric Data Processing**: Scientific applications processing 3D volumetric data (weather modeling, fluid dynamics) use 3D pooling for multi-scale analysis.
