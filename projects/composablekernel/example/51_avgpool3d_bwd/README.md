# 3D Average Pooling Backward

This example demonstrates the **backward pass of 3D average pooling**. This operation computes the gradient of the loss with respect to the input of a 3D average pooling layer, which is essential for training 3D convolutional neural networks used in video analysis, medical imaging, and volumetric data processing.

## Mathematical Formulation

The backward pass of 3D average pooling distributes the output gradients uniformly across all input positions that contributed to each pooling window.

Given:
- Input tensor `X` with shape `[N, C, D_in, H_in, W_in]`
- Output gradients `dL/dY` with shape `[N, C, D_out, H_out, W_out]`
- Pooling parameters: window size `(pool_d, pool_h, pool_w)`, stride `(stride_d, stride_h, stride_w)`, padding `(pad_d, pad_h, pad_w)`

The backward pass computes input gradients `dL/dX` with the same shape as `X`.

For 3D average pooling, the gradient is distributed uniformly across all positions in each pooling window:
$\frac{\partial L}{\partial X_{ncdhw}} = \sum_{\text{windows containing } (d,h,w)} \frac{1}{|W|} \cdot \frac{\partial L}{\partial Y_{ncd'h'w'}}$

Where `|W|` is the effective window size (accounting for padding and boundaries), and the sum is over all output positions whose pooling windows include the input position `(d,h,w)`.

## Algorithmic Strategy: Parallel Gradient Distribution

The backward pass distributes gradients from output positions to all input positions that contributed to each pooling window.

1.  **Grid Scheduling**: The computation can be parallelized over either input or output tensor elements, depending on the implementation strategy.

2.  **Gradient Distribution Algorithm** (output-centric approach):
    -   **Initialize**: Set all input gradients to zero.
    -   **For each output position**: Each thread processes one output gradient position `(n, c, d_out, h_out, w_out)`.
    -   **Calculate Input Window**: Determine the 3D input window that contributed to this output position.
    -   **Effective Window Size**: Calculate the actual number of input elements in the window (accounting for padding and boundaries).
    -   **Distribute Gradient**: Add `grad_output / window_size` to each input position in the window (using atomic operations for thread safety).

3.  **Boundary Handling**: Careful handling of:
    -   **Padding**: Input positions outside the valid range should not receive gradients
    -   **Partial Windows**: Windows at boundaries may have fewer than `pool_d × pool_h × pool_w` elements
    -   **Edge Cases**: Zero-sized windows or invalid configurations

4.  **Memory Access Optimization**:
    -   Coalesced reading from output gradients
    -   Efficient atomic operations for gradient accumulation
    -   Minimized redundant boundary checks

## Source Code Organization

-   [`avgpool3d_bwd_xdl.cpp`](./avgpool3d_bwd_xdl.cpp): The main example file. It sets up the input tensor, output gradients, pooling parameters, and instantiates the `DeviceAvgpool3dBwd` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_avgpool3d_bwd.hpp`](../../include/ck/tensor_operation/gpu/device/device_avgpool3d_bwd.hpp): The high-level device interface for 3D average pooling backward operations.
-   [`../../include/ck/tensor_operation/gpu/grid/gridwise_avgpool3d_bwd.hpp`](../../include/ck/tensor_operation/gpu/grid/gridwise_avgpool3d_bwd.hpp): The grid-wise kernel implementing the gradient distribution algorithm.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/51_avgpool3d_bwd
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
./avgpool3d_bwd_xdl

# Run with verification, data initialization, and timing
./avgpool3d_bwd_xdl 1 2 1
```

## Comparison with Max Pooling Backward

3D average pooling backward differs significantly from max pooling backward:

| Aspect | Max Pooling | Average Pooling |
|--------|-------------|-----------------|
| **Gradient Flow** | Sparse (only to argmax positions) | Dense (to all window positions) |
| **Distribution** | Single position per window | Uniform across window |
| **Computation** | Requires argmax information | Simple arithmetic division |
| **Memory Pattern** | Irregular write pattern | Regular, predictable pattern |
| **Atomic Operations** | Needed for gradient routing | Needed for accumulation |

## Applications in 3D Deep Learning

3D average pooling backward is essential for training models that process volumetric data:

-   **Video Understanding**: 3D CNNs for action recognition, video classification, and temporal modeling
-   **Medical Imaging**: 3D segmentation and classification of CT scans, MRI, and other volumetric medical data
-   **3D Object Recognition**: Processing 3D point clouds, voxel grids, and depth data
-   **Scientific Computing**: Climate modeling, fluid dynamics, and other physics simulations
-   **Augmented Reality**: 3D scene understanding and object tracking in real-time applications
