# N-Dimensional Convolution Backward Pass for Data

This example demonstrates the backward data pass of an N-dimensional convolution, often denoted as `conv_bwd_data`. This operation is a crucial part of the backpropagation algorithm for training Convolutional Neural Networks (CNNs). Its purpose is to compute the gradient of the loss function with respect to the convolution's *input data*, which is then passed back to the preceding layer in the network.

## Mathematical Formulation

The backward data pass computes the gradient $\frac{\partial L}{\partial \text{In}}$, given the gradient from the subsequent layer, $\frac{\partial L}{\partial \text{Out}}$, and the filter weights `W` used in the forward pass.

Let the forward convolution be defined as:
$\text{Out} = \text{In} \star W$

The backward data pass is mathematically equivalent to a "full" convolution between the output gradient tensor `dL/dOut` and the 180-degree rotated (or transposed and flipped) weight tensor `W`.

$\frac{\partial L}{\partial \text{In}} = \frac{\partial L}{\partial \text{Out}} \star \text{rot180}(W)$

This operation propagates the error signal from the output back to the input, weighted by the same filters that were used in the forward pass.

## Algorithmic Strategy: Implicit GEMM

As with the forward pass, the most efficient way to implement the backward data pass on a GPU is to transform the convolution into a General Matrix-Matrix Multiplication (GEMM) problem.

1.  **Output Gradient Reshaping**: The output gradient tensor `dL/dOut` is logically reshaped into a matrix `dL/dOut'` of shape `[K, (N*Ho*Wo)]`. This becomes the "A" matrix in the GEMM.

2.  **Weight Reshaping**: The weight tensor `W` is logically reshaped into a matrix `W'` of shape `[K, (C*Y*X)]`. This becomes the "B" matrix in the GEMM.

3.  **Implicit GEMM**: The core computation is then formulated as a GEMM operation. However, the output of this GEMM is not a simple matrix; it's the `dL/dIn` tensor.
    $(\text{dL/dIn})' = (W')^T \times (\text{dL/dOut})'$

    The key insight is that this operation can be performed without explicitly forming the matrices. The GEMM kernel is designed to read from `dL/dOut` and `W` and write its results directly to the appropriate locations in the `dL/dIn` tensor. This process is sometimes referred to as an "implicit `col2im`" (column-to-image), as it is the inverse of the `im2col` transformation used in the forward pass.

This "implicit GEMM" approach is highly efficient. It avoids the massive memory and bandwidth overhead of materializing intermediate matrices, which is critical for performance.

## Source Code Organization

-   [`conv_bwd_data_xdl.cpp`](./conv_bwd_data_xdl.cpp): The main example file that defines the parameters for a 2D convolution and instantiates the generic `DeviceConvNdBwdData` kernel to compute the input gradients.
-   [`../../include/ck/tensor_operation/gpu/device/device_conv_bwd_data.hpp`](../../include/ck/tensor_operation/gpu/device/device_conv_bwd_data.hpp): The high-level device interface for the backward data convolution. It is templated on the dimensionality, layouts, and data types of the problem.
-   [`../../include/ck/tensor_operation/gpu/grid/gridwise_gemm_implicit_gemm_v1r2_xdlops_nchw_kcyx_nkhw.hpp`](../../include/ck/tensor_operation/gpu/grid/gridwise_gemm_implicit_gemm_v1r2_xdlops_nchw_kcyx_nkhw.hpp): An example of a specific grid-wise kernel that implements the implicit GEMM algorithm for the backward data pass. The library contains multiple such kernels optimized for different layouts and problem types, and the `DeviceConvNdBwdData` interface selects the most appropriate one.
-   [`../../library/include/ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp`](../../library/include/ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp): A CPU reference implementation used to verify the correctness of the GPU kernel's output.

## Build and Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build the Example
```bash
cd /path/to/composable_kernel/example/17_convnd_bwd_data
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run the Example
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4: num_dim_spatial(1|2|3)
#arg5 to ...: N, K, C, [Z,] [Y,] X, [Di,] [Hi,] Wi, S[z,] [Sy,] Sx, [Dz,] [Dy,] Dx, [LeftPz,] [LeftPy,] LeftPx, [RightPy,] [RightPy,] RightPx
./bin/example_convnd_bwd_data_xdl 0 1 5 
```

Result
```
in_n_c_hi_wi: dim 4, lengths {128, 128, 71, 71}, strides {645248, 1, 9088, 128}
wei_k_c_y_x: dim 4, lengths {256, 128, 3, 3}, strides {1152, 1, 384, 128}
out_n_k_ho_wo: dim 4, lengths {128, 256, 36, 36}, strides {331776, 1, 9216, 256}
arg.a_grid_desc_k0_m_k1_container_{128, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{128, 128, 8}
arg.c_grid_desc_m_n_container_{ 175232, 128}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 2, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {1369, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
arg.a_grid_desc_k0_m_k1_container_{64, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{64, 128, 8}
arg.c_grid_desc_m_n_container_{ 175232, 128}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 2, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {1369, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
arg.a_grid_desc_k0_m_k1_container_{64, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{64, 128, 8}
arg.c_grid_desc_m_n_container_{ 175232, 128}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 2, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {1369, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
arg.a_grid_desc_k0_m_k1_container_{32, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{32, 128, 8}
arg.c_grid_desc_m_n_container_{ 175232, 128}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 2, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {1369, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
Perf: 1.40031 ms, 69.8734 TFlops, 179.037 GB/s
```

## Relationship to Other Passes

The training of a single convolutional layer requires three distinct steps:

1.  **Forward Pass (`conv_fwd`)**: Computes the output feature maps.
    -   `Out = In * W`
2.  **Backward Data Pass (`conv_bwd_data`)**: Computes the gradient with respect to the input, propagating the error to the previous layer. This is the focus of the current example.
    -   `dL/dIn = dL/dOut * rot180(W)`
3.  **Backward Weight Pass (`conv_bwd_weight`)**: Computes the gradient with respect to the weights, which is needed for the weight update.
    -   `dL/dW = In * dL/dOut`

All three passes are critical for training a CNN, and all are typically implemented as high-performance implicit GEMM operations.
