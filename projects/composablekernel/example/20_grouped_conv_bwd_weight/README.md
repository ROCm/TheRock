# Grouped Convolution Backward Pass for Weights

This example demonstrates the backward weight pass for a **grouped convolution**, often denoted as `grouped_conv_bwd_weight`. This operation is essential for training neural networks that use grouped or depthwise convolutions, such as ResNeXt, MobileNets, and EfficientNets. Its purpose is to compute the gradient of the loss function with respect to the convolution's *filter weights*, which is then used by an optimizer (like SGD or Adam) to update the model's parameters.

## Mathematical Formulation

The backward weight pass computes the gradient $\frac{\partial L}{\partial W}$, given the input tensor from the forward pass, `In`, and the gradient from the subsequent layer, `dL/dOut`.

For a single group `g`, the operation is mathematically equivalent to a convolution between the input tensor for that group, `In_[g]`, and the output gradient tensor for that group, `dL/dOut_[g]`.

$\frac{\partial L}{\partial W_{[g]}} = \text{In}_{[g]} \star \frac{\partial L}{\partial \text{Out}_{[g]}}$

This operation correlates the input activations with the output error signals to determine how each weight should be adjusted to reduce the overall loss. The total gradient `dL/dW` is the collection of gradients for all `G` groups.

## Algorithmic Strategy: Implicit Grouped GEMM

This operation is a perfect candidate for the **Grouped GEMM** primitive. The convolution for each of the `G` groups is independently transformed into a GEMM problem, and all `G` GEMMs are executed in a single kernel launch.

For each group `g`:

1.  **Input to Columns (`im2col`)**: The input tensor `In_[g]` is logically unrolled into a matrix `In'_[g]`. This is the same `im2col` transformation used in the forward pass. This matrix becomes the "A" matrix in the GEMM.

2.  **Output Gradient Reshaping**: The output gradient tensor `dL/dOut_[g]` is logically reshaped into a matrix `(dL/dOut)'_[g]`. This matrix becomes the "B" matrix in the GEMM.

3.  **Implicit Grouped GEMM**: The weight gradient `dL/dW_[g]` is computed by a single GEMM:
    $(\text{dL/dW})'_{[g]} = (\text{dL/dOut})'_{[g]} \times (\text{In}'_{[g]})^T$

The key to performance is that this is executed as a **Grouped GEMM**. The `DeviceGroupedConvBwdWeight` interface takes the `G` independent problems and maps them to a `DeviceGroupedGemm` kernel. This kernel schedules the `G` independent GEMMs across the GPU's compute units. The `im2col` transformation is performed implicitly; the GEMM kernel reads data directly from the original `In` and `dL/dOut` tensors in the correct pattern, avoiding the materialization of large intermediate matrices.

This approach is highly efficient as it leverages the task-parallel nature of the grouped convolution and the computational efficiency of highly optimized GEMM kernels.

## Source Code Organization

-   [`grouped_conv_bwd_weight_xdl.cpp`](./grouped_conv_bwd_weight_xdl.cpp): The main example file. It sets up a grouped convolution problem and instantiates the `DeviceGroupedConvBwdWeight` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp`](../../include/ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp): The high-level device interface. It internally translates the grouped convolution problem into a set of arguments for the `DeviceGroupedGemm` interface.
-   [`../../include/ck/tensor_operation/gpu/device/device_grouped_gemm.hpp`](../../include/ck/tensor_operation/gpu/device/device_grouped_gemm.hpp): The underlying Grouped GEMM device interface that is called by the grouped convolution operator.
-   [`../../library/include/ck/library/reference_tensor_operation/cpu/reference_grouped_conv_bwd_weight.hpp`](../../library/include/ck/library/reference_tensor_operation/cpu/reference_grouped_conv_bwd_weight.hpp): A CPU reference implementation for verifying the correctness of the GPU kernel.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/20_grouped_conv_bwd_weight
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
./grouped_conv_bwd_weight_xdl

# Run with verification, data initialization, and timing
./grouped_conv_bwd_weight_xdl 1 2 1
```

## Importance in Modern CNNs

Grouped and depthwise convolutions are the cornerstone of many efficient, state-of-the-art CNN architectures.
-   **Parameter Efficiency**: By not connecting every input channel to every output channel, grouped convolutions significantly reduce the number of weights in a layer, leading to smaller and faster models.
-   **Depthwise Separable Convolutions**: Used in MobileNets, EfficientNets, and Xception, these layers factorize a standard convolution into a depthwise convolution (a grouped convolution with `G = C`) and a pointwise convolution (`1x1` conv). The backward pass for the depthwise part requires an efficient `grouped_conv_bwd_weight` implementation.
-   **ResNeXt**: This architecture introduced the "cardinality" dimension, which is simply the number of groups in a grouped convolution, demonstrating that increasing the number of groups can be more effective than increasing layer depth or width.

An optimized `grouped_conv_bwd_weight` kernel is therefore not an exotic feature but a critical requirement for training a wide range of modern and efficient deep learning models.
