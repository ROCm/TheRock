# Elementwise Normalization

This example demonstrates a fused **elementwise operation followed by normalization**. This pattern combines elementwise tensor arithmetic with a normalization operation in a single kernel, which is particularly useful for implementing custom normalization layers or fused activation-normalization blocks.

## Mathematical Formulation

The operation performs an elementwise computation followed by a normalization operation.

1.  **Elementwise Stage**: An elementwise operation is applied to one or more input tensors.
    $C_{temp} = f(A, B, \dots)$
    Where `f` is a user-defined elementwise function that operates on corresponding elements of the input tensors.

2.  **Normalization Stage**: The result is then normalized. The normalization can be performed along specified dimensions.
    -   **Compute Statistics**: For each normalization group, compute the mean and variance.
        $\mu = \frac{1}{N} \sum C_{temp}$
        $\sigma^2 = \frac{1}{N} \sum (C_{temp} - \mu)^2$
    -   **Normalize**: Apply the normalization formula.
        $\hat{C} = \frac{C_{temp} - \mu}{\sqrt{\sigma^2 + \epsilon}}$
    -   **Scale and Shift**: Apply learnable parameters.
        $D = \gamma \cdot \hat{C} + \beta$

The key optimization is that the intermediate tensor `C_temp` is **never written to global memory**. The elementwise computation feeds directly into the normalization calculation.

## Algorithmic Strategy: Fused Elementwise with Online Normalization

The implementation combines elementwise computation with an online normalization algorithm.

1.  **Grid Scheduling**: The normalization groups are distributed among thread blocks. Each block handles one or more normalization groups.

2.  **Fused Two-Pass Algorithm**:
    -   **Pass 1 - Compute Elementwise and Moments**:
        -   Threads cooperatively load input tensors and apply the elementwise function `f`.
        -   The elementwise results are kept in registers/shared memory.
        -   **Welford's Algorithm**: Threads use Welford's online algorithm to compute the mean and variance of the elementwise results within their normalization group.
        -   **Intra-Block Reduction**: A parallel reduction in shared memory computes the final statistics for the group.
    -   **Pass 2 - Normalize and Store**:
        -   Using the computed statistics, threads apply the normalization formula to their elementwise results.
        -   The final normalized result is written to the output tensor `D`.

This approach ensures that the elementwise computation is performed only once, and the results are immediately consumed by the normalization process without requiring additional memory bandwidth.

## Source Code Organization

-   [`elementwise_normalization_xdl.cpp`](./elementwise_normalization_xdl.cpp): The main example file. It sets up the input tensors, defines the elementwise operation and normalization parameters, and instantiates the `DeviceElementwiseNormalization` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_elementwise_normalization.hpp`](../../include/ck/tensor_operation/gpu/device/device_elementwise_normalization.hpp): The high-level device interface for the fused elementwise normalization operation.
-   The underlying grid-wise kernel implements the complex fusion of elementwise operations with the two-pass normalization algorithm.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/45_elementwise_normalization
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
./elementwise_normalization_xdl

# Run with verification, data initialization, and timing
./elementwise_normalization_xdl 1 2 1
```

## Applications

This fused operation is valuable for implementing custom normalization layers and optimizing activation-normalization sequences.

-   **Custom Activation-Normalization Blocks**: Some architectures use non-standard activation functions followed by normalization. For example, a Swish activation followed by layer normalization can be fused into a single kernel using this pattern.
-   **Residual Connection with Normalization**: In some variants of residual networks, the residual addition is immediately followed by normalization. This can be expressed as an elementwise addition (residual) followed by normalization.
-   **Preprocessing Pipelines**: In data preprocessing, tensors might need elementwise transformations (e.g., color space conversion) followed by normalization (e.g., standardization). This kernel can fuse these operations.
-   **Research Architectures**: Novel normalization techniques often involve custom elementwise operations before the normalization step. This kernel provides a flexible foundation for implementing such research ideas efficiently.
