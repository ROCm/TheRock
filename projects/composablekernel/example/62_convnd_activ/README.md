# N-Dimensional Convolution with Activation

This example demonstrates an **N-dimensional convolution forward pass fused with an activation function**. This fusion pattern combines the convolution operation with elementwise activation functions in a single kernel, which is extremely common in convolutional neural networks and provides significant performance benefits.

## Mathematical Formulation

The operation performs an N-dimensional convolution followed immediately by an activation function.

1.  **N-Dimensional Convolution**: A standard N-dimensional forward convolution.
    $C_{temp} = \text{Conv}_{\text{ND}}(\text{In}, \text{W})$
    Where `In` is the input tensor, `W` is the weight tensor, and the convolution can be 1D, 2D, 3D, or higher-dimensional.

2.  **Activation Function**: Apply an elementwise activation function to the convolution result.
    $\text{Out} = \text{Activation}(C_{temp})$
    Common activation functions include:
    - **ReLU**: $\text{ReLU}(x) = \max(0, x)$
    - **Sigmoid**: $\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$
    - **Tanh**: $\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
    - **GELU**: $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard Gaussian CDF
    - **Swish**: $\text{Swish}(x) = x \cdot \text{Sigmoid}(x)$

The key optimization is that the intermediate tensor `C_temp` is **never written to global memory**. The activation function is applied directly to the convolution result held in registers.

## Algorithmic Strategy: Implicit GEMM with Fused Activation Epilogue

The implementation uses the implicit GEMM algorithm for convolution with the activation function fused into the epilogue.

1.  **Implicit GEMM Core**: The convolution is transformed into an equivalent GEMM operation:
    -   **Input Transformation**: The input tensor is implicitly transformed using the im2col operation.
    -   **Matrix Multiplication**: The core computation is performed as a tiled matrix multiplication.
    -   **Output Accumulation**: Results are accumulated in registers as standard GEMM tiles.

2.  **Fused Activation Epilogue**: Before storing results to global memory:
    -   **Elementwise Activation**: Apply the activation function to each element in the accumulated tile.
    -   **Vectorized Operations**: Use vectorized instructions where possible for activation computation.
    -   **Store Activated Result**: Write the final activated output directly to global memory.

This approach eliminates the need for a separate activation kernel and the associated memory bandwidth for reading and writing the intermediate convolution result.

## Source Code Organization

-   [`convnd_activ_xdl.cpp`](./convnd_activ_xdl.cpp): The main example file. It sets up the N-dimensional input tensor, weight tensor, specifies the activation function, and instantiates the `DeviceConvNdActiv` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_convnd_activ.hpp`](../../include/ck/tensor_operation/gpu/device/device_convnd_activ.hpp): The device interface for N-dimensional convolution with activation fusion.
-   The underlying kernel implements the implicit GEMM algorithm with templated activation functions in the epilogue.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/62_convnd_activ
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
./convnd_activ_xdl

# Run with verification, data initialization, and timing
./convnd_activ_xdl 1 2 1
```

## Applications

Convolution with activation fusion is fundamental to many neural network architectures.

-   **Convolutional Neural Networks (CNNs)**: Nearly every convolutional layer in CNNs is followed by an activation function, making this fusion extremely valuable.
-   **Computer Vision Models**: Image classification, object detection, and segmentation networks all benefit from this fusion.
-   **3D CNNs**: Video analysis and medical imaging applications using 3D convolutions with activations.
-   **Mobile and Edge Deployment**: The reduced memory bandwidth makes this fusion especially valuable for resource-constrained environments.
-   **Training Acceleration**: Reducing the number of kernel launches and memory operations accelerates both forward and backward passes during training.

## Performance Benefits

This fusion provides several performance advantages:

-   **Reduced Memory Bandwidth**: Eliminates one full read/write cycle of the intermediate tensor
-   **Improved Cache Locality**: Data stays in cache/registers between convolution and activation
-   **Fewer Kernel Launches**: Reduces GPU kernel launch overhead
-   **Better Instruction Scheduling**: Allows better interleaving of compute and memory operations

## Activation Function Considerations

Different activation functions have different computational characteristics:

-   **ReLU**: Very fast, just a comparison and conditional assignment
-   **Sigmoid/Tanh**: Require expensive exponential calculations
-   **GELU**: Involves error function computation, typically approximated
-   **Swish**: Combines multiplication with sigmoid computation

The choice of activation function can significantly impact the overall performance of the fused kernel, with simpler functions like ReLU providing the best performance improvements.
