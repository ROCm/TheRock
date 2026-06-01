# Complex Tensor Contraction with Bilinear Operations

This example demonstrates a **complex tensor contraction combined with bilinear operations**. This advanced operation handles complex-valued tensors (with real and imaginary components) and performs both tensor contractions and bilinear transformations, which is particularly important for applications in quantum computing, signal processing, and advanced scientific computing.

## Mathematical Formulation

The operation combines complex tensor contraction with bilinear operations on complex-valued data.

Given complex tensors with real and imaginary components:
- Complex tensor `A = A_real + i × A_imag`
- Complex tensor `B = B_real + i × B_imag`  
- Auxiliary complex tensors `D, E, ...`

1.  **Complex Tensor Contraction**: Perform tensor contraction using Einstein summation on complex tensors.
    $C_{temp} = \text{einsum}(\text{pattern}, A, B)$
    
    For complex multiplication: $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$

2.  **Bilinear Operations**: Apply bilinear transformations involving the contraction result and auxiliary tensors.
    $F = \text{BilinearOp}(C_{temp}, D, E, \ldots)$

The bilinear operations can include various combinations such as:
- $F = C_{temp} \odot D + E$ (elementwise multiply and add)
- $F = \alpha \cdot C_{temp} + \beta \cdot (D \odot E)$ (scaled combinations)
- More complex multi-term bilinear expressions

## Algorithmic Strategy: Complex-Arithmetic GEMM with Bilinear Epilogue

The implementation handles complex arithmetic throughout the computation pipeline.

1.  **Complex Tensor-to-GEMM Mapping**: 
    -   **Real/Imaginary Separation**: Complex tensors are logically separated into real and imaginary components
    -   **Complex GEMM**: Four real GEMM operations represent one complex GEMM:
        - $C_{real} = A_{real} \times B_{real} - A_{imag} \times B_{imag}$
        - $C_{imag} = A_{real} \times B_{imag} + A_{imag} \times B_{real}$

2.  **Multi-Component Computation**: Within each thread block:
    -   **Parallel Real/Imaginary Processing**: Simultaneously compute real and imaginary components
    -   **Complex Accumulation**: Maintain separate accumulators for real and imaginary parts
    -   **Register Management**: Carefully orchestrate register usage for multiple complex components

3.  **Complex Bilinear Epilogue**: 
    -   **Load Complex Auxiliary Tensors**: Read real and imaginary components of auxiliary tensors
    -   **Complex Bilinear Operations**: Apply the specified bilinear transformations using complex arithmetic
    -   **Complex Result Storage**: Store final complex result with proper real/imaginary organization

## Source Code Organization

-   [`complex_contraction_bilinear_xdl.cpp`](./complex_contraction_bilinear_xdl.cpp): The main example file. It sets up complex tensors (with real and imaginary components), defines contraction patterns and bilinear operations, and instantiates the `DeviceComplexContractionBilinear` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_complex_contraction_bilinear.hpp`](../../include/ck/tensor_operation/gpu/device/device_complex_contraction_bilinear.hpp): The device interface for complex tensor operations with bilinear fusion.
-   The underlying kernel implements sophisticated complex arithmetic with optimized memory layouts for real/imaginary components.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/66_complex_contraction_bilinear
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
#arg3: time kernel (0=no, 1=yes)
./bin/example_contraction_bilinear_xdl_fp32 1 1 1
```

## Applications

Complex tensor operations with bilinear transformations are essential in several advanced domains:

-   **Quantum Computing**: Quantum circuit simulations require complex tensor contractions for state evolution and gate operations
-   **Signal Processing**: Digital signal processing with complex-valued signals, such as in communications and radar systems
-   **Fourier Analysis**: FFT-related computations that naturally involve complex arithmetic and tensor operations
-   **Quantum Chemistry**: Electronic structure calculations often involve complex-valued wavefunctions and operators
-   **Machine Learning**: Some advanced neural network architectures use complex-valued weights and activations
-   **Scientific Computing**: Simulations involving wave equations, electromagnetic fields, or quantum mechanical systems

## Complex Arithmetic Considerations

Working with complex numbers introduces several computational challenges:

-   **Memory Layout**: Efficient storage of real and imaginary components (interleaved vs. separate arrays)
-   **Arithmetic Complexity**: Complex multiplication requires 4 real multiplications and 2 real additions
-   **Numerical Precision**: Maintaining accuracy across multiple complex operations
-   **Performance Trade-offs**: Balancing between computational complexity and memory bandwidth

## Performance Characteristics

Complex operations have unique performance profiles:

-   **Computational Intensity**: ~2× the arithmetic operations compared to real-valued equivalents
-   **Memory Bandwidth**: 2× the memory requirements for storing complex values
-   **Register Pressure**: Higher register usage due to separate real/imaginary components
-   **Instruction Complexity**: More complex instruction sequences for complex arithmetic

This kernel demonstrates the ability to handle sophisticated mathematical operations efficiently while maintaining the benefits of deep fusion for complex-valued computations.
