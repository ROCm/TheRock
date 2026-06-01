# Tensor Contraction with Multiple A, B, and D Tensors

This example demonstrates a **tensor contraction operation with multiple A, B, and D tensors**. This extends the basic tensor contraction to handle multiple input tensor pairs and auxiliary tensors simultaneously, enabling complex multi-input tensor network computations to be executed in a single kernel launch.

## Mathematical Formulation

This operation performs multiple tensor contractions simultaneously and combines them with auxiliary tensors.

1.  **Multiple Tensor Contractions**: Compute contractions from multiple A and B tensor pairs using Einstein summation notation.
    $C_{temp0} = \text{einsum}(\text{pattern}_0, A_0, B_0)$
    $C_{temp1} = \text{einsum}(\text{pattern}_1, A_1, B_1)$
    $\vdots$
    $C_{tempK} = \text{einsum}(\text{pattern}_K, A_K, B_K)$

2.  **Combination with Auxiliary Tensors**: Apply a user-defined function that combines all contraction results with multiple D tensors.
    $E = f(C_{temp0}, C_{temp1}, \ldots, C_{tempK}, D_0, D_1, \ldots, D_M)$

Each contraction can have different Einstein summation patterns, allowing for complex tensor network computations. The key optimization is that all intermediate tensors are **never written to global memory**.

## Algorithmic Strategy: Multi-Input Contraction with Tensor-to-GEMM Mapping

This kernel extends the tensor contraction algorithm to handle multiple simultaneous contractions.

1.  **Unified Tensor-to-GEMM Mapping**: Each tensor contraction is mapped to a GEMM operation through tensor reshaping:
    -   **Multiple Reshaping Operations**: For each contraction pair `(A_i, B_i)`, the tensors are logically reshaped into 2D matrices based on their Einstein summation pattern.
    -   **Coordinated Memory Layout**: The reshaping operations are coordinated to enable efficient memory access patterns across all contractions.

2.  **Multi-Contraction Tile Computation**: Within each thread block:
    -   **Parallel GEMM Execution**: Multiple GEMM operations (representing the contractions) are computed simultaneously.
    -   **Complex Address Calculation**: Each contraction requires its own address calculation logic for the tensor descriptor interpretation.
    -   **Register Management**: Multiple accumulator arrays are maintained for the different contraction results.

3.  **Tensor Fusion Epilogue**: After computing all contractions:
    -   **Multi-Tensor Reshape**: The GEMM results are logically reshaped back to their target tensor shapes.
    -   **Load Auxiliary Tensors**: Read corresponding elements from all D tensors.
    -   **Apply Fusion Function**: Execute the user-defined function `f` combining all results.
    -   **Store Final Tensor**: Write the combined result to the output tensor.

## Source Code Organization

-   [`contraction_multi_ABD_xdl.cpp`](./contraction_multi_ABD_xdl.cpp): The main example file. It sets up multiple pairs of tensors for contraction, defines the Einstein summation patterns, sets up auxiliary D tensors, and instantiates the `DeviceContractionMultiABD` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_contraction_multi_abd.hpp`](../../include/ck/tensor_operation/gpu/device/device_contraction_multi_abd.hpp): The device interface for this multi-contraction fusion pattern.
-   The underlying kernel manages multiple simultaneous tensor contractions with complex tensor descriptor logic and memory access patterns.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/61_contraction_multi_ABD
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
./contraction_multi_ABD_xdl

# Run with verification, data initialization, and timing
./contraction_multi_ABD_xdl 1 2 1
```

## Applications

This kernel is valuable for complex tensor network computations found in advanced scientific and machine learning applications.

-   **Tensor Network Methods**: Computing multiple tensor contractions simultaneously in quantum physics simulations, such as DMRG (Density Matrix Renormalization Group) or PEPS (Projected Entangled Pair States).
-   **Multi-Modal Tensor Analysis**: Processing multiple tensor contractions for different data modalities in machine learning applications.
-   **Higher-Order Statistics**: Computing multiple statistical tensor operations simultaneously, such as different moments or correlation patterns.
-   **Advanced Neural Network Layers**: Implementing complex layers that require multiple tensor operations, such as tensor decomposition layers or high-dimensional convolutions.
-   **Scientific Computing**: Simulating physical systems that require multiple tensor contractions, such as in quantum chemistry or condensed matter physics.

## Computational Complexity

The complexity depends on the specific contraction patterns used:

-   **Multiple Contractions**: Each contraction has its own complexity based on tensor dimensions and contraction indices
-   **Memory Access**: Complex patterns due to multiple tensor descriptors and reshaping operations
-   **Register Pressure**: High due to multiple accumulator arrays and intermediate results
-   **Instruction Diversity**: Different contractions may have different computational patterns

## Comparison with Single Contraction

| Aspect | Single Contraction | Multi-Contraction |
|--------|-------------------|-------------------|
| **Input Complexity** | Single tensor pair | Multiple tensor pairs |
| **Memory Layout** | Single reshaping pattern | Multiple coordinated patterns |
| **Computation** | Single GEMM operation | Multiple parallel GEMMs |
| **Fusion Opportunity** | Simple epilogue | Complex multi-input epilogue |
| **Applications** | Basic tensor operations | Complex tensor networks |

This kernel showcases the ability to handle extremely complex tensor network computations efficiently, making it valuable for advanced scientific computing and machine learning research applications.
