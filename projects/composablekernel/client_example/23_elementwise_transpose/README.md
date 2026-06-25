# Client Example: Elementwise Operation with 3D Transpose

## Theory

This client example demonstrates **elementwise operations fused with 3D tensor transpose**. This pattern is used in deep learning for applying activation functions or scaling while simultaneously reordering tensor dimensions (e.g., for layout conversion or attention head reshaping).

**Mathematical Formulation:**
- Elementwise: $Z = f(X)$ or $Z = f(X, Y)$
- Transpose: $Y_{i_0, i_1, i_2} = Z_{i_{\pi(0)}, i_{\pi(1)}, i_{\pi(2)}}$
  - $\pi$ is a permutation of the axes.

**Algorithmic Background:**
- The elementwise operation and transpose are fused in a single kernel.
- Intermediate results are kept in registers, not written to global memory.
- Used for layout conversion with activation, attention head reshaping, and more.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/23_elementwise_transpose
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (elementwise + 3D transpose)
./elementwise_transpose_3d
```

## Source Code Structure

### Directory Layout
```
client_example/23_elementwise_transpose/
├── elementwise_transpose_3d.cpp         # Main client example: elementwise + 3D transpose
├── CMakeLists.txt                       # Build configuration for the example
```

### Key Functions

- **main()** (in `elementwise_transpose_3d.cpp`):  
  Sets up input tensors, configures elementwise and transpose parameters, launches the fused kernel, and verifies the result.
- **Fused kernel invocation**:  
  Uses the Composable Kernel device API to launch the elementwise+transpose operation.

---

## Additional Details

- Supports fusion of elementwise operations with 3D transpose.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [44_elementwise_permute](../../example/44_elementwise_permute/README.md): Elementwise operation with permutation in the main example directory

---
[Back to Client Examples](../README.md)
