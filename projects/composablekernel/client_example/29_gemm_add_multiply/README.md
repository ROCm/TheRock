# Client Example: GEMM with Add and Multiply Fusion

## Theory

This client example demonstrates **GEMM fused with addition and multiplication operations**. This pattern is used in neural networks for bias addition, scaling, gating, and other elementwise transformations after a linear layer.

**Mathematical Formulation:**
- GEMM: $Y = A \times B$
- Add: $Z = Y + D_0$
- Multiply: $E = Z \odot D_1$
  - $D_0$, $D_1$: auxiliary tensors (e.g., bias, scale, gate)

**Algorithmic Background:**
- The GEMM result is kept in registers, addition and multiplication are fused in the epilogue.
- No intermediate results are written to global memory.
- Used for bias+scale, gating, and other fused epilogue patterns.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/29_gemm_add_multiply
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm_add_multiply
```

## Source Code Structure

### Directory Layout
```
client_example/29_gemm_add_multiply/
├── gemm_add_multiply.cpp         # Main client example: GEMM+Add+Multiply
├── CMakeLists.txt                # Build configuration for the example
```

### Key Functions

- **main()** (in `gemm_add_multiply.cpp`):  
  Sets up input matrices, configures GEMM and epilogue parameters, launches the fused kernel, and verifies the result.
- **Fused kernel invocation**:  
  Uses the Composable Kernel device API to launch the GEMM with fused addition and multiplication.

---

## Additional Details

- Supports fusion of multiple elementwise operations with GEMM.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [02_gemm_bilinear](../../example/02_gemm_bilinear/README.md): Multi-tensor bilinear operations
- [46_gemm_add_multiply](../../example/46_gemm_add_multiply/README.md): GEMM with add and multiply in the main example directory

---
[Back to Client Examples](../README.md)
