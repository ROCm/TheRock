# Client Example: Grouped GEMM with FastGELU Activation

## Theory

This client example demonstrates **grouped GEMM fused with FastGELU activation**. Grouped GEMM performs multiple independent GEMM operations (with potentially different shapes) in a single kernel launch, and FastGELU is a fast approximation of the GELU activation used in transformers and MLPs.

**Mathematical Formulation:**
For $G$ groups, each with its own $A_g$, $B_g$:
- GEMM: $Y_g = A_g \times B_g$
- FastGELU: $E_g = \text{FastGELU}(Y_g)$

FastGELU is defined as:
$$
\text{FastGELU}(x) = x \cdot \sigma(1.702 \cdot x)
$$
where $\sigma$ is the sigmoid function.

**Algorithmic Background:**
- Each group can have different matrix sizes and strides.
- The kernel launches a grid covering all groups, with each block assigned to a group.
- FastGELU is applied in the epilogue for each group.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/17_grouped_gemm_fastgelu
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./grouped_gemm_fastgelu
```

## Source Code Structure

### Directory Layout
```
client_example/17_grouped_gemm_fastgelu/
├── grouped_gemm_fastgelu.cpp         # Main client example: grouped GEMM + FastGELU
├── CMakeLists.txt                    # Build configuration for the example
```

### Key Functions

- **main()** (in `grouped_gemm_fastgelu.cpp`):  
  Sets up input matrices for each group, configures GEMM and epilogue parameters, launches the grouped kernel, and verifies the result.
- **Grouped GEMM kernel invocation**:  
  Uses the Composable Kernel device API to launch grouped GEMM with FastGELU activation.

---

## Additional Details

- Supports multiple groups with different matrix shapes.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [15_grouped_gemm](../../example/15_grouped_gemm/README.md): Grouped GEMM in the main example directory
- [04_gemm_add_add_fastgelu](../../example/04_gemm_add_add_fastgelu/README.md): GEMM with FastGELU fusion

---
[Back to Client Examples](../README.md)
