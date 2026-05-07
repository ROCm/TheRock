# Client Example: General Tensor Contraction

## Theory

This client example demonstrates **general tensor contraction** operations, including bilinear and scaled contractions. Tensor contraction generalizes matrix multiplication to higher dimensions and is used in scientific computing, quantum chemistry, and advanced neural network layers.

**Mathematical Formulation:**
- General contraction: $C_{i,j} = \sum_k A_{i,k} \cdot B_{k,j}$
- Bilinear contraction: $C = \alpha (A \cdot B) + \beta D$
- Scale contraction: $C = \text{scale}(A, B)$ (elementwise or broadcasted scaling)

**Algorithmic Background:**
- Contraction can be performed over arbitrary axes and supports broadcasting.
- Bilinear and scale contractions are used for feature fusion, gating, and scientific workloads.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/04_contraction
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (bilinear FP32)
./contraction_bilinear_fp32

# Example run (scale FP64)
./contraction_scale_fp64
```

## Source Code Structure

### Directory Layout
```
client_example/04_contraction/
├── contraction_bilinear_fp32.cpp         # Bilinear contraction (FP32)
├── contraction_bilinear_fp64.cpp         # Bilinear contraction (FP64)
├── contraction_g1m2n3k1_add_xdl_fp16.cpp # Grouped contraction with addition (FP16)
├── contraction_scale_fp32.cpp            # Scale contraction (FP32)
├── contraction_scale_fp64.cpp            # Scale contraction (FP64)
├── CMakeLists.txt                        # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures contraction parameters, launches the contraction kernel, and verifies the result.
- **Contraction kernel invocation**:  
  Uses the Composable Kernel device API to launch the contraction operation.

This client example provides several variants to demonstrate different contraction types and data types for scientific and ML workloads.
