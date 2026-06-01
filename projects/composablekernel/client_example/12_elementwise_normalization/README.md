# Client Example: Elementwise Layer Normalization

## Theory

This client example demonstrates **elementwise layer normalization** for 2D tensors. Layer normalization is used in transformers and other neural networks to normalize activations across the feature dimension, improving training stability. Elementwise normalization fuses normalization with other elementwise operations for efficiency.

**Mathematical Formulation:**
Given input $X$:
- Mean: $\mu = \frac{1}{N} \sum_{i=1}^N X_i$
- Variance: $\sigma^2 = \frac{1}{N} \sum_{i=1}^N (X_i - \mu)^2$
- Normalized: $\hat{X}_i = \frac{X_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
- Output: $Y_i = \gamma \hat{X}_i + \beta$

$\gamma$, $\beta$ are learnable scale and shift parameters.

**Algorithmic Background:**
- Computes mean and variance per row (sample).
- Applies normalization and affine transformation.
- Can be fused with other elementwise operations for efficiency.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/12_elementwise_normalization
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./elementwise_layernorm2d
```

## Source Code Structure

### Directory Layout
```
client_example/12_elementwise_normalization/
├── elementwise_layernorm2d.cpp         # Main client example: elementwise layernorm for 2D tensors
├── CMakeLists.txt                      # Build configuration for the example
```

### Key Functions

- **main()** (in `elementwise_layernorm2d.cpp`):  
  Sets up input tensors, configures normalization parameters, launches the normalization kernel, and verifies the result.
- **Elementwise normalization kernel invocation**:  
  Uses the Composable Kernel device API to launch layer normalization, optionally fused with other elementwise ops.

---

## Additional Details

- Supports fusion with other elementwise operations for efficiency.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [05_layernorm](../05_layernorm/README.md): Layer normalization client API
- [27_layernorm2d_fwd](../../example/27_layernorm2d_fwd/README.md): Layer normalization in the main example directory

---
[Back to Client Examples](../README.md)
