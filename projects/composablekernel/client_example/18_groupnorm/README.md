# Client Example: Group Normalization (Forward and Backward)

## Theory

This client example demonstrates **group normalization** in both forward and backward modes, including fusion with Swish activation. Group normalization normalizes activations across groups of channels, improving training stability for small batch sizes or non-i.i.d. data.

**Mathematical Formulation:**
Given input $X[N, C, ...]$ divided into $G$ groups:
- For each group $g$:
  - Mean: $\mu_g = \frac{1}{|g|} \sum_{i \in g} X_i$
  - Variance: $\sigma^2_g = \frac{1}{|g|} \sum_{i \in g} (X_i - \mu_g)^2$
  - Normalized: $\hat{X}_i = \frac{X_i - \mu_g}{\sqrt{\sigma^2_g + \epsilon}}$
  - Output: $Y_i = \gamma \hat{X}_i + \beta$

$\gamma$, $\beta$ are learnable scale and shift parameters.

- Swish activation: $\text{Swish}(x) = x \cdot \sigma(x)$, where $\sigma$ is the sigmoid function.

**Algorithmic Background:**
- Forward pass computes mean, variance, normalization, and affine transformation per group.
- Backward pass computes gradients with respect to input, gamma, and beta.
- Swish activation can be fused with normalization for efficiency.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/18_groupnorm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (forward with Swish)
./groupnorm_swish_fwd

# Example run (backward, data)
./groupnorm_bwd_data

# Example run (backward, gamma/beta)
./groupnorm_bwd_gamma_beta
```

## Source Code Structure

### Directory Layout
```
client_example/18_groupnorm/
├── groupnorm_swish_fwd.cpp         # Groupnorm forward with Swish activation
├── groupnorm_bwd_data.cpp          # Groupnorm backward (data)
├── groupnorm_bwd_gamma_beta.cpp    # Groupnorm backward (gamma/beta)
├── CMakeLists.txt                  # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures groupnorm parameters, launches the forward or backward kernel, and verifies the result.
- **GroupNorm kernel invocation**:  
  Uses the Composable Kernel device API to launch group normalization for different modes.

---

## Additional Details

- Supports fusion with Swish activation for efficiency.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [42_groupnorm_fwd](../../example/42_groupnorm_fwd/README.md): Group normalization in the main example directory
- [54_groupnorm_bwd](../../example/54_groupnorm_bwd/README.md): Group normalization backward in the main example directory

---
[Back to Client Examples](../README.md)
