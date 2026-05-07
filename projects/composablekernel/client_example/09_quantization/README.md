# Client Example: Quantization for GEMM and Conv2D

## Theory

This client example demonstrates **quantized GEMM and 2D convolution** operations, including per-layer and per-channel quantization, and fusion with bias and activation functions. Quantization reduces memory and computation by representing values with lower-precision integer types (e.g., int8), enabling efficient inference in deep learning.

**Mathematical Formulation:**
- Quantized GEMM: $C = \text{dequant}(A_q) \times \text{dequant}(B_q)$
- Quantized Conv2D: $Y = \text{dequant}(X_q) * \text{dequant}(W_q)$
- $\text{dequant}(x_q) = (x_q - z) \cdot s$ (scale $s$, zero-point $z$)
- Per-layer: one scale/zero-point per tensor
- Per-channel: scale/zero-point per output channel

**Algorithmic Background:**
- Quantized values are dequantized on-the-fly during computation.
- Accumulation is performed in higher precision for accuracy.
- Supports bias addition and activation fusion (ReLU, Tanh).
- Per-channel quantization improves accuracy for convolutional layers.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/09_quantization
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (GEMM quantization)
./gemm_quantization

# Example run (Conv2D per-layer quantization)
./conv2d_fwd_perlayer_quantization

# Example run (Conv2D per-channel quantization)
./conv2d_fwd_perchannel_quantization

# Example run (Conv2D + bias + ReLU + per-channel quantization)
./conv2d_fwd_bias_relu_perchannel_quantization
```

## Source Code Structure

### Directory Layout
```
client_example/09_quantization/
├── gemm_quantization.cpp                         # Quantized GEMM
├── conv2d_fwd_perlayer_quantization.cpp          # Conv2D per-layer quantization
├── conv2d_fwd_perchannel_quantization.cpp        # Conv2D per-channel quantization
├── conv2d_fwd_bias_relu_perlayer_quantization.cpp # Conv2D + bias + ReLU + per-layer quantization
├── conv2d_fwd_bias_relu_perchannel_quantization.cpp # Conv2D + bias + ReLU + per-channel quantization
├── conv2d_fwd_bias_tanh_perlayer_quantization.cpp # Conv2D + bias + Tanh + per-layer quantization
├── conv2d_fwd_bias_tanh_perchannel_quantization.cpp # Conv2D + bias + Tanh + per-channel quantization
├── CMakeLists.txt                                # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures quantization parameters, launches the quantized kernel, and verifies the result.
- **Quantization kernel invocation**:  
  Uses the Composable Kernel device API to launch quantized GEMM or Conv2D with optional bias and activation.

---

## Additional Details

- Supports int8 quantization, per-layer and per-channel scaling.
- Demonstrates fusion with bias and activation (ReLU, Tanh).
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [01_gemm](../01_gemm/README.md): GEMM for quantized matrix multiplication
- [14_gemm_quantization](../../example/14_gemm_quantization/README.md): GEMM quantization in the main example directory
- [40_conv2d_fwd_quantization](../../example/40_conv2d_fwd_quantization/README.md): Conv2D quantization in the main example directory

---
[Back to Client Examples](../README.md)
