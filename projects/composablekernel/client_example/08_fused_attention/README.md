# Fused Attention Examples

This directory contains comprehensive examples demonstrating CK's high-performance fused attention implementations, which are critical for modern transformer architectures and large language models.

---

## Theory

**Fused Multi-Head Attention Operation:**
The fused attention mechanism performs the core transformer operation in a single, optimized kernel:

$$
\text{Attention}(Q, K, V) = \text{Softmax}(Q K^T / \sqrt{d_k}) V
$$

**Detailed Mathematical Steps:**
1. **Query-Key Attention Scores**: $S = Q K^T$
2. **Scale**: $S_{\text{scaled}} = S / \sqrt{d_k}$
3. **Softmax**: $A = \text{Softmax}(S_{\text{scaled}})$
4. **Weighted Value Sum**: $\text{Output} = A V$

- Multi-head extension: Each head computes attention independently, then results are concatenated and projected.
- Tensor shapes: Q, K, V, Output are typically [Batch, Seq_len, Num_heads, Head_dim].

**Algorithmic Background:**
- Fused attention combines two GEMMs and a softmax in a single kernel, minimizing memory traffic.
- Supports bias, masking, and permutation for transformer and LLM workloads.

---

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/08_fused_attention
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (basic fused attention)
./fused_attention

# Example run (fused attention with bias)
./fused_attention_bias
```

---

## Source Code Structure

### Directory Layout
```
client_example/08_fused_attention/
├── fused_attention.cpp         # Main client example: fused attention (Q, K, V)
├── fused_attention_bias.cpp    # Fused attention with bias
├── CMakeLists.txt              # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up Q, K, V tensors, configures attention parameters, launches the fused kernel, and verifies the result.
- **Fused attention kernel invocation**:  
  Uses the Composable Kernel device API to launch the fused attention operation, optionally with bias.

---

## Additional Details

- Supports FP16, BF16, FP32, and mixed precision.
- Handles causal and generic masking for autoregressive and variable-length models.
- Optimized for memory efficiency (no intermediate attention matrix in global memory).
- Example parameters can be adjusted in the source for different transformer workloads.

---

## Related Examples

- [01_gemm](../01_gemm/README.md): GEMM for Q×K^T and Attn×V
- [06_softmax](../06_softmax/README.md): Softmax client API usage
- [03_gemm_layernorm](../03_gemm_layernorm/README.md): Fused GEMM + layer normalization
- [07_grouped_convnd_fwd](../07_grouped_convnd_fwd/README.md): Grouped convolution for vision transformers

---
[Back to Client Examples](../README.md)
