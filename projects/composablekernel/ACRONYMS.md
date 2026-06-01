# Acronyms in Composable Kernel

The following acronyms are used in the Composable Kernel codebase:

| Acronym | Expansion | Explanation |
|---------|-----------|-------------|
| BF16    | Brain Floating Point 16 | 1 Signed bit, 8 Exponent bits, 7 Significand bits |
| BF8     | 8-bit Brain Floating Point | 1 Signed bit, 3 Exponent bits, 4 Significand bits |
| DLA     | Deep Learning Accelerator | Specialized hardware for deep learning workloads |
| DRAM    | Dynamic Random-Access Memory | Main memory. Global memory on GPU |
| E2E     | End-to-End | Complete pipeline or process from input to output |
| ELU     | Exponential Linear Unit | Activation function: $x$ if $x>0$ else $\alpha(e^x-1)$ |
| FMHA    | Fused Multi-Head Attention | Efficient transformer attention kernel, fusing softmax, masking, and matmul |
| FP16    | Half-Precision Floating Point | 16-bit IEEE floating point format |
| FP32    | Single-Precision Floating Point | 32-bit IEEE floating point format |
| FP64    | Double-Precision Floating Point | 64-bit IEEE floating point format |
| FP8     | 8-bit Floating Point | Experimental 8-bit floating point format for inference |
| GEMM    | General Matrix Multiply | Matrix multiplication operation: $C = A \times B$ |
| GELU    | Gaussian Error Linear Unit | Activation function: $x \cdot \Phi(x)$ |
| GQA     | Grouped Query Attention | Variant of multi-head attention with grouped queries/keys/values |
| HBM     | High Bandwidth Memory | Fast memory used in modern GPUs |
| HIP     | Heterogeneous-Compute Interface for Portability | AMD's CUDA-like GPU programming API |
| INT8    | 8-bit Integer | Quantized integer format for inference |
| KVS     | Key-Value Store | Data structure for storing key-value pairs (context: QKV in transformers) |
| L2/L1   | Level 2/Level 1 Cache | On-chip memory hierarchy in CPUs/GPUs |
| LDS     | Local Data Share | Shared memory on AMD GPUs (equivalent to CUDA's shared memory) |
| LLM     | Large Language Model | Transformer-based model for NLP tasks |
| LSE     | Log-Sum-Exp | Numerically stable softmax computation: $\log(\sum \exp(x))$ |
| MHA     | Multi-Head Attention | Attention mechanism with multiple heads in transformers |
| MFMA    | Matrix Fused Multiply-Add | AMD GPU hardware instruction for matrix-matrix multiplication |
| MoE     | Mixture of Experts | Neural network architecture with multiple expert subnetworks |
| MQA     | Multi-Query Attention | Variant of multi-head attention with shared keys/values across heads |
| RCCL    | ROCm Collective Communications Library | AMD Library for multi-GPU communication |
| NCHW    | Batch, Channel, Height, Width | Tensor layout: batch-major, channels-first |
| NHWC    | Batch, Height, Width, Channel | Tensor layout: batch-major, channels-last |
| OOM     | Out Of Memory | Error when memory allocation fails |
| QAT     | Quantization Aware Training | Training technique for quantized inference |
| QKV     | Query, Key, Value | Components of transformer attention mechanism |
| RDMA    | Remote Direct Memory Access | High-speed network memory access |
| RDQuant | Rowwise Dynamic Quantization | Quantization technique with per-row scaling for int8 inference |
| ReLU    | Rectified Linear Unit | Activation function: $\max(0, x)$ |
| ROCm    | Radeon Open Compute | AMD's open GPU computing stack |
| SGD     | Stochastic Gradient Descent | Optimization algorithm for training neural networks |
| SM      | Streaming Multiprocessor | GPU compute unit (NVIDIA terminology) |
| SWA     | Sliding Window Attention | Attention mechanism with a limited window for each token |
| TLB     | Translation Lookaside Buffer | Memory management unit cache for virtual-to-physical address translation |
| VGPR    | Vector General Purpose Register | GPU register for vector operations |
| WARP    | Group of Threads | Smallest scheduling unit on NVIDIA GPUs (32 threads) |
| WMMA    | Warp Matrix Multiply-Accumulate | NVIDIA's matrix-multiply hardware primitive |
| XLA     | Accelerated Linear Algebra | Compiler for optimizing ML computations (Google) |

### Common Variable Acronyms in Code

| Symbol | Meaning | Context |
|--------|---------|---------|
| M, N, K | Matrix dimensions | GEMM: $A[M,K] \times B[K,N] = C[M,N]$ |
| Q, K, V | Query, Key, Value | Transformer attention |
| S       | Sequence length | NLP, transformers |
| D       | Dimension | Hidden size, feature dim |
| B       | Batch size | ML batch processing |
| H       | Head count | Multi-head attention |
| C       | Channel | CNNs, tensor layouts |
| T       | Token | NLP, sequence models |

---

If you find an acronym not listed here, please submit a pull request or issue!
