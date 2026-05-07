# CK Tile Example Suite

This directory contains a comprehensive suite of examples demonstrating the CK Tile programming model for high-performance GPU kernels. Each example illustrates a key deep learning or HPC operation, implemented using tile-based parallelism, modular pipelines, and data movement policy.

---

## What is CK Tile?

CK Tile is a composable GPU programming API that expresses kernels as a composition of "tiles"—rectangular blocks of computation and data movement. The pipeline & policy orchestrates data movement (global <-> LDS <-> registers), computation, and synchronization, enabling high efficiency and flexibility.

---

## Example Index

| Example | Operation | Description |
|---------|-----------|-------------|
| [01_fmha](01_fmha/README.md) | Fused Multi-Head Attention | Tile-based FMHA with masking, quantization, and epilogue fusion |
| [02_layernorm2d](02_layernorm2d/README.md) | LayerNorm2D | Blockwise layer normalization with fusion and quantization |
| [03_gemm](03_gemm/README.md) | GEMM | Matrix multiplication with tilewise parallelism |
| [04_img2col](04_img2col/README.md) | im2col | Image-to-column transformation for GEMM-based convolution |
| [05_reduce](05_reduce/README.md) | Reduction | Tilewise sum, max, mean reductions |
| [06_permute](06_permute/README.md) | Permute | Generic tensor permutation (up to rank-8) |
| [09_topk_softmax](09_topk_softmax/README.md) | TopK-Softmax | Rowwise softmax and top-k selection for MoE gating |
| [10_rmsnorm2d](10_rmsnorm2d/README.md) | RMSNorm2D | Root mean square normalization for LLMs |
| [11_add_rmsnorm2d_rdquant](11_add_rmsnorm2d_rdquant/README.md) | Add + RMSNorm2D + RDQuant | Fused add, RMSNorm, and rowwise dynamic quantization |
| [12_smoothquant](12_smoothquant/README.md) | SmoothQuant | Per-channel scaling and quantization for int8 inference |
| [13_moe_sorting](13_moe_sorting/README.md) | MoE Sorting | Token-to-expert rearrangement for MoE dispatch |
| [14_moe_smoothquant](14_moe_smoothquant/README.md) | MoE-SmoothQuant | Expert-dependent quantization fused with top-k selection |
| [15_fused_moe](15_fused_moe/README.md) | Fused MoE | End-to-end fused MoE block: sorting, group-GEMM, activation, weighting |
| [16_batched_gemm](16_batched_gemm/README.md) | Batched GEMM | Parallel computation of multiple GEMMs |
| [17_grouped_gemm](17_grouped_gemm/README.md) | Grouped GEMM | Multiple independent GEMMs with different shapes |
| [18_flatmm](18_flatmm/README.md) | FLATMM | Flattened matrix multiplication for packed layouts |
| [19_gemm_multi_d](19_gemm_multi_d/README.md) | Multi-D GEMM | GEMM with multiple side inputs (bias, residual, etc.) |
| [35_batched_transpose](35_batched_transpose/README.md) | Batched Transpose | NCHW <-> NHWC and other layout conversions |
| [36_copy](36_copy/README.md) | Copy | Minimal example for tile-based memory movement |
| [37_transpose](37_transpose/README.md) | Block Transpose | High-performance tiled transpose for large tensors |

---

## Technical Highlights

- **Tile Distribution**: See [`include/ck_tile/tile_program/tile_distribution/`](../../include/ck_tile/tile_program/tile_distribution/) for mapping tiles to thread blocks.
- **Block Tile Pipelines**: See [`include/ck_tile/tile_program/block_tile_pipeline/`](../../include/ck_tile/tile_program/block_tile_pipeline/) for memory/computation pipelines.
- **Policies and Utilities**: Many examples use custom policies for tile/block size and memory access.

---

## How to Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make -j
```
Each example produces its own executable in `build/bin/`.

---

## Learning and Extending

- **Start Simple**: Try [03_gemm](03_gemm/README.md) or [36_copy](36_copy/README.md) to learn tile basics.
- **Explore Fusion**: See [11_add_rmsnorm2d_rdquant](11_add_rmsnorm2d_rdquant/README.md), [15_fused_moe](15_fused_moe/README.md), or [14_moe_smoothquant](14_moe_smoothquant/README.md) for advanced fusion.
- **Experiment**: Modify tile sizes, layouts, or pipelines to explore performance and flexibility.

---

## References

- [CK Tile Programming API Documentation](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile)
- [Block Tile Pipeline Source](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/tile_program/block_tile_pipeline)
- [Tile Distribution Source](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/tile_program/tile_distribution)

---

[Back to Composable Kernel Examples](../README.md)
