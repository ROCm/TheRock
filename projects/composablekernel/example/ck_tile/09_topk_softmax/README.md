# TopK-Softmax with CK Tile

This example demonstrates a tile-programming implementation of TopK-Softmax, commonly used in Mixture-of-Experts (MoE) models to select top-k experts per token after softmax.  This kernel is often used in MoE model, before launching the fused-moe-gemm block. The input is a `token*expert` 2d matrix. The op will do a softmax per row(`expert`), then find the `topk` value for each row. Output is a `token*topk` weight (typically fp32) and index(int32) 2D tensor.

---

## Algorithm and Math

Given a matrix $X$ of shape $[\text{tokens}, \text{experts}]$:
1. **Softmax per row**: $S_{i,j} = \frac{\exp(X_{i,j})}{\sum_k \exp(X_{i,k})}$
2. **TopK selection**: For each row $i$, select the $k$ largest $S_{i,j}$ and their indices.

**Output**:  
- $[\text{tokens}, k]$ weights (fp32)
- $[\text{tokens}, k]$ indices (int32)

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (block of rows).
- **Pipeline**: Modular, can be extended for fused operations.

---

## Build & Run

```bash
# in the root of ck_tile
mkdir build && cd build
../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_topk_softmax -j
```
This will result in an executable `build/bin/tile_example_topk_softmax`

### Arguments

```bash
args:
          -v    weather do CPU validation or not (default:1)
       -pr_i    input data type. fp16/fp32 (representing 8/16/32 bit data) (default:fp16)
       -pr_w    output weight data type(currently only fp32 supported now) (default:fp32)
          -t    number of input tokens (default:32)
          -e    number of experts (default:8)
          -k    topk (default:2)
       -st_i    row stride of input, -1 means same as experts (default:-1)
       -st_o    row stride of output/indices, -1 means same as topk (default:-1)
       -seed    seed to be used, -1 means random every time (default:-1)
      -kname    when set to 1 it will print kernel name (default:0)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:topk_softmax.json)

```

---

## Source Structure

- **Kernel**: [`topk_softmax_api.hpp`](topk_softmax_api.hpp) (tile-programming kernel template)
- **Executable**: [`topk_softmax.cpp`](topk_softmax.cpp) (argument parsing, kernel launch)
- **Build**: `CMakeLists.txt`, `script/`

---

## Related CK Tile Examples

- [15_fused_moe](../15_fused_moe/README.md): Fused MoE block using TopK-Softmax
- [05_reduce](../05_reduce/README.md): Reductions with tiles
- [03_gemm](../03_gemm/README.md): GEMM with tiles

For distribution, see [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
