# MoE Sorting with CK Tile

This example demonstrates MoE (Mixture-of-Experts) sorting using the CK Tile programming model. MoE sorting rearranges token-to-expert assignments for efficient dispatch to expert GEMMs, a key step in large language models with MoE layers. This kernel is often used in Moe model, before launching the fused-moe-gemm block. The input&weight is a `token*topk` 2d matrix. The op rearange the input weight ids into different experts and feed into fuse moe gemm kernel.

---

## Algorithm and Math

Given:
- **Input**: $[\text{tokens}, \text{topk}]$ indices and weights (from TopK-Softmax)
- **Goal**: Rearrange tokens so each expert receives its assigned tokens in contiguous blocks

**Steps:**
1. For each token, for each of its top-k experts, assign the token to the expert's input buffer.
2. Output:
   - Expert-wise token lists (indices)
   - Corresponding weights

This enables efficient batched GEMM per expert.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (block of tokens or experts).
- **Pipeline**: Modular, can be extended for further fusion or dispatch.

---

## Build & Run

```bash
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_moe_sorting -j`nproc`
```

This will result in an executable `build/bin/tile_example_moe_sorting`.

### Arguments
```
args:
                 -v    turn CPU validation on (1) or off (0). (default:1)
              -pr_i    index data type.  Only int32 is currently supported. (default:int32)
              -pr_w    output weight data type. Only fp32 is currently supported. (default:fp32)
                 -t    number of input tokens. (default:128)
                       If "local_t" presents, this value indicates global concurrency of all ranks.
           -local_t    Number of local input tokens for curent rank. (default:-1)
                       This value must be within range "[0, t)", or "-1"(no such feature)
                       This feature is to simulate EP case where where each rank has different tokens.
                       Besides, this value will be stored in a GPU buffer, which is friendly for CUDA graph.
                 -e    number of num_experts (default:8)
                 -k    topk (default:4)
              -unit    unit_size (default:32)
-moe_buf_interm_dim    interm_dim(col) of the following fmoe buf (default:0)
-moe_buf_elem_bytes    fmoe buf element byte size, 1:8bit, 2:16bit, 4:32bit... (default:2)
                -ci    clear workspace inside API or not(if "0", require manually clear outside) (default:1)
          -dispatch    dispatch policy. 0:automatically pick up kernel, 1:use single kernel, 2:use mp kernel (default:0)
         -local_eid    a list of experts enabled as local expert. e.g. "0,1,4,5" (default:-1)
                       please make sure eid is in ascending order!
              -seed    seed to be used. When set to -1, a random seed will be generated each time invoking this example (default:-1)
             -kname    prints the kernel name when set to 1 (default:0)
            -warmup    number of iterations before benchmark the kernel (default:5)
            -repeat    number of iterations to benchmark the kernel (default:20)
              -json    0: No Json, 1: Dump Results in Json format (default:0)
          -jsonfile    json file name to dump results (default:moe_sorting.json)
```

---

## Source Structure

- **Kernel**: [`moe_sorting_api.hpp`](moe_sorting_api.hpp) (tile-programming kernel template)
- **Executable**: [`moe_sorting.cpp`](moe_sorting.cpp), [`moe_sorting_api.cpp`](moe_sorting_api.cpp)
- **Build**: `CMakeLists.txt`, `script/`

---

## Related CK Tile Examples

- [09_topk_softmax](../09_topk_softmax/README.md): TopK-Softmax for MoE gating
- [15_fused_moe](../15_fused_moe/README.md): Fused MoE block
- [03_gemm](../03_gemm/README.md): GEMM with tiles

For distribution, see [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
