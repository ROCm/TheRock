# MoE-SmoothQuant with CK Tile

This example demonstrates MoE-SmoothQuant, a fused quantization operation for Mixture-of-Experts (MoE) models, using the CK Tile programming model. Unlike standard SmoothQuant, the input scale is expert-dependent, and the operation is fused with top-k expert selection. Specifically, it quantizes the top-k experts' outputs for each token using their respective expert scales. The input scale is from different expert `[expert, hidden]`, and we need reuse the `topk-id` from previous `topk-softmax` and select the corresponding `expert` from current topk, and expand the output/per-token-scale by `topk`. 

This diagram depicts moe-smoothquant using ck_tile tile-programming implementation.
![](misc/moe-sm.png)

---

## Algorithm and Math

Given:
- **Input**: $X$ of shape $[\text{tokens}, \text{topk}, \text{hidden}]$
- **Expert scales**: $S$ of shape $[\text{experts}, \text{hidden}]$
- **TopK indices**: $I$ of shape $[\text{tokens}, \text{topk}]$

**Steps:**
1. For each token $t$ and its $k$ selected experts:
   - Select scale $S_{I_{t,k}, :}$ for the $k$-th expert.
   - Scale: $Y_{t,k,j} = X_{t,k,j} \cdot S_{I_{t,k}, j}$
2. **Rowwise Dynamic Quantization** (per token-expert pair):
   - $s_{t,k} = \max_j |Y_{t,k,j}| / 127$
   - $Q_{t,k,j} = \text{round}(Y_{t,k,j} / s_{t,k})$, $Q_{t,k,j} \in \text{int8}$

**Output**:  
- Quantized tensor $Q$ (int8)
- Per-token-expert scale $s$ (fp32)

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (block of tokens, experts, or hidden units).
- **Tile Engine**: Loads input, selects expert scales via top-k indices, applies scaling and quantization, and writes results.
- **Pipeline**: Modular, can be extended for further fusion.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_moe_smoothquant -j`nproc`
./bin/tile_example_moe_smoothquant -?
```

---

## Source Structure

- **Kernel**: [`moe_smoothquant.hpp`](moe_smoothquant.hpp) (tile-programming kernel template)
- **Executable**: [`moe_smoothquant.cpp`](moe_smoothquant.cpp)
- **Build**: `CMakeLists.txt`, `instances/`, `misc/`, `script/`

---

## Technical Notes

- **Expert-dependent scaling**: Each token's top-k experts use their own per-hidden-unit scale, requiring indirect indexing and efficient memory access.
- **Fused with top-k**: The kernel uses top-k indices from gating to select the correct expert scale for each token.
- **Rowwise quantization**: Each token-expert pair is quantized independently for maximum accuracy.

---

## Related CK Tile Examples

- [09_topk_softmax](../09_topk_softmax/README.md): TopK-Softmax for MoE gating
- [13_moe_sorting](../13_moe_sorting/README.md): MoE sorting for expert dispatch
- [12_smoothquant](../12_smoothquant/README.md): Standard SmoothQuant

For distribution, see [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)

## example
```
args:
          -t    tokens dimension (default:3328)
          -h    hidden_size dimension (default:4096)
          -e    experts (default:32)
          -k    topk (default:5)
     -stride    stride per row, if -1 then equal to hidden_size (default:-1)
          -v    cpu validation or not (default:1)
      -kname    print kernel name or not (default:1)
     -prec_i    input precision, fp16/bf16 (default:fp16)
     -prec_o    precision, int8/fp8 (default:int8)
     -warmup    cold iter (default:5)
     -repeat    hot iter (default:20)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:moe_smoothquant.json)
```