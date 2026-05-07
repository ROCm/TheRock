# Add + RMSNorm2D + Rowwise Dynamic Quantization (RDQuant) with CK Tile

This example demonstrates a fused kernel for elementwise addition, 2D RMSNorm, and rowwise dynamic quantization using the CK Tile programming model. This pattern is common in LLMs for efficient normalization and quantized inference.

---

## Algorithm and Math

Given input $X$ and residual $R$:
1. **Elementwise Add**: $Z = X + R$
2. **RMSNorm**: $\text{rms}(Z) = \sqrt{\frac{1}{N} \sum_{i=1}^N Z_i^2 + \epsilon}$, $Y_i = \frac{Z_i}{\text{rms}(Z)} \cdot \gamma_i$
3. **Rowwise Dynamic Quantization**:
   - For each row, $s = \max(|Y|) / 127$
   - $Q_i = \text{round}(Y_i / s)$, $Q_i \in \text{int8}$

**Output**:  
- Quantized tensor $Q$ (int8)
- Per-row scale $s$ (fp32)

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (row or block).
- **Tile Engine**: Loads tiles, performs add, RMSNorm, and quantization.
- **Pipeline**: Modular, can be extended for further fusion.

---

## Build & Run

```bash
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_add_rmsnorm2d_rdquant_fwd -j`nproc`
```
This will result in an executable `build/bin/tile_add_rmsnorm2d_rdquant_fwd`

### Arguments

```bash
args:
          -m    m dimension (default:3328)
          -n    n dimension (default:4096)
     -stride    stride per row, if -1 then equal to n (default:-1)
          -e    epsilon (default:1e-5)
     -save_x    save rms(invrms) or not. set to 1 in training case (default:1)
          -v    cpu validation or not (default:1)
      -kname    print kernel name or not (default:1)
       -prec    precision (default:fp16)
      -quant    precision (default:int8)
     -warmup    cold iter (default:5)
     -repeat    hot iter (default:20)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:add_rmsnorm2d_rdquant_fwd.json)
```

---

## Source Structure

- **Kernel**: [`add_rmsnorm2d_rdquant_fwd.hpp`](add_rmsnorm2d_rdquant_fwd.hpp) (tile-programming kernel template)
- **Executable**: [`add_rmsnorm2d_rdquant_fwd.cpp`](add_rmsnorm2d_rdquant_fwd.cpp), [`example_add_rmsnorm2d_rdquant_fwd.cpp`](example_add_rmsnorm2d_rdquant_fwd.cpp)
- **Build**: `CMakeLists.txt`, `instances/`, `script/`

---

## Related CK Tile Examples

- [10_rmsnorm2d](../10_rmsnorm2d/README.md): RMSNorm2D with tiles
- [12_smoothquant](../12_smoothquant/README.md): SmoothQuant with tiles
- [02_layernorm2d](../02_layernorm2d/README.md): LayerNorm2D with tiles

For distribution, see [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
