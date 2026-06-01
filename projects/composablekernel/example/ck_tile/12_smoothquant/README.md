# SmoothQuant with CK Tile

This example demonstrates SmoothQuant, a quantization technique for transformer models, using the CK Tile programming model. SmoothQuant enables efficient int8 inference by scaling activations and weights to balance quantization error.

---

## Algorithm and Math

Given input $X$ and per-channel scale $S$:
1. **Scale**: $Y_{i,j} = X_{i,j} \cdot S_j$
2. **Rowwise Dynamic Quantization**:
   - For each row, $s = \max(|Y|) / 127$
   - $Q_{i,j} = \text{round}(Y_{i,j} / s)$, $Q_{i,j} \in \text{int8}$

**Output**:  
- Quantized tensor $Q$ (int8)
- Per-row scale $s$ (fp32)

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (row or block).
- **Pipeline**: Modular, can be extended for further fusion.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_smoothquant -j`nproc`
```
This will result in an executable `build/bin/tile_smoothquant`

## cmdline
```
args:
          -m    m dimension (default:3328)
          -n    n dimension (default:4096)
   -x_stride    input stride per row, if -1 then equal to n (default:-1)
   -y_stride    output stride per row, if -1 then equal to n (default:-1)
          -v    cpu validation or not (default:1)
      -kname    print kernel name or not (default:1)
       -prec    precision (default:fp16)
     -warmup    cold iter (default:5)
     -repeat    hot iter (default:20)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:smoothquant.json)
```
