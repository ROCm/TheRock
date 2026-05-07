# RMSNorm2D Forward with CK Tile

This example demonstrates 2D Root Mean Square Layer Normalization (RMSNorm) using the CK Tile programming model, a normalization technique widely used in LLMs and transformers.

---

## Algorithm and Math

For each row $x$:
$$
\text{rms}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2 + \epsilon}
$$
$$
y_i = \frac{x_i}{\text{rms}(x)} \cdot \gamma_i
$$
where $\gamma$ is a learnable scale parameter.

- **Tilewise RMSNorm**: Each thread block processes a tile (row or block), computes the mean square, normalizes, and applies scale.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input matrix.
- **Pipeline**: Modular, can be extended for fused operations.

---

## Build & Run

```bash
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_rmsnorm2d_fwd -j`nproc`
```

This will result in an executable `build/bin/tile_rmsnorm2d_fwd`

### Arguments

```bash
args:
           -m    m dimension (default:3328)
           -n    n dimension (default:4096)
    -x_stride    x row_stride, if -1 then equal to n (default:-1)
   -xr_stride    x residule row_stride, if -1 then equal to n (default:-1)
    -y_stride    y row_stride, if -1 then equal to n (default:-1)
   -yr_stride    y residule row_stride, if -1 then equal to n (default:-1)
           -e    epsilon (default:1e-5)
    -save_rms    save rms(invrms) or not. set to 1 in training case (default:0)
-save_unquant    save result before quant (default:0)
           -v    cpu validation or not (default:1)
       -kname    print kernel name or not (default:1)
      -prec_i    input precision (default:fp16)
      -prec_o    output precision, set auto will be the same as input (default:auto)
     -prec_sm    output quant scale type, set auto will use fp32. used when fquant=1 (default:auto)
     -prec_sy    output quant scale type, set auto will use fp32. used when fquant=1 or 2 (default:auto)
        -fadd    fused-add, 0:no fused add, 1:preadd+store, 2:preadd only (default:0)
      -fquant    fused-quant, 0:no, 1:smooth-dynamic-quant, 2:dynamic-quant (default:0)
      -warmup    cold iter (default:5)
      -repeat    hot iter (default:20)
           -s    sensitive model mode, 0: for no specific model, 1: for T5-like model (default:0)
        -json    0: No Json, 1: Dump Results in Json format (default:0)
    -jsonfile    json file name to dump results (default:rmsnorm2d_fwd.json)
```

---

## Source Structure

- **Kernel**: [`rmsnorm2d_fwd.hpp`](rmsnorm2d_fwd.hpp) (tile-programming kernel template)
- **Executable**: [`rmsnorm2d_fwd.cpp`](rmsnorm2d_fwd.cpp) (argument parsing, kernel launch)
- **Build**: `CMakeLists.txt`, `generate.py`, `script/`

---

## Related CK Tile Examples

- [02_layernorm2d](../02_layernorm2d/README.md): LayerNorm2D with tiles
- [12_smoothquant](../12_smoothquant/README.md): SmoothQuant with tiles
- [05_reduce](../05_reduce/README.md): Reductions with tiles

For distribution, see  [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
