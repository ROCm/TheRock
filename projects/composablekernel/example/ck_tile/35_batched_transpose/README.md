# Batched Transpose with CK Tile

This example demonstrates batched tensor transpose using the CK Tile programming model. It supports common layout conversions such as NCHW <-> NHWC, which are essential for deep learning frameworks and hardware accelerators.  Currently, it supports the batched transpose with NCHW to NHWC or NHWC to NCHW. So in this way from NCHW you could transpose to either NHWC or NWCH(two transposes). Now the transpose read with single data point. We would soon put it in vectorized transpose.

---

## Algorithm and Math

Given a batch of tensors $X$ of shape $[N, C, H, W]$, the transpose operation rearranges axes to produce $Y$ of shape $[N, H, W, C]$ (NCHW to NHWC) or other permutations.

For each element:
$$
Y_{n, h, w, c} = X_{n, c, h, w}
$$

- **Tilewise Batched Transpose**: Each thread block processes a tile (block) of the input, computes the permuted indices, and writes to the output.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input tensor for a given batch.
- **Pipeline**: Modular, can be extended for vectorized or fused operations.

---

## Features

- **Flexible Layouts**: Supports NCHW <-> NHWC and other axis permutations.
- **Batching**: Efficiently transposes multiple tensors in parallel.
- **Validation**: CPU validation and benchmarking options.

---

## Build & Run

```bash
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# Make the transpose executable
make tile_example_batched_transpose -j
```

This will result in an executable `build/bin/tile_example_batched_transpose`

### Arguments

```bash
args:
          -N    input batch size (default:2)
          -C    input channel size. (default:16)
          -H    input height size. (default:1)
          -W    input width size. (default:16)
          -v    whether do CPU validation or not (default: 1)
  -layout_in    input tensor data layout - NCHW by default
 -layout_out    output tensor data layout - NHWC by default
       -seed    seed to be used, -1 means random every time (default:-1)
     -k_name    t to 1 will print kernel name (default:0)
     -warmup    warmup iterations to run this kernel (default:50)
     -repeat    number of iterations to run this kernel (default:100)
```

---

## Source Structure

- **Kernel**: [`batched_transpose_example.hpp`](batched_transpose_example.hpp) (tile-programming kernel template)
- **Executables**: [`batched_transpose_example.cpp`](batched_transpose_example.cpp), [`batched_transpose_api.cpp`](batched_transpose_api.cpp)
- **Build**: `CMakeLists.txt`, `script/`

---

## Related CK Tile Examples

- [06_permute](../06_permute/README.md): Generic permutation with tiles
- [03_gemm](../03_gemm/README.md): GEMM with tiles
- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles

For distribution, see [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
