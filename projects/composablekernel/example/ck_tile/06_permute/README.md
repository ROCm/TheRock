# Permute with CK Tile

This example demonstrates generic tensor permutation which is similiar to [torch.permute](https://pytorch.org/docs/stable/generated/torch.permute.html) (combined with [torch.contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html)). Currently we implement a generic permute kernel that support up to rank 8 arbitrary permutation with a single kernel instance. Performance is not the first consideration, we prefer a simple and general kernel implementation using `ck_tile` in this example.


---

## Algorithm and Math

Given a tensor $X$ of shape $[d_0, d_1, ..., d_{n-1}]$ and a permutation $\pi$, compute:
$$
Y_{i_0, i_1, ..., i_{n-1}} = X_{i_{\pi(0)}, i_{\pi(1)}, ..., i_{\pi(n-1)}}
$$

- **Tilewise Permute**: Each thread block processes a tile (block) of the input, computes the permuted indices, and writes to the output.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input tensor.
- **Alternative Implementation**: For rank-7 tensors, a swizzled layout is supported for matrix core-friendly data loading.

---

## Build & Run

### Arguments
```
args:
          -v    weather do CPU validation or not (default:1)
       -prec    data type. fp16/bf16/fp32 (default:fp16)
      -shape    the shape of the input tensor (default:2,3,4)
       -perm    permute perm (default:2,1,0)
```
```
# in the root of ck_tile
mkdir build && cd build
../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_permute -j
```

This will result in an executable `build/bin/tile_example_permute`


### Further Examples

```
# torch
x=torch.randn(2,3,4,6)
y=x.permute(0,3,2,1).contiguous()

# ck_tile
./build/bin/tile_example_permute -shape=2,3,4,6 -perm=0,3,2,1
```

You can try the smoke_test:

```
# in the root of ck_tile, after you build this example
sh example/ck_tile/06_permute/script/smoke_test.sh
```

### Alternative Implementation

We have an alternative implementation under `alternative_impl/` folder, that can swizzle the tensor to be more friendly for data loading for matrix core layout. This can be enabled when dealing with a `rank-7` tensor, with a fixed pattern of either `0,1,4,2,5,3,6` or `0,1,2,4,5,3,6`. There are other shape limitation of this implementation, check the source code of `permute.cpp` for detail.

```
# example
./build/bin/tile_example_permute -shape=3,6,4,32,16,2,8 -perm=0,1,4,2,5,3,6 # b_n0_k0_n1_k1_n2_k2
./build/bin/tile_example_permute -shape=3,8,4,16,16,4,8 -perm=0,1,2,4,5,3,6 # b_n0_n1_k0_k1_n2_k2
```

---

## Source Structure

- **Kernel**: `permute.hpp` (tile-programming kernel template)
- **Executable**: `permute.cpp` (argument parsing, kernel launch)
- **Alternative**: `alternative_impl/` (swizzled layout for rank-7 tensors)
- **Build**: `CMakeLists.txt`, `script/`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): GEMM with tiles
- [05_reduce](../05_reduce/README.md): Reductions with tiles
- [35_batched_transpose](../35_batched_transpose/README.md): Batched transpose with tiles

For distribution, `include/ck_tile/tile_program/tile_distribution/`.

---
[Back to CK Tile Examples](../README.md)
