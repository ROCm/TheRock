# Quant GEMM Matrix Multiplication

This folder contains examples of quant GEMMs using the ck_tile tile-programming implementation.

- AQuant kernel with blocks of A matrix sharing scales: custom GEMM pipeline
- BQuant kernel with blocks of B matrix sharing scales: custom GEMM pipeline
- Row and Column-wise scaled: All of the row-wise elements in A Matrix and column-wise elements in B Matrix will share the same quantization element and the element-wise operation will complete in epilogue.
- Tensor-wise scaled: Share the same scalar scale across the whole tensor of A or B

## Quantization Mode Comparison

| Quant Mode | A Matrix Organization | A Scale Shape | B Matrix Organization | B Scale Shape |
|------------|----------------------|---------------|----------------------|---------------|
| **AQuant** | Blocks along K dimension<br/>Each M×GroupSize block shares one scale | `[M, K/GroupSize]` | Not quantized | N/A |
| **BQuant** | Not quantized | N/A | Blocks along K dimension<br/>Each GroupSize×N block shares one scale | `[K/GroupSize, N]` |
| **RowColQuant** | Per-row quantization<br/>All K elements in each row share one scale | `[M, 1]` | Per-column quantization<br/>All K elements in each column share one scale | `[1, N]` |
| **TensorQuant** | Tensor-wise quantization<br/>All M×K elements share one scale | `[1]` | Tensor-wise quantization<br/>All K×N elements share one scale | `[1]` |

---

## Features

- **Preshuffled GEMM**: Shuffle the GEMM of B (weight) matrix in the warp layout and bypass the shared memory to do the GEMM calculation. Best performance solution for GEMM.
- **TransposeC**: Transpose the C Matrix Output layout to have the best coalesced scale reading
- **Preshuffled Quant**: Preshuffle the input matrix to load multiple Quant warp blocks along the selected dimension.
- **Precision**: Supports fp16, bf16, fp8, bf8, int4 (for B Matrix), uint8 (split into two fp4 in the pipeline (for B Matrix)).
- **Validation**: CPU/GPU validation and error tolerance options.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# Compile the quant kernels
make tile_example_gemm_quant -j
```
This will result in an executable `build/bin/tile_example_gemm_quant`

## example
```
args:
               -h    Print help message (default:false)
               -m    m dimension (default:3840)
               -n    n dimension (default:4096)
               -k    k dimension (default:2048)
        -a_layout    A tensor data layout - R for Row or C for Column (default:R)
        -b_layout    B tensor data layout - R for Row or C for Column (default:C)
       -bq_layout    Bq tensor data layout - R for Row or C for Column (default:C)
        -c_layout    C tensor data layout - R for Row or C for Column (default:R)
        -stride_a    Tensor A stride (default:0)
        -stride_q    Tensor AQ stride (default:0)
        -stride_b    Tensor B stride (default:0)
        -stride_c    Tensor C stride (default:0)
               -v    0: No validation, 1: Validation on CPU, 2: Validation on GPU (default:1)
            -prec    Data type. For AQuant: fp8, bf8, i4fp8, or i4bf8;  for Bquant: fp8, bf8, fp8i4, bf8i4, or bf16fp4 (default for both AQuant and Bquant: fp8)
          -warmup    Number of iterations before benchmarking the kernel (default:50)
          -repeat    Number of iterations to benchmark the kernel (default:1000)
           -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
         -split_k    SplitK value (default:1)
          -device    Device id that will be used to run the kernel (default:0)
            -init    0:random, 1:linear, 2:constant(1) (default:0)
     -flush_cache    Flush cache before running the kernel (default:true)
  -rotating_count    Rotating count (default:1000)
      -quant_mode    Choose aquant, bquant, tensor or rowcol (default:bquant)
     -preshuffleb    Enable preshuffle of tensor B (default:false)
 -preshufflequant   Enable preshuffle of quant tensor (default:false)
      -group_size    Quantization group size as MxNxK, e.g., 1x1x128, 1x32x128, 1x64x128 (default:1x1x128)
```

User need to select correct mapping of config for each quant mode:

|  | quant_mode as runtime argument | Corresponding cpp file | GemmConfig at the top of cpp file |
|:--------|:-----:|:-----:|-------|
| For selecting AQuant  | aquant  | gemm_aquant_quantgrouped.cpp|  GemmConfigQuantDecode    |
| For selecting AQuant with Preshuffle quant    | aquant  | gemm_aquant_quantgrouped_preshufflequant.cpp |  GemmConfigPreshuffleQuantDecode    |
| For selecting BQuant  | bquant  | gemm_bquant_quantgrouped_<prec_type>.cpp| GemmConfigQuantDecode (or) GemmConfigQuantPrefill     |
| For selecting BQuant with Preshuffle quant | bquant  | gemm_bquant_quantgrouped_preshufflequant.cpp|  GemmConfigPreshuffleQuantDecode  (or) GemmConfigPreshuffleBQuantPrefill     |
| For selecting PreShuffle B with BQuant | bquant | gemm_bquant_quantgrouped_preshuffleb.cpp| GemmConfigPreshuffleB_BQuant_Decode (or) GemmConfigPreshuffleB_BQuant_Prefill
| For selecting PreShuffle B with preshuffle BQuant | bquant | gemm_bquant_quantgrouped_preshuffleb_preshufflequant.cpp |GemmConfigPreshuffleB_PreshuffleBQuant_Decode (or) GemmConfigPreshuffleB_PreshuffleBQuant_Prefill
| For selecting RowCol quant  | rowcolquant  | gemm_quant_rowcol| GemmConfigRowColQuant    |

