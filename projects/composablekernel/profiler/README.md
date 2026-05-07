[Back to the main page](../README.md)
# Composable Kernel profiler

## Building Specific Profilers
To reduce build time, filter which operations to compile using CMake options:

```bash
# Build all grouped_gemm variants (grouped_gemm, grouped_gemm_fastgelu, grouped_gemm_tile_loop, etc.)
cmake -DCK_PROFILER_OP_FILTER="grouped_gemm" <other options> ..

# Build ONLY base grouped_gemm (excludes variants - use exact regex match with ^ and $)
cmake -DCK_PROFILER_OP_FILTER="^grouped_gemm$" <other options> ..
```

Both `CK_PROFILER_OP_FILTER` and `CK_PROFILER_INSTANCE_FILTER` accept regex patterns. Default builds all operations.

To find the complete list of operations, run the following command:
```bash
find profiler/src -name "profile_*.cpp" | sed 's|profiler/src/profile_||' | sed 's|.cpp||' | sort
```
## Profiler GEMM UNIVERSAL kernels
```bash
# arg1: tensor operation (gemm_universal: Universal GEMM)
# arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8; 4: f8@f16; 5: f16@f8; 6: f16->f8; 7: f8->bf16, comp f8; 8: f16@i4; 9: bf16@i4
# arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];
#                     1: A[m, k] * B[n, k] = C[m, n];
#                     2: A[k, m] * B[k, n] = C[m, n];
#                     3: A[k, m] * B[n, k] = C[m, n])
# arg4: verification (0: no; 1: yes)
# arg5: initialization (0: no init; 1: integer value; 2: decimal value)
# arg6: print tensor value (0: no; 1: yes)
# arg7: time kernel (0=no, 1=yes)
# arg8 to 13: M, N, K, StrideA, StrideB, StrideC
# arg14: split k into  mulitiple batch
# optional:
# arg15: number of warm-up cycles (default 1)
# arg16: number of iterations (default 10)
# arg17: memory for rotating buffer (default 0, size in MB)


################        op  datatype  layout  verify  init  print  time  M N K  StrideA StrideB StrideC  SplitK  WarmupCycles  Iterations  MemoryBuffer
./bin/ckProfiler gemm_universal 1 0 1 1 0 1 4096 4096 4096 4096 4096 4096 1 1 10 0
```

## Profile GEMM kernels
```bash
#arg1: tensor operation (gemm=GEMM)
#arg2: data type (0=fp32, 1=fp16)
#arg3: matrix layout (0=NN, 1=NT, 2=TN, 3=TT)
#arg4: verification (0=no, 1=yes)
#arg5: initialization (0=no init, 1=integer value, 2=decimal value)
#arg6: print matrix value (0=no, 1=yes)
#arg7: run kernel # of times (>1)
#arg8 to 13: M, N, K, StrideA, StrideB, StrideC

################        op  datatype  layout  verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC
./bin/ckProfiler      gemm         1       1       1     1    0       5  3840 4096 4096     4096    4096    4096
```

## Profile 2D forward convolution kernels
```bash
#arg1: tensor operation (conv=Convolution)
#arg2: data type (0=fp32, 1=fp16)
#arg3: input tensor layout (0=NCHW, 1=NHWC)
#arg4: weight tensor layout (0=KCYX, 1=KYXC)
#arg5: output tensor layout (0=NKHW, 1=NHWK)
#arg6: verification (0=no, 1=yes)
#arg7: initialization (0=no init, 1=integer value, 2=decimal value)
#arg8: print matrix value (0=no, 1=yes)
#arg9: run kernel # of times (>1)
#arg10 to 24: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, RightPx
 ################          op datatype  in_layout   wei_layout  out_layout  verify  init  log  repeat  N__ K___ C___ Y X Hi__ Wi__ Strides Dilations LeftPads RightPads
 ./bin/ckProfiler  conv2d_fwd        1          1            1           1       1     1    0       5  128  256  192 3 3   71   71     2 2       1 1      1 1       1 1
```

## Profile contraction kernels
```bash
#arg1: tensor operation (contraction_bilinear=CONTRACTION+Bilinear)
#arg2: data type (0: fp32; 1: f64; 2: f16; 3: bf16)
#arg3: compute data type (0: fp32; 1: f64; 2: f16; 3: bf16)
#arg4: Number of dimension for M, N and K (one for all)
#arg5: matrix layout (0: A[m0, m1, k0, k1] * B[k0, k1, n0, n1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1];
#                     1: A[m0, m1, k0, k1] * B[n0, n1, k0, k1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1];
#                     2: A[k0, k1, m0, m1] * B[k0, k1, n0, n1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1];
#                     3: A[k0, k1, m0, m1] * B[n0, n1, k0, k1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1])
#arg6: verification (0: no; 1: yes)
#arg7: initialization (0: no init; 1: integer value; 2: decimal 
#      value)
#arg8: print tensor value (0: no; 1: yes)
#arg9: time kernel (0: no, 1: yes)
#arg10: alpha
#arg11: beta
#arg12 to 17/29: M0, M1, N0, N1, K0, K1
#arg18/30 to 33/77: Strides for A, B, D and E (skip for default)

################                   op  datatype  compute_datatype  num_dim layout  verify  init  log  time  alpha  beta  M0  M1  N0  N1  K0  K1
./bin/ckProfiler contraction_bilinear         0                 0        2      1       0     0    0     1    1.0   1.0 128 128 128 128 128 128
```

## Profile batched gemm multiple D kernels
```bash
#arg1: tensor operation (batched_gemm_multi_d=Batched GEMM multi D);
#arg2: data type (0: fp16; 1: int8)
#arg3: matrix layout (0: A[g, m, k] * B[g, k, n] = C[g, m, n];
#                     1: A[g, m, k] * B[g, n, k] = C[g, m, n];
#                     2: A[g, k, m] * B[g, k, n] = C[g, m, n];
#                     3: A[g, k, m] * B[g, n, k] = C[g, m, n])
#arg4: verification (0: no; 1: yes)
#arg5: initialization (0: no init; 1: integer value; 2: decimal value)
#arg6: print tensor value (0: no; 1: yes)
#arg7: time kernel (0=n0, 1=yes)
#arg8 to 17: M, N, K, StrideA, StrideB, StrideC, BatchStrideA, BatchStrideB, BatchStrideC, BatchCount

################                   op  datatype  layout  verify  init  log  time    M    N    K StrideA StrideB StrideC BatchStrideA BatchStrideB BatchStrideC BatchCount
./bin/ckProfiler batched_gemm_multi_d         0       1       0     0    0     1 4096 4096 4096    4096    4096    4096     16777216     16777216     16777216         16
```

## Profile grouped convolution backward data kernels
```bash
# arg1: tensor operation (grouped_conv_bwd_data: Grouped Convolution Backward Data)
# arg2: data type (0: Output fp32, Weight fp32, Input fp32
#                  1: Output fp16, Weight fp16, Input fp16
#                  2: Output bf16, Weight bf16, Input bf16
# arg3: tensor layout (0: Output[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Input[G, N, Ho, Wo, K]
#                      1: Output[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Input[N, Ho, Wo, G, K])
# arg4: verification (0: no, 1: yes)
# arg5: initialization (0: no init, 1: integer value, 2: decimal value)
# arg6: print tensor value (0: no; 1: yes)
# arg7: time kernel (0: no, 1: yes)
# Following arguments (depending on number of spatial dims):
#  Number of spatial dimensions (1=Conv1D, 2=Conv2D, 3=Conv3D)
#  G, N, K, C, 
#  <filter spatial dimensions>, (ie Y, X for 2D)
#  <input image spatial dimensions>, (ie Hi, Wi for 2D)
#  <strides>, (ie Sy, Sx for 2D)
#  <dilations>, (ie Dy, Dx for 2D)
#  <left padding>, (ie LeftPy, LeftPx for 2D)
#  <right padding>, (ie RightPy, RightPx for 2D)

 ################                   op   datatype  layout  verify  init  log  time  Ndims  G  N   K   C  Y  X  Hi  Wi  Sy  Sx  Dy  Dx  LeftPy  LeftPx  RightPy  RightPx
./bin/ckProfiler grouped_conv_bwd_data          1       0       1     1    0     1      2 32  4 192 192  3  3  28  28   1   1   1   1       1       1        1        1

```

## Profile grouped convolution backward weight kernels
```bash
# arg1: tensor operation (grouped_conv_bwd_weight: Grouped Convolution Backward Weight)
# arg2: data type (0: Input fp32, Weight fp32, Output fp32
#                  1: Input fp16, Weight fp16, Output fp16
#                  2: Input bf16, Weight fp32, Output bf16
#                  3: Input fp16, Weight fp16, Output fp16, Gemm bf8@fp8
#                  4: Input int8, Weight int8, Output int8)
# arg3: tensor layout (0: Input[G, N, C, Hi, Wi], Weight[G, K, C, Y, X], Output[G, N, K, Ho, Wo]
#                      1: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, N, Ho, Wo, K]
#                      2: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, Ho, Wo, G, K]
# arg4: verification (0: no, 1: yes)
# arg5: initialization (0: no init, 1: integer value, 2: decimal value)
# arg6: print tensor value (0: no; 1: yes)
# arg7: time kernel (0: no, 1: yes)
# Following arguments (depending on number of spatial dims):
#  Number of spatial dimensions (1=Conv1D, 2=Conv2D, 3=Conv3D)
#  G, N, K, C, 
#  <filter spatial dimensions>, (ie Y, X for 2D)
#  <input image spatial dimensions>, (ie Hi, Wi for 2D)
#  <strides>, (ie Sy, Sx for 2D)
#  <dilations>, (ie Dy, Dx for 2D)
#  <left padding>, (ie LeftPy, LeftPx for 2D)
#  <right padding>, (ie RightPy, RightPx for 2D)
# SplitK (-1 for internally computed split-K value, positive value to set k batches explicitly, or 'all' to test all internal split-K values)

 ################                   op   datatype  layout  verify  init  log  time  Ndims  G   N   K   C  Y  X  Hi  Wi  Sy  Sx  Dy  Dx  LeftPy  LeftPx  RightPy  RightPx  SplitK
./bin/ckProfiler grouped_conv_bwd_weight         1       1      0     1    0     1      2 32 256 256 512  3  3  28  28   1   1   1   1       1       0        0        0       1

```

Note: This kernel use atomic add, this will cause output buffer to be accumulated multiple times, causing verification failure. To work around it, do not use CK's own timer and do verification at the same time.

## Profile image to column/column to image kernels

```bash
# arg1: tensor operation ( conv_tensor_rearrange : Conv Tensor Rearrange )
# arg2: data type (0: Input fp32, Weight fp32, Output fp32
#                  1: Input fp16, Weight fp16, Output fp16
#                  2: Input bf16, Weight bf16, Output bf16
#                  3: Input int8, Weight int8, Output int8)
# arg3: tensor layout (0: Input[G, N, Hi, Wi, C], Output[G * N * Ho * Wo, Y * X * C],
#                      1: Input[N, Hi, Wi, G, C], Output[N * Ho * Wo * G, Y * X * C])
# arg4: verification (0: no, 1: yes)
# arg5: initialization (0: no init, 1: integer value, 2: decimal value)
# arg6: print tensor value (0: no; 1: yes)
# arg7: time kernel (0: no, 1: yes)
# arg8: operation type (0: ImageToColumn, 1: ColumnToImage)
# Following arguments (depending on number of spatial dims):
#  Number of spatial dimensions (1=Conv1D, 2=Conv2D, 3=Conv3D)
#  G, N, K, C, 
#  <filter spatial dimensions>, (ie Y, X for 2D)
#  <input image spatial dimensions>, (ie Hi, Wi for 2D)
#  <strides>, (ie Sy, Sx for 2D)
#  <dilations>, (ie Dy, Dx for 2D)
#  <left padding>, (ie LeftPy, LeftPx for 2D)
#  <right padding>, (ie RightPy, RightPx for 2D)

 ################                   op   datatype  layout  verify  init  log  time opType Ndims  G   N   K   C  Y  X  Hi  Wi  Sy  Sx  Dy  Dx  LeftPy  LeftPx  RightPy  RightPx
./bin/ckProfiler conv_tensor_rearrange          0       0       0     1    0     1      0     2  1 256   1 512  3  3   28  28   1   1   1   1        0       0       0        0

```

Note: Column to image kernel adds to the output memory, this will cause output buffer to be accumulated multiple times, causing verification failure. To work around it, do not use CK's own timer and do verification at the same time.

## Profile Permute scale kernels

```bash
# arg1: tensor operation ( permute_scale : Permute Scale )
# arg2: data type (0: Input fp32, Output fp32
#                  1: Input fp16, Output fp16
# arg4: verification (0: no, 1: yes)
# arg5: initialization (0: no init, 1: integer value, 2: decimal value)
# arg6: print tensor value (0: no; 1: yes)
# arg7: time kernel (0: no, 1: yes)
# from arg8: tensor lengths
#            input strides
#            output strides

################            op datatype  verify  init  log  time  dim0 dim1 dim2 in_stride0 in_stride1 in_stride2 out_stride0 out_stride1 out_stride2
./bin/ckProfiler permute_scale        0       1     1    0     1    64   64   64       4096         64          1           1          64        4096
```

## Convert MIOpen driver command to CKProfiler

```bash
python3 ../script/convert_miopen_driver_to_profiler.py
/opt/rocm/bin/MIOpenDriver conv -n 32 -c 64 -H 28 -W 28 -k 64 -y 3 -x 3
-p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 32 -F 1 -t 1 
```

Only convolution driver is supported.
