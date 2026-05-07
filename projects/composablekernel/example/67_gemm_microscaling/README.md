# GEMM with Microscaling

This example demonstrates a **GEMM operation with microscaling**, an advanced quantization technique that applies fine-grained scaling to small blocks of data. Microscaling enables more precise quantization than traditional methods by using different scale factors for small groups of elements, leading to better accuracy preservation in quantized neural network inference.

## Source Code Organization

-   [`gemm_microscaling_xdl.cpp`](./gemm_microscaling_xdl.cpp): The main example file. It sets up microscaled matrices with quantized data and scale factors, and instantiates the `DeviceGemmMicroscaling` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_gemm_microscaling.hpp`](../../include/ck/tensor_operation/gpu/device/device_gemm_microscaling.hpp): The device interface for GEMM with microscaling support.
-   The underlying kernel implements sophisticated block-wise dequantization integrated into the GEMM computation pipeline.

## Build and Run

### example_gemm_mx_fp8

Custom verification parameters:
```bash
# arg1: verification (0=no, 1=CPU)
# arg2: initialization (0=constant values, 1=integer values, 2=decimal values)
# arg3: time kernel (0=no, 1=yes)
# arg4: verbosity (0=no info, 1=verbose info)
# arg5 to 10: M(256x), N(256x), K(512x), StrideA, StrideB, StrideC
# arg11: KBatch
# arg12: warmup runs pre-timing
# arg13: repeat run count for timing
./bin/example_gemm_mx_fp8 1 1 0 1
```

Custom tensor shapes:
```bash
./bin/example_gemm_mx_fp8 1 2 1 0 256  256  512 -1 -1 -1 1 10 10
```

### Run the Example

Custom verification parameters:
```bash
# arg1: verification (0=no, 1=CPU)
# arg2: initialization (0=constant values, 1=integer values, 2=decimal values)
# arg3: time kernel (0=no, 1=yes)
# arg4: verbosity (0=no info, 1=verbose info)
# arg5 to 10: M(128x), N(128x), K(64x), StrideA, StrideB, StrideC
# arg11: KBatch
./bin/example_gemm_mx_fp8 1 1 0 1
```

Custom tensor shapes:
```bash
./bin/example_gemm_mx_fp8 1 2 1 0 128  128  256 -1 -1 -1 1
```

Default invocation:
```bash
# Implies: ./bin/example_gemm_mx_fp8 1 2 0 0
./bin/example_gemm_mx_fp8
```

