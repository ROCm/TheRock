# Client Example: Grouped N-Dimensional Convolution Forward

## Theory

This client example demonstrates **grouped N-dimensional convolution forward** for 1D, 2D, and 3D inputs, supporting multiple data types (including BF8 and FP8). Grouped convolution is used in modern CNNs and vision transformers to reduce computation and enable channel-wise or expert-wise processing.

**Mathematical Formulation:**
Given input $X$ and weights $W$ for $G$ groups:
- For each group $g$:
  $$
  Y^g[n, c_{out}, ...] = \sum_{c_{in}} \sum_{k_1} ... \sum_{k_n} X^g[n, c_{in}, ...] \cdot W^g[c_{out}, c_{in}, ...]
  $$
- Each group operates on a subset of input/output channels.

**Algorithmic Background:**

## Grouped Convolution Forward
Grouped convolution operation for 1D, 2D or 3D spatial dimensions. Convolution utilizes GEMM kernel after tensor coordinate transform. In CK Grouped Convolution Forward operation is called as `DeviceGroupedConvFwdMultipleABD` and requires following types as template parameters:

* **NumDimSpatial** - number of spatial dimensions (1D, 2D, 3D).
* **InLayout** - input layout (NHWGC, GNHWC, NGCHW).
* **WeiLayout** - weight layout (GKYXC).
* **DsLayout** - layouts for additional tensors for fused operations.
* **OutLayout** - output layout (NHWGK, GNHWK, NGKHW).
* **ADataType** - input data type. Pass tuple if there is fused operation with input.
* **BDataType** - weight data type. Pass tuple if there is fused operation with weight.
* **DsDataType** - data types for additional tensors for fused operations.
* **EDataType** - Output data type.
* **AElementwiseOperation** - fused operation on tensor A (input).
* **BElementwiseOperation** - fused operation on tensor B (weight).
* **CDEElementwiseOperation** - fused operation on tensor C (output).
* **AComputeType** - compute data type of tensor A for mfma instruction (ADataType by default).
* **BComputeType** - compute data type of tensor B for mfma instruction (AComputeType by default).

Grouped convolution forward support tensors larger than 2GB.

List of the device operations for grouped convolution forward in CK:

* **DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3** - Device operation with XDL instructions. Optimized for AMD Instinct MI300 series.
* **DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle** - Device operation with XDL instructions and support of fused operations to input, weight and output.
* **DeviceGroupedConvFwdMultipleD_Wmma_CShuffle** - Device operation with WMMA instructions.
* **DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK** - Device operation with DL instructions.

Table of supported cases by instance factory with XDL instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|NGCHW/GKCYX/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|---|
|bf16 |2D, 3D|2D|2D|1D, 2D, 3D|
|fp16 |2D, 3D|2D|2D|1D, 2D, 3D|
|fp32 |2D, 3D|2D|2D|1D, 2D, 3D|
|int8 |2D, 3D|2D|2D|1D, 3D|
|fp8  |3D|&cross;|&cross;|&cross;|
|bf8  |3D|&cross;|&cross;|&cross;|

Table of supported cases by instance factory with WMMA instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|fp16 |2D, 3D|&cross;|2D, 3D|
|int8 |2D, 3D|&cross;|2D, 3D|

Table of supported cases by instance factory with DL instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|bf16 |&cross;|&cross;|2D|
|fp16 |&cross;|&cross;|2D|
|fp32 |&cross;|&cross;|2D|
|int8 |&cross;|&cross;|2D|

Table of supported cases by instance factory with fused elementwise operation:

* **Dynamic elementwise operation** - 2D/3D, NHWGC, bf16/fp16/fp32/int8
* **Bilinear** - 3D, NHWGC, bf16/fp16/fp32/int8
* **ConvInvScale** - 3D, NHWGC, fp8
* **ConvScale** - 3D, NHWGC, fp8/bf8
* **ConvScale + Add** - 3D, NHWGC, fp8
* **ConvScale + Relu** - 3D, NHWGC, fp8
* **Scale** - 3D, NHWGC, bf16/fp16/fp32/int8
* **Scale + Add (for A and B)** - 3D, NHWGC, bf16/fp16/fp32/int8
* **Scale + Add + Scale + Add + Relu** - 3D, NHWGC, bf16/fp16/fp32/int8


## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/07_grouped_convnd_fwd
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (2D grouped convolution)
./grouped_conv2d_fwd

# Example run (3D grouped convolution, BF8)
./grouped_conv3d_fwd_bf8

# Example run (3D grouped convolution, FP8)
./grouped_conv3d_fwd_fp8
```

## Source Code Structure

### Directory Layout
```
client_example/07_grouped_convnd_fwd/
├── grouped_conv1d_fwd.cpp         # 1D grouped convolution
├── grouped_conv2d_fwd.cpp         # 2D grouped convolution (NCHW)
├── grouped_conv2d_fwd_ngchw.cpp   # 2D grouped convolution (NGCHW)
├── grouped_conv3d_fwd_bf8.cpp     # 3D grouped convolution (BF8)
├── grouped_conv3d_fwd_fp8.cpp     # 3D grouped convolution (FP8)
├── grouped_conv3d_fwd_bf8_fp8.cpp # 3D grouped convolution (BF8/FP8 mixed)
├── grouped_conv3d_fwd_fp8_bf8.cpp # 3D grouped convolution (FP8/BF8 mixed)
├── common.hpp                     # Common utilities for grouped convolution
├── CMakeLists.txt                 # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures grouped convolution parameters, launches the kernel, and verifies the result.
- **Grouped convolution kernel invocation**:  
  Uses the Composable Kernel device API to launch grouped convolution for different dimensions and data types.

This client example provides a comprehensive demonstration of grouped convolution for efficient CNN and vision transformer models.
