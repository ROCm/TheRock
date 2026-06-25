[Back to supported operations](../../include/ck/README.md)
# Composable Kernel Grouped Convolution

## Grouped Convolution Backward Data

Grouped convolution operation for 1D, 2D or 3D spatial dimensions. Convolution utilizes GEMM kernel after tensor coordinate transform. In CK Grouped Convolution Backward Data operation is called as `DeviceGroupedConvBwdDataMultipleD` and requires following types as template parameters:

* **NumDimSpatial** - number of spatial dimensions (1D, 2D, 3D).
* **ALayout** - output layout (NHWGK, GNHWK, NGKHW).
* **BLayout** - weight layout (GKYXC).
* **DsLayout** - layouts for additional tensors for fused operations.
* **ELayout** - input layout (NHWGC, GNHWC, NGCHW).
* **ADataType** - output data type.
* **BDataType** - weight data type.
* **DsDataType** - data types for additional tensors for fused operations.
* **EDataType** - input data type.
* **AElementwiseOperation** - fused operation on tensor A (output).
* **BElementwiseOperation** - fused operation on tensor B (weight).
* **CDEElementwiseOperation** - fused operation on tensor C (input).
* **AComputeType** - compute data type of tensor A for mfma instruction (ADataType by default).
* **BComputeType** - compute data type of tensor B for mfma instruction (AComputeType by default).

Grouped convolution backward data supports tensors larger than 2GB (except when image is larger than 2GB).

List of the device operations for grouped convolution backward data in CK:

* **DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1** - Device operation with XDL instructions and support of fused operations to input.
* **DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle** - Device operation with WMMA instructions.

Table of supported cases by instance factory with XDL instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|bf16|2D, 3D|2D, 3D|2D, 3D|
|fp16 |2D, 3D|2D, 3D|2D, 3D|
|fp32  |2D, 3D|2D, 3D|2D, 3D|

Table of supported cases by instance factory with WMMA instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|fp16 |2D, 3D|&cross;|2D, 3D|
|int8 |2D, 3D|&cross;|2D, 3D|

Table of supported cases by instance factory with fused elementwise operation:

* **Bilinear** - 3D, NHWGC, bf16/fp16/fp32
* **Scale** - 3D, NHWGC, bf16/fp16/fp32

---

## Theory

**Grouped convolution backward data** computes the gradient of the input tensor with respect to the loss, given the output gradient and the weights, for each group independently. This is essential for training CNNs and grouped/expert models.

**Mathematical Formulation:**
For each group $g$:
$$
\text{InputGrad}^g = \text{ConvBwdData}(\text{OutputGrad}^g, \text{Weights}^g)
$$

- Supports 1D, 2D, and 3D grouped convolutions.
- Utilizes implicit GEMM for efficient computation.
- Supports fused elementwise operations (e.g., bilinear, scale).

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/10_grouped_convnd_bwd_data
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (2D grouped convolution backward data)
./grouped_conv2d_bwd_data
```

## Source Code Structure

### Directory Layout
```
client_example/10_grouped_convnd_bwd_data/
├── grouped_conv1d_bwd_data.cpp         # 1D grouped convolution backward data
├── grouped_conv2d_bwd_data.cpp         # 2D grouped convolution backward data
├── grouped_conv3d_bwd_data.cpp         # 3D grouped convolution backward data
├── CMakeLists.txt                      # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input/output tensors, configures grouped convolution parameters, launches the backward data kernel, and verifies the result.
- **Grouped convolution backward kernel invocation**:  
  Uses the Composable Kernel device API to launch grouped convolution backward data for different dimensions and data types.

This client example provides a comprehensive demonstration of grouped convolution backward data for efficient CNN and vision transformer training.
