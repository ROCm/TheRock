# Client Example: Grouped Convolution with Activation and Fusion

## Theory

This client example demonstrates **grouped convolution fused with various activation and elementwise operations**. Grouped convolution splits the input and weights into groups and applies convolution independently to each group, while fusion with activation and scaling improves efficiency.

**Mathematical Formulation:**
For each group $g$:
- Convolution: $Y^g = \text{Conv}(X^g, W^g)$
- Fused operations: $E^g = f(Y^g, D_0^g, D_1^g, ...)$
  - $f$ can be bilinear, scale, add, relu, etc.

**Algorithmic Background:**
- Grouped convolution is used in efficient CNNs, depthwise separable convolutions, and expert models.
- Fused epilogue operations (scale, add, relu, reduce) are performed in registers before writing to memory.
- Supports 1D, 2D, and 3D grouped convolutions and a variety of fusion patterns.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/24_grouped_conv_activation
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (grouped conv + scale)
./grouped_convnd_fwd_scale/grouped_convnd_fwd_scale

# Example run (grouped conv + bilinear)
./grouped_convnd_fwd_bilinear/grouped_convnd_fwd_bilinear

# Example run (grouped conv + scale + relu)
./grouped_convnd_fwd_convscale_relu/grouped_convnd_fwd_convscale_relu

# Example run (grouped conv + scale + add + relu)
./grouped_convnd_fwd_scaleadd_scaleadd_relu/grouped_convnd_fwd_scaleadd_scaleadd_relu
```

## Source Code Structure

### Directory Layout
```
client_example/24_grouped_conv_activation/
├── grouped_convnd_fwd_scale/                  # Grouped conv + scale
├── grouped_convnd_fwd_bilinear/               # Grouped conv + bilinear
├── grouped_convnd_fwd_convscale/              # Grouped conv + scale (convscale)
├── grouped_convnd_fwd_convscale_add/          # Grouped conv + scale + add
├── grouped_convnd_fwd_convscale_reduce/       # Grouped conv + scale + reduce
├── grouped_convnd_fwd_convscale_relu/         # Grouped conv + scale + relu
├── grouped_convnd_fwd_convinvscale/           # Grouped conv + inverse scale
├── grouped_convnd_fwd_scaleadd_ab/            # Grouped conv + scale + add (A/B)
├── grouped_convnd_fwd_scaleadd_scaleadd_relu/ # Grouped conv + scale + add + relu
├── grouped_convnd_bwd_data_bilinear/          # Grouped conv bwd data + bilinear
├── grouped_convnd_bwd_data_scale/             # Grouped conv bwd data + scale
├── grouped_convnd_bwd_weight_bilinear/        # Grouped conv bwd weight + bilinear
├── grouped_convnd_bwd_weight_scale/           # Grouped conv bwd weight + scale
├── CMakeLists.txt                             # Build configuration for the example
```

### Key Functions

- **main()** (in each subdirectory's `.cpp`):  
  Sets up input tensors, configures grouped convolution and fusion parameters, launches the kernel, and verifies the result.
- **Grouped convolution kernel invocation**:  
  Uses the Composable Kernel device API to launch grouped convolution with various fused epilogue operations.

---

## Additional Details

- Supports a wide range of fusion patterns (bilinear, scale, add, relu, reduce, etc.).
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [10_grouped_convnd_bwd_data](../10_grouped_convnd_bwd_data/README.md): Grouped convolution backward data
- [11_grouped_conv_bwd_weight](../11_grouped_conv_bwd_weight/README.md): Grouped convolution backward weight
- [30_grouped_conv_fwd_multiple_d](../../example/30_grouped_conv_fwd_multiple_d/README.md): Grouped convolution forward with multiple D

---
[Back to Client Examples](../README.md)
