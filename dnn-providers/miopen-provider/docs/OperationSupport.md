# MIOpen Legacy Plugin - Operation Support

This document provides detailed information about the operations supported by the MIOpen Legacy Plugin for hipDNN.

For general information about hipDNN's operation support, please see the [hipDNN Operation Support](../../../docs/OperationSupport.md) documentation.

## Current Operation Support

The following table lists all operations currently supported in hipDNN:

| Operation | Datatypes | Layouts | Notes |
|-----------|-----------|---------|-------|
| Batchnorm Backward  | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Spatial mode only¹ |
| Batchnorm Inference + DRelu + Backward | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Fused graph³ |
| Batchnorm Training  | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Spatial mode only¹, No running stats⁴ |
| Batchnorm Training + Activation | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Fused graph³⁴ |
| Convolution Dgrad   | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only² |
| Convolution Forward | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only² |
| Convolution Forward + (Bias) + Activation⁵ | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Fused graph²³ |
| Convolution Wgrad   | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Cross-correlation only² |

¹ See Batchnorm Operations note below
² See Convolution Operations note below
³ See Fused Operations note below
⁴ See Batchnorm Training Running Statistics note below
⁵ See Detailed Requirements below

## Detailed Requirements

### Convolution Forward + (Bias) + Activation
#### Convolution forward node
- Compute data type: FP32
- Y tensor
    - Virtual
    - Data type:  FP32 or the input data type (the latter only if bias is used)
#### Bias node (optional)
- Compute data type: input data type
- Output tensor
    - Virtual
    - Data type: FP32 or the input data type
#### Activation node
- Compute data type: FP32
- Activation mode: RELU_FORWARD
- Supports
    - no clipping
    - relu_lower_clip set
    - relu_lower_clip and relu_upper_clip set

## Operation Notes

> [!NOTE]
> **Batchnorm Operations:** Currently, only spatial batchnorm mode is supported. Spatial mode computes statistics over the batch (N) and spatial dimensions (H, W, or D, H, W) for each channel.

> [!NOTE]
> **Convolution Operations:** Currently, only cross-correlation convolutions are supported. True mathematical convolution (with kernel flipping) is not yet implemented. In practice, cross-correlation is the standard operation used in modern deep learning frameworks.

> [!NOTE]
> **Fused Operations:** Fused graph patterns combine multiple operations:
> - **Batchnorm Inference + DReLU + Backward:** Combines batchnorm inference, activation backward (DReLU), and batchnorm backward
> - **Batchnorm Training + Activation:** Combines batchnorm training with forward activation
> - **Convolution Forward + (Bias) + Activation:** Combines convolution forward, optional bias addition, and forward activation

> [!NOTE]
> **Activation Functions:** Supports ReLU, Clipped ReLU (with configurable upper clip), and CLAMP (with configurable lower/upper clips).

> [!NOTE]
> **Sparse Support:** All operations currently work with dense tensors only.

> [!NOTE]
> **Batchnorm Training Running Statistics:** Currently, batchnorm training only supports computing batch statistics (mean and inverse variance) without updating running statistics.

## Legend

### Datatypes
- **FP16**: Half-precision floating point (16-bit)
- **BFP16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)

### Layouts
- **NCHW**: Batch, Channels, Height, Width (2D, channel-first)
- **NHWC**: Batch, Height, Width, Channels (2D, channel-last)
- **NCDHW**: Batch, Channels, Depth, Height, Width (3D, channel-first)
- **NDHWC**: Batch, Depth, Height, Width, Channels (3D, channel-last)
