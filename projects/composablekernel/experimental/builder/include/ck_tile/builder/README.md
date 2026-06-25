# Composable Kernel Builder Design Documentation

This directory contains the builder framework for Composable Kernel, which provides a compile-time, type-safe interface for constructing convolution operations with various configurations.

## Table of Contents

- [Convolution Signature](#convolution-signature)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Core Components](#core-components)
  - [Concepts and Validation](#concepts-and-validation)
- [Convolution Algorithm](#convolution-algorithm)
- [Convolution Factory](#convolution-factory)
---

## Convolution Signature

### Overview

The convolution signature system provides a **compile-time description** of grouped convolution operations. A signature is a collection of properties that fully characterize a convolution kernel's mathematical and operational behavior, enabling:

- **Compile-time validation**: Ensures type safety and correctness before kernel instantiation
- **Kernel selection**: Matches user requirements to optimized implementations
- **Specialization**: Enables optimized code paths for specific configurations
- **Composability**: Supports building complex operations from simpler components

The signature leverages modern C++20 features, particularly **concepts**, to provide expressive, self-documenting interfaces with compile-time guarantees.

### Architecture

The signature system is organized into a hierarchical structure:

```
┌─────────────────────────────────────────────────────────┐
│                    ConvSignature                        │
├─────────────────────────────────────────────────────────┤
│ Properties:                                             │
│   • spatial_dim: int           (1D, 2D, or 3D)          │
│   • direction: ConvDirection   (Fwd/BwdData/BwdWeight)  │
│   • data_type: DataType        (default data type)      │
│   • accumulation_data_type: DataType                    │
│   • input: ConvTensor          ──┐                      │
│   • weight: ConvTensor         ──│                      │
│   • output: ConvTensor         ──│                      │
└──────────────────────────────────┼──────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │           ConvTensor                    │
              ├─────────────────────────────────────────┤
              │ ╔═════════════════════════════════════╗ │
              │ ║ TensorConfig (required)             ║ │
              │ ╠═════════════════════════════════════╣ │
              │ ║  • layout: ConvLayout               ║ │
              │ ║  • data_type: DataType (optional)   ║ │
              │ ║  • compute_type: DataType (optional)║ │
              │ ╚═════════════════════════════════════╝ │
              │                                         │
              │ ┌─────────────────────────────────────┐ │
              │ │ TensorOperation (optional)          │ │
              │ ├─────────────────────────────────────┤ │
              │ │  • elementwise_operation            │ │
              │ │  • auxiliary_operand_configs[]      │ │
              │ │    (each is also ConvTensor)  ◄───────┼─┐
              │ └─────────────────────────────────────┘ │ │
              └─────────────────────────────────────────┘ │
                                                          │
                                 Recursive ───────────────┘
```
Key Design Points:
  - ConvSignature contains three ConvTensor instances (input, weight, output)
  - All tensors share the same ConvTensor structure
  - Each ConvTensor has:
    - TensorConfig (required): Defines layout as well as optional data and compute type overrides
    - TensorOperation (optional): Defines fused elementwise operations
  - Auxiliary operands (e.g., bias) in TensorOperation also use the ConvTensor type

### Core Components

#### 1. Signature Level

The top-level signature contains global properties that apply to the entire convolution operation:

```cpp
template <typename T>
concept ConvSignatureDescriptor = requires(T t) {
    { t.spatial_dim } -> std::convertible_to<unsigned int>;  // 1, 2, or 3
    { t.data_type } -> std::convertible_to<DataType>;        // Default data type
    { t.input } -> ConvTensorDescriptor;
    { t.weight } -> ConvTensorDescriptor;
    { t.output } -> ConvTensorDescriptor;
    requires ConvolutionDirectionWellDefinedIfProvided<T>;   // Optional direction
};
```

**Properties:**
- **`spatial_dim`**: Dimensionality of the convolution (1D, 2D, or 3D)
- **`direction`**: Operation type (optional, defaults to FORWARD)
  - `FORWARD`: Standard forward convolution
  - `BACKWARD_DATA`: Gradient computation w.r.t. input
  - `BACKWARD_WEIGHT`: Gradient computation w.r.t. weights
- **`data_type`**: Default data type for all tensors (FP32, FP16, BF16, FP8, I8, U8)
- **`accumulation_data_type`**: Type used for internal accumulation

#### 2. Tensor Level

Each tensor (input, weight, output) has its own descriptor:

```cpp
template <typename T>
concept ConvTensorDescriptor = requires(T t) {
    { t.config } -> TensorConfigDescriptor;
    requires ElementwiseOpWellDefinedIfProvided<T>;
};
```

A tensor descriptor encapsulates:
- **Configuration**: Layout and data type information
- **Operation** (optional): Fused elementwise operations on this tensor

#### 3. Tensor Configuration

Describes the memory layout and data types:

```cpp
template <typename T>
concept TensorConfigDescriptor = requires(T t) {
    { t.layout } -> std::convertible_to<ConvLayout>;
    { t.data_type } -> std::convertible_to<DataType>;  // Optional override
};
```

**Layout Types** (dimension-specific):
- **1D Convolution**:
  - Input: `GNCW`, `GNWC`, `NWGC`, `NGCW`, `G_NW_C_strided`
  - Weight: `GKXC`, `GKCX`, `KXGC`, `G_K_X_C_strided`
  - Output: `GNKW`, `GNWK`, `NWGK`, `NGKW`, `G_NW_K_strided`
  
- **2D Convolution**:
  - Input: `GNCHW`, `GNHWC`, `NHWGC`, `NGCHW`, `G_NHW_C_strided`
  - Weight: `GKYXC`, `GKCYX`, `KYXGC`, `G_K_YX_C_strided`
  - Output: `GNKHW`, `GNHWK`, `NHWGK`, `NGKHW`, `G_NHW_K_strided`
  
- **3D Convolution**:
  - Input: `GNCDHW`, `GNDHWC`, `NDHWGC`, `NGCDHW`, `G_NDHW_C_strided`
  - Weight: `GKZYXC`, `GKCZYX`, `KZYXGC`, `G_K_ZYX_C_strided`
  - Output: `GNKDHW`, `GNDHWK`, `NDHWGK`, `NGKDHW`, `G_NDHW_K_strided`

Where:
- `G` = Groups
- `N` = Batch size
- `C` = Input channels
- `K` = Output channels (filters)
- `W`, `H`, `D` = Width, Height, Depth (spatial dimensions)
- `X`, `Y`, `Z` = Filter dimensions

#### 4. Tensor Operations

Describes fused elementwise operations applied to a tensor:

```cpp
template <typename T>
concept TensorOperatorDescriptor = requires(T t) {
    { t.elementwise_operation } -> std::convertible_to<ElementwiseOperation>;
    requires AuxiliaryOperandConfigsWellDefinedIfProvided<T>;
};
```

**Supported Operations:**
- `PASS_THROUGH`: No operation (identity)
- `SCALE`: Multiply by a scalar
- `CLAMP`: Clamp values to a range
- `BIAS_BNORM_CLAMP`: Bias addition + batch normalization + clamp
- `SCALEADD_SCALEADD_RELU`: Fused scale-add operations + ReLU activation

**Auxiliary Operands:**
Some operations require additional tensor inputs (e.g., bias tensors, scaling factors). These are specified through `auxiliary_operand_configs`, which is an array of `TensorConfigDescriptor` objects describing the layout and data type of each auxiliary input.

### Concepts and Validation

The signature system uses C++20 concepts for compile-time validation at multiple levels:

#### Constraint Concepts

```cpp
// Spatial dimension must be 1, 2, or 3
template <auto N>
concept ConvSpatialDim = std::is_integral_v<decltype(N)> && (N == 1 || N == 2 || N == 3);

// Valid data types for convolution
template <DataType T>
concept ValidConvDataType = 
    (T == DataType::FP32) || (T == DataType::FP16) || (T == DataType::BF16) ||
    (T == DataType::FP8) || (T == DataType::I8) || (T == DataType::U8);
```

#### Validation Concept

```cpp
// Validates a complete signature
template <auto Sig>
concept ValidConvSignature = requires {
    requires ConvSpatialDim<Sig.spatial_dim>;
    requires ValidConvDataType<Sig.data_type>;
};
```

#### Tensor Descriptors

The layout/data type/elementwise operation are described per tensor. This multi-level hierarchy allows:
- **Flexibility**: Each tensor can have independent layout and data type
- **Reusability**: Common configurations can be shared across different signatures
- **Extensibility**: New properties can be added to specific levels without affecting others
- **Clarity**: Separates concerns (global properties vs. tensor-specific properties)

#### Optional Signature Fields

Several fields in the signature are optional:
- **`direction`**: Defaults to `FORWARD` if not specified, reducing boilerplate for the common case
- **Tensor `data_type`**: Falls back to signature's default, allowing mixed-precision with minimal specification
- **Tensor `operation`**: Defaults to `PASS_THROUGH`, supporting both fused and non-fused operations with the same interface

This design follows the principle of "make the common case simple, the complex case possible."

## Convolution Algorithm

## Convolution Factory

Convolution factory builds the instance based on the convolution signature and convolution algorithm.
The signature and the algorithm descriptions are dispatched to the relevant algorithm specific factory for instance creation. The convolution factory design is described in a separate [Readme](factory/README.md).
