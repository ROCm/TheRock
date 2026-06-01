# Convolution Reflection Directory

This directory contains tools for "reflecting" on convolution kernel instances. It allows developers to inspect the compile-time configuration of a kernel and generate detailed, human-readable descriptions.

See the [main builder documentation](../README.md) for an overview.

## Design Overview

The reflection system works by extracting properties from a convolution kernel *type* and formatting them into a string. This is useful for debugging, performance tuning, and generating documentation.

1. **Trait Extraction**: The `ConvTraits` template (in `conv_traits.hpp`) is specialized for each kernel instance. It extracts low-level details like tile sizes, data layouts, and pipeline versions from the kernel's type definition.

2. **Description Generation**: The `describe<Instance>()` function (in `conv_description.hpp`) uses `ConvTraits` to populate a `ConvDescription` (`Description`) object.

3. **Formatting**: The `ConvDescription` class (which implements `Description`) contains methods like `brief()` and `detailed()` that format the extracted properties into well-structured strings for display.

## Key Files

- **`description.hpp`**: The generalized Description base class with no implementation.

- **`conv_description.hpp`**: The main entry point. Contains the `ConvDescription` struct and the `describe()` factory function.
- **`conv_traits.hpp`**: Home of the `ConvTraits` template, which is the core of the property extraction mechanism.
- **`tree_formatter.hpp`**: A simple utility for generating the indented, tree-like format used in the `detailed()` description.

## Usage

To get a description of a convolution kernel instance, use the `describe` function and call one of its formatting methods:

```cpp
#include "ck_tile/builder/reflect/conv_description.hpp"

// Assume MyConvFwdInstance is a type alias for a specific kernel instance
using MyConvFwdInstance = /* ... some kernel type ... */;

// Describe the instance
const auto description = ck_tile::reflect::conv::Describe<MyConvFwdInstance>();

// Print the detailed description
std::cout << description.detailed() << std::endl;
```

## Appendix: Current Limitations

### Supported Instance Types

The reflection system (`ckr::describe`) currently supports the following convolution instance types:

- **Standard XDL Forward Convolution** (`DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle`)
- **Large Tensor XDL Forward Convolution** (`DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor`)
- **V3 XDL Forward Convolution** (`DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3`)

These variants all share similar template parameter structures and are compatible with the current `ConvTraits` implementation.

### Unsupported Instance Types

The following instance types are **not yet supported** by the reflection system:

- **DL (pre-XDL) Variants** (`DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK`)
  - Uses different internal structure with parameters like `K0PerBlock`, `K1`, `M1PerThread`, etc.
  - Missing standard members like `kKPerBlock`, `kMPerXDL`, `kAK1`

- **WMMA Variants** (`DeviceGroupedConvFwdMultipleD_Wmma_CShuffle`)
  - Uses WMMA-specific parameters like `MPerWmma`, `NPerWmma`, `MRepeat`, `NRepeat`
  - Different tile transfer structure incompatible with current `ConvTraits`

- **Backward Weight Convolution** (`DeviceGroupedConvBwdWeight_Xdl_CShuffle`)
  - Uses different layout naming: `InLayout`, `WeiLayout`, `OutLayout` instead of `ALayout`, `BLayout`, `ELayout`
  - Different specialization type: `ConvBackwardWeightSpecialization` vs `ConvForwardSpecialization`
  - Missing several members expected by forward convolution traits

### Future Work

To support these additional instance types, the reflection system would need:

1. Specialized `ConvTraits` templates for each variant type
2. Updated `conv_layout`, `conv_data_type`, and other helper functions to handle different parameter structures
3. Conditional compilation or SFINAE techniques to select the appropriate trait extraction logic based on instance type
4. Customize `ConvDescription` methods for more general kernels.

For now, these unsupported types can still use `GetInstanceString()` through the base class pointer, but cannot use the `ckr::describe` reflection API.
