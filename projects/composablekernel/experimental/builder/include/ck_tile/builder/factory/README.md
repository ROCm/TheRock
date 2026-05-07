# Convolution Builder Factory Directory

This directory implements compile-time dispatch from high-level signature algorithm descriptors to our exisitng specialized convolution kernel implementations.

See the [main builder documentation](../README.md) for an overview.

## Design Overview

The factory system operates in two phases:

1. **Algorithm Classification**: The function `make_conv_instance` in  `conv_dispatcher.hpp` inspects the signature and algorithm descriptors to determine which kernel variant they satisfy (XDL V3, XDL, WMMA, DL, or Large Tensor)

2. **Factory Instantiation**: Each factory (`conv_fwd_*_factory.hpp`) transforms builder descriptors into CK device operation template parameters and instantiates the corresponding kernel device operation.

## Key Files

- **`conv_dispatcher.hpp`**: Entry point with `make_conv_instance()` function. Contains dispatch logic and algorithm classification predicates. **Start here** to understand the overall flow.

- **`conv_fwd_*_factory.hpp`**: Individual factories for each kernel variant. Each extracts configuration from descriptors, validates parameters, and instantiates the underlying CK device operation.

- **`helpers/`**: Transformation utilities that map builder types to CK device operation parameters (layouts, data types, elementwise ops, block configurations, etc.)

## Usage

```cpp
#include "ck_tile/builder/factory/conv_dispatcher.hpp"

using Factory = decltype(make_conv_instance<signature, algorithm, "v1">());
```

The dispatcher automatically selects the appropriate factory following explicit logic.
