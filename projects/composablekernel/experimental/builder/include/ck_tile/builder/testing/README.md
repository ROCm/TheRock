# CK-Builder Testing Utilities

This directory contains testing utilities designed to simplify the process of writing unit tests for GPU kernels built with `ck_tile::builder`. These utilities enable a clean, expressive **Given-When-Then** (Given-When-Then) testing pattern that separates test setup, execution, and validation.

See the [main builder documentation](../README.md) for an overview of the CK-Builder API components.

## Overview

Testing GPU kernels typically involves significant boilerplate: allocating device memory, initializing test data, launching kernels, and validating results. The utilities in this directory abstract away these repetitive tasks, allowing you to focus on defining test cases and verifying correctness.

The core components are:

- **`Args`**: A struct template that holds runtime parameters for a specific test case.
- **`Input`** and **`Output`**: Helper classes that groups operation inputs and outputs.
- **`Validator`**: A utility that performs on-GPU validation and integrates with GoogleTest/GoogleMock.

Together, these components enable a structured approach to kernel testing that mirrors the Given-When-Then pattern commonly used in behavior-driven development.

## The Given-When-Then Testing Pattern

The Given-When-Then pattern organizes tests into three distinct phases:

1. **Given**: Set up the preconditions and test data
2. **When**: Execute the action being tested
3. **Then**: Verify the expected outcome

This structure makes tests easier to read, write, and maintain. Each phase has a clear purpose, and the testing utilities are designed to support this workflow.

### Given: Defining the Test Case

The "Given" phase establishes the context for your test. This includes both the compile-time characteristics of the kernel and the runtime parameters for the specific test case.

#### Operation Signature

The "signature" defines the **mathematical contract** that the kernel must satisfy. It specifies compile-time properties such as:

- Spatial dimensionality (1D, 2D, or 3D)
- Convolution direction (Forward, Backward Data, Backward Weight)
- Tensor memory layout (e.g., NHWC, NCHW)
- Data types (FP32, FP16, BF16, etc.)
- Fused element-wise operations (e.g., Bias, ReLU)

The format of the signature struct is enforced at compile time using C++20 concepts by the CK-Builder API, ensuring type safety and enabling compile-time optimizations. The design of these concepts and the required constraints are discussed in the [CK Builder design description](../include/ck_tile/builder/README.md).

```cpp
// Define our custom signature struct.
struct ConvSignature {
    int spatial_dim = 2;
    ck_tile::builder::ConvDirection direction =
        ck_tile::builder::ConvDirection::FORWARD;
    ck_tile::builder::GroupConvLayout2D layout =
        ck_tile::builder::GroupConvLayout2D::NHWGC_GKYXC_NHWGK;
    ck_tile::builder::DataType data_type =
        ck_tile::builder::DataType::FP16;
    ck_tile::builder::ElementwiseOperation elementwise_operation =
        ck_tile::builder::ElementwiseOperation::NONE;
};

// Double-check that out structure is well-defined according to the CK-Builder API.
static_assert(ck_tile::builder::ConvSignatureDescriptor<ConvSignature>);

// Instantiate the signature with a configuration. These values are again checked
// by the CK-Builder API when a device operation is built.
constexpr auto SIGNATURE = ConvSignature{
    .spatial_dim = 2,
    .direction = ck_tile::builder::ConvDirection::FORWARD,
    .layout = ck_tile::builder::GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
    .data_type = ck_tile::builder::DataType::FP16,
    .elementwise_operation = ck_tile::builder::ElementwiseOperation::NONE,
};
```

#### Run-time Arguments

The `Args` struct template provides the **runtime parameters** for your test case. It is parameterized by the `SIGNATURE` and contains fields for tensor dimensions, strides, dilations, and other dynamic properties. Note that the exact parameters required for each `Args` depends on the `SIGNATURE`: For example, a `SIGNATURE` that represents a forward convolution requires specifying the number of batches, groups, input- and output-channels, filter dimensions, filter strides, and so on. A `SIGNATURE` that represents a simple GEMM operation may instead require only the dimensions of the A-, B- and C-matrices.

```cpp
    ck_tile::builder::test::Args<SIGNATURE> args = {
    .lengths = {
        .batch_size      = 128,
        .groups          = 1,
        .input_channels  = 64,
        .output_channels = 128,
        .image           = {.height = 56, .width = 56},
        .filter          = {.height = 3,  .width = 3},
    },
    .filter_strides  = {.height = 1, .width  = 1},
    .filter_dilation = {.height = 1, .width  = 1},
    .input_left_pad  = {.width  = 1, .height = 1},
    .input_right_pad = {.width  = 1, .height = 1},
};
```

#### Tensor Memory Management

Tensor memory is passed using the `Inputs<SIGNATURE>` and `Outputs<SIGNATURE>` structures. These group all inputs and outputs for an operation. Note that these structures do not "own" the memory inside: They only logically group the inputs so that they can be passed as a common type. The amount of inputs and outputs may differ depending on the `SIGNATURE`, and this avoids having to pass additional values and accept additional parameters in those situations.

The exact fields in `Inputs` and `Outputs` depend again on the particular `SIGNATURE` that they are constructed with. In general, these structures are intended to be freely constructible from external data and only serve to group relevant information. Automatic memory management can be performed using the `UniqueInputs<SIGNATURE>` and `UniqueOutputs<SIGNATURE>` structures instead. The `alloc_inputs` and `alloc_outputs` functions are used to initialize these types: They take an `Args` structure and allocate the appropriate amounts of memory. `.get()` is used to return an instance of the appropriate `Input` or `Output`.

```cpp
auto inputs = ck_tile::builder::test::allocate_inputs(args);
auto outputs = ck_tile::builder::test::allocate_outputs(args);
```

Note that these functions merely _allocate_ memory: After allocation, the memory is still uninitialized.

#### Tensor Memory Initialization

Operation inputs can be initialized by using `ck_tile::builder::test::init_inputs()`. Crucially, this operation accepts _all_ inputs, as well as the `args` structure. This is because initializing tensor memory is a context-dependent operation: We need to understand the operation in detail in order to generate inputs which do not overflow, do not generate NaNs or all zeros, etc. Passing the `args` allows `init_inputs` to generate a good test for the operation at hand.

### When: Executing the Kernel

The "When" phase is where the kernel to be tested is actually executed. This involves selecting an algorithm and using the `Builder` to generate the kernel.

#### Operation Algorithm

The "algorithm" defines the **implementation strategy** for the kernel. It specifies low-level details such as:

- Thread block dimensions and tile sizes
- GEMM implementation (XDL or WMMA)
- Data transfer vectorization
- Pipeline scheduling

As with the signature struct, the format of the algorithm struct is enforced at compile time using C++20 concepts by the CK-Builder API. The design of these concepts and the required constraints are discussed in the [CK Builder factory design description](../include/ck_tile/builder/factory/README.md).


```cpp
// Define our custom algorithm struct.
struct ConvAlgorithm {
    // Thread block configuration
    ThreadBlock thread_block;

    // Gridwise GEMM configuration
    GridwiseXdlGemm gridwise_gemm;

    // Block transfer configuration
    Transfer transfer;

    // Additional tuning parameters
    // ...
};

// Double-check that our algorithm is well-defined according to the CK-Builder API.
static_assert(ck_tile::builder::ConvAlgorithmDescriptor<ConvAlgorithm>);

// Instantiate the algorithm with a configuration. Like with the signature struct
// the CK-Builder API will check that the values are correct when a device
// operation is built.
constexpr auto ALGORITHM = ConvAlgorithm{
    .thread_block = /* ... */;
    .gridwise_gem = /* ... */;
    .transfer = /* ... */;
    // ...
};
```

#### Building the Kernel

The `Builder` combines the signature (what to compute) with the algorithm (how to compute it) to generate a kernel type which represents the operation. The implementation details, including invocation method, depend on the particular signature and algorithm.

```cpp
using Conv = ck_tile::builder::ConvBuilder<SIGNATURE, ALGORITHM>::Instance;
auto conv = Conv{};
```

#### Invoking the Kernel

After creating the kernel instance, it can be invoked by passing the instance, the arguments, the inputs, and the outputs to `run()`. This operation writes results into the buffers in `outputs`.

```cpp
ck_tile::builder::test::run(conv, args, inputs.get(), outputs.get());
```

### Then: Verifying the Results

The "Then" phase validates that the kernel produced the expected output. This is done by running a reference kernel and comparing the results.

#### Building the Reference Kernel

The reference kernel is just another kernel instance of the builder, one that's been externally verified to produce the correct results. As this kernel is also running on the GPU, we can use it to perform tests far more quickly than when comparing the outputs to a CPU-based reference implementation.

In order to obtain an instance of the reference kernel, the correct `ALGORITHM` needs to be passed to the `Builder`.

```cpp
struct ReferenceAlgorithm {
    ck_tile::builder::ConvAlgorithmSpecialization specialization;
};
static_assert(ck_tile::builder::ConvAlgorithmDescriptor<ReferenceAlgorithm>);
constexpr auto REFERENCE_ALGORITHM = ReferenceAlgorithm{
    .specialization = ck_tile::builder::ConvAlgorithmSpecialization::REFERENCE;
};
using ReferenceConv = ck_tile::builder::ConvBuilder<SIGNATURE, REFERENCE_ALGORITHM>::Instance;
auto reference_conv = ReferenceConv{};
```

This instance can then be invoked using `ck_tile::builder::test::run()`, the same as the kernel to be tested. Note that another instance of the `Outputs` structure needs to be passed here in order to store the results.

```cpp
auto reference_outputs = ck_tile::builder::test::allocate_outputs(args);
ck_tile::builder::test::run(conv, args, inputs.get(), reference_outputs.get());
```

#### `Validator<SIGNATURE>`

The `Validator` class encapsulates the validation logic. It performs on-GPU correctness checks by comparing two instances of the `Outputs` structure.

```cpp
ck_tile::builder::test::Validator<SIGNATURE> validator(outputs.get(), reference_outputs.get());
```

The `Validator` provides methods that return GoogleMock matchers, enabling clean integration with GoogleTest:

```cpp
EXPECT_THAT(validator.result(), validator.matches_reference_output());
```

The `matches_reference_output()` matcher checks that the output is numerically correct within acceptable tolerances. The `Validator` can also provide more detailed diagnostics, such as:

- Maximum absolute error
- Maximum relative error
- Number of mismatched elements
- Specific locations of errors

## Complete Example

Here's a complete test that demonstrates the Given-When-Then pattern:

```cpp
#include <gtest/gtest.h>
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_builder.hpp"
#include "ck_tile/testing/tensor_memory_manager.hpp"
#include "ck_tile/testing/validator.hpp"

// Define the convolution signature
struct ConvSignature {
    int spatial_dim = 2;
    ck_tile::builder::ConvDirection direction =
        ck_tile::builder::ConvDirection::FORWARD;
    ck_tile::builder::GroupConvLayout2D layout =
        ck_tile::builder::GroupConvLayout2D::NHWGC_GKYXC_NHWGK;
    ck_tile::builder::DataType data_type =
        ck_tile::builder::DataType::FP16;
    ck_tile::builder::ElementwiseOperation elementwise_operation =
        ck_tile::builder::ElementwiseOperation::NONE;
};
static_assert(ck_tile::builder::ConvSignatureDescriptor<ConvSignature>);
constexpr auto SIGNATURE = ConvSignature{
    .spatial_dim = 2,
    .direction = ck_tile::builder::ConvDirection::FORWARD,
    .layout = ck_tile::builder::GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
    .data_type = ck_tile::builder::DataType::FP16,
    .elementwise_operation = ck_tile::builder::ElementwiseOperation::NONE,
};

// Define the convolution algorithm
struct ConvAlgorithm {
    // Algorithm configuration details...
    // (Omitted for brevity)
};
static_assert(ck_tile::builder::ConvAlgorithmDescriptor<ConvAlgorithm>);
constexpr auto ALGORITHM = ConvAlgorithm{/* ... */};

// Define the reference convolution algorithm
struct ReferenceAlgorithm {
    ck_tile::builder::ConvAlgorithmSpecialization specialization;
};
static_assert(ck_tile::builder::ConvAlgorithmDescriptor<ReferenceAlgorithm>);
constexpr auto REFERENCE_ALGORITHM = ReferenceAlgorithm{
    .specialization = ck_tile::builder::ConvAlgorithmSpecialization::REFERENCE;
};

// The actual test
TEST(ConvolutionTest, Forward2D_FP16) {
    // ===== GIVEN: Set up the test case =====

    // Define runtime parameters
    ck_tile::builder::test::Args<ConvSignature> args = {
        .lengths = {
            .batch_size      = 128,
            .groups          = 1,
            .input_channels  = 64,
            .output_channels = 128,
            .image           = {.height = 56, .width = 56},
            .filter          = {.height = 3,  .width = 3},
        },
        .filter_strides  = {.height = 1, .width  = 1},
        .filter_dilation = {.height = 1, .width  = 1},
        .input_left_pad  = {.width  = 1, .height = 1},
        .input_right_pad = {.width  = 1, .height = 1},
    };

    // Allocate GPU memory
    auto inputs = ck_tile::builder::test::allocate_inputs(args);
    auto outputs = ck_tile::builder::test::allocate_outputs(args);
    auto reference_outputs = ck_tile::builder::test::allocate_outputs(args);

    // Initialize inputs
    ck_tile::builder::test::init_inputs(args, inputs);

    // ===== WHEN: Execute the kernel =====

    // Build the kernel
    using Conv = ck_tile::builder::ConvBuilder<SIGNATURE, ALGORITHM>::Instance;
    auto conv = Conv{};

    // Compute actual results
    ck_tile::builder::test::run(conv, args, inputs.get(), outputs.get());

    // ===== THEN: Verify the results =====

    // Build the reference kernel
    using ReferenceConv = ck_tile::builder::ConvBuilder<SIGNATURE, REFERENCE_ALGORITHM>::Instance;
    auto reference_conv = ReferenceConv{};

    // Compute reference results
    ck_tile::builder::test::run(conv, args, inputs.get(), reference_outputs.get());

    // Check the results
    ck_tile::builder::test::Validator<SIGNATURE> validator(outputs.get(), reference_outputs.get());
    EXPECT_THAT(validator.result(), validator.is_ok());
}
```

## Benefits of This Approach

1. **Clarity**: The Given-When-Then structure makes tests self-documenting. Each phase has a clear purpose.

2. **Reduced Boilerplate**: The utilities handle memory management, initialization, and validation, eliminating repetitive code.

3. **Type Safety**: The use of C++20 concepts ensures that signatures and algorithms are well-formed at compile time.

4. **Flexibility**: The `Args` struct can be easily extended to support different test scenarios, `Inputs` and `Outputs` can be modified to support additional tensors where necessary, and alternatives to `init_inputs()` can be provided to support additional testing strategies.

5. **Integration**: The `Validator` integrates seamlessly with GoogleTest/GoogleMock, providing familiar assertion syntax.

6. **Maintainability**: Changes to the testing infrastructure are localized to the utility classes, not scattered across individual tests.

## Future Enhancements

Potential improvements to the testing utilities include:

- Performance benchmarking utilities
- Automatic test case generation from parameter ranges
- Enhanced error reporting with visual diffs
- Support for multi-GPU testing scenarios
