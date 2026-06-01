// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>

#include "ck_tile/builder/testing/validation.hpp"

/// This file is the main header for the CK-Builder testing system. A high-level
/// description of this testing system is documented in
/// `ck_tile/builder/testing/README.md`. This file deals mainly deals with the
/// documentation of the implementation details by forward-declaring and documenting
/// the relevant types.
///
/// The intention is that the basic testing strategy (explained in the testing
/// documentation) is available for every different type of device operation. This
/// requires us to provide some implementations in two fronts: Support for the
/// Args, Inputs, Outputs, UniqueInputs, and UniqueOutputs for all SIGNATUREs which
/// are supported by CK Builder, and support for invoking the different
/// implementations returned by CK Builder, depending on the Algorithm.
///
/// Different SIGNATUREs may require different arguments and different (amounts of)
/// input/output tensors. Rather than trying to cram all this in the same structure,
/// or to provide different types, we will use dependent typing to specialize the
/// implementation for the SIGNATURE at hand. For this reason, the Args, Inputs,
/// Outputs, UniqueInputs, and UniqueOutputs structures are all parameterized by the
/// SIGNATURE. The idea is to use C++20 concepts to limit the specialization to the
/// subset of SIGNATUREs that conceptually make sense for that implementation. For
/// example, to provide an implementation of the testing framework for forward
/// convolutions, we can use a concept to check whether the SIGNATURE is a valid
/// forward convolution signature:
///
///     template <auto SIGNATURE>
///         requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
///     struct Args<SIGNATURE> { ... }; // Similar for the other types
///
/// Invocation of instances is another matter: The Builder may return instances from
/// either CK or CK-Tile depending on the ALGORITHM configuration. The only place
/// where this matters is the implementation of `run()`, which needs to provide a
/// custom implementation for all instances which the Builder may return, including
/// the reference implementation. The strategy is the same here: Use concepts to
/// check whether the instance returned by the builder is of a particular type, and
/// overload the `run()` function for that concept:
///
///     template <auto SIGNATURE, typename Conv>
///         requires
///             // Check that the SIGNATURE is of the type that we expect
///             ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE> &&
///             // Also check that the instance is of a type which we can invoke here
///             IsCkConvInstance<SIGNATURE, Conv>
///     void run(Conv& conv, ...);
///
/// Note that this is only the suggested strategy; you may also use `if constexpr`
/// or similar to dispatch the correct implementation of the instance in the
/// implementation of the `run()` function for a particular group of device
/// operations.
///
/// The remainder of this file describes the types and functions that should be
/// overloaded for a particular device operation, and in which situation.

namespace ck_tile::builder::test {

/// @brief Run-time arguments corresponding to a signature.
///
/// The `Args` structure is the main point of runtime configuration for a device
/// operation. Depending on the SIGNATURE, it is used to provide the run-time
/// parameters for a device operation, for instance, for the tensor dimensions,
/// tensor strides, parameters such as padding, split-K batch size, fused
/// element-wise operator instances, etc. In short, a complete run-time
/// configuration of the tensor operation at hand.
///
/// This structure does not require additional member functions, any which are
/// provided should be considered implementation details of Args structure for
/// that particular SIGNATURE.
///
/// @note A good indicator of the fields necessary here are the values that should
/// be passed to the CK `MakeArgument()` function or CK-Tile `HostArgs` structure
/// of the device operation that you are trying to implement. It is the intention
/// that this structure is an aggregrate so that it can be initialized using C++20
/// designated initializers to keep the tests readable.
///
/// @tparam SIGNATURE The signature to specialize the structure for.
template <auto SIGNATURE>
struct Args;

/// @brief Non-owning input collection corresponding to a signature.
///
/// The `Input` structure represents the collection of input tensor data on the
/// device, associated to a particular SIGNATURE. The exact fields in this structure
/// may again depend on the exact SIGNATURE. This structure is non-owning: its use
/// is intended as a way to pass all inputs around as a single value.
///
/// This structure does not require additional member functions, any which are
/// provided should be considered implementation details of Args structure for
/// that particular SIGNATURE.
///
/// @note The implementation can just be a set of void-pointers which conceptually
/// represent the inputs of the device operation. It is the intention that this
/// structure is an aggregrate so that it can be initialized using C++20
/// designated initializers to keep the tests readable.
///
/// @tparam SIGNATURE The signature to specialize the structure for.
template <auto SIGNATURE>
struct Inputs;

/// @brief Non-owning outputs collection corresponding to a signature.
///
/// The `Output` structure represents the collection of input tensor data on the
/// device, associated to a particular SIGNATURE. The exact fields in this structure
/// may again depend on the exact SIGNATURE. This structure is non-owning: its use
/// is intended as a way to pass all outputs around as a single value.
///
/// This structure does not require additional member functions, any which are
/// provided should be considered implementation details of Args structure for
/// that particular SIGNATURE.
///
/// @note The implementation can just be a set of void-pointers which conceptually
/// represent the outputs of the device operation. It is the intention that this
/// structure is an aggregrate so that it can be initialized using C++20
/// designated initializers to keep the tests readable.
///
/// @tparam SIGNATURE The signature to specialize the structure for.
template <auto SIGNATURE>
struct Outputs;

/// @brief RAII-enabled inputs collection corresponding to a signature.
///
/// The `UniqueInputs` is used to automatically manage the memory of a set of
/// inputs. Unlike the corresponding `Inputs` structure, the implementation is
/// opaque; the only requirements for this structure is that an instance can
/// be created using `alloc_inputs()` and that an instance of the corresponding
/// `Inputs` structure can be obtained using `.get()`.
///
/// @note The easiest way to implement this type is to use the `DeviceBuffer`
/// type to allocate individual device buffers for each input tensor.
///
/// @tparam SIGNATURE The signature to specialize the structure for.
///
/// @see alloc_inputs()
/// @see ValidUniqueInputs
/// @see DeviceBuffer
template <auto SIGNATURE>
struct UniqueInputs;

/// @brief RAII-enabled outputs collection corresponding to a signature.
///
/// The `UniqueOutputs` is used to automatically manage the memory of a set of
/// outputs. Unlike the corresponding `Outputs` structure, the implementation is
/// opaque; the only requirements for this structure is that an instance can
/// be created using `alloc_outputs()` and that an instance of the corresponding
/// `Outputs` structure can be obtained using `.get()`.
///
/// @note The easiest way to implement this type is to use the `DeviceBuffer`
/// type to allocate individual device buffers for each output tensor.
///
/// @tparam SIGNATURE The signature to specialize the structure for.
///
/// @see alloc_outputs()
/// @see ValidUniqueOutputs
/// @see DeviceBuffer
template <auto SIGNATURE>
struct UniqueOutputs;

/// @brief Concept to check the validity of `UniqueInputs`.
///
/// The `ValidUniqueInputs` concept can be used to check whether the definition
/// of `UniqueInputs` is valid for a particular SIGNATURE.
///
/// - SIGNATURE is signature to specialize the structure for.
///
/// @see UniqueInputs
template <auto SIGNATURE>
concept ValidUniqueInputs = requires(UniqueInputs<SIGNATURE>& inputs) {
    /// `.get()` is used to obtain a non-owning version of the `Inputs` collection.
    { inputs.get() } -> std::convertible_to<Inputs<SIGNATURE>>;
};

/// @brief Concept to check the validity of `UniqueOutputs`.
///
/// The `ValidUniqueOutputs` concept can be used to check whether the definition
/// of `UniqueOutputs` is valid for a particular SIGNATURE.
///
/// - SIGNATURE is signature to specialize the structure for.
///
/// @see UniqueOutputs
template <auto SIGNATURE>
concept ValidUniqueOutputs = requires(UniqueOutputs<SIGNATURE>& inputs) {
    /// `.get()` is used to obtain a non-owning version of the `Outputs` collection.
    { inputs.get() } -> std::convertible_to<Outputs<SIGNATURE>>;
};

/// @brief Allocate inputs corresponding to a signature.
///
/// The `alloc_inputs()` function is used to create an instance of
/// `UniqueInputs`. This function uses the `args` structure to compute the
/// amount of memory required and then allocate it on the device, for example
/// using `alloc_buffer` or `alloc_tensor_buffer`.
///
/// @tparam SIGNATURE The signature to specialize the structure for.
///
/// @param args The run-time arguments of the operation.
///
/// @see Inputs
/// @see UniqueInputs
/// @see alloc_buffer()
/// @see alloc_tensor_buffer()
template <auto SIGNATURE>
    requires ValidUniqueInputs<SIGNATURE>
UniqueInputs<SIGNATURE> alloc_inputs(const Args<SIGNATURE>& args);

/// @brief Allocate inputs corresponding to a signature.
///
/// The `init_inputs()` function is used to initialize pseudo-random data
/// to the tensors specified in the Inputs structure. Implementors should
/// fill each of the tensors in `inputs` with appropriate random data.
///
/// @tparam SIGNATURE the signature to specialize the structure for.
///
/// @param args The run-time arguments of the operation.
/// @param inputs The operation inputs to initialize with random data.
///
/// @note This function is explicitly deleted to generate compile errors
/// for missing implementations.
///
/// @see Inputs
/// @see tensor_initialization
template <auto SIGNATURE>
void init_inputs(const Args<SIGNATURE>& args, Inputs<SIGNATURE> inputs) = delete;

/// @brief Allocate outputs corresponding to a signature.
///
/// The `alloc_outputs()` function is used to create an instance of
/// `UniqueOutputs`. This function uses the `args` structure to compute the
/// amount of memory required and then allocate it on the device, for example
/// using `alloc_buffer` or `alloc_tensor_buffer`.
///
/// @tparam SIGNATURE The signature to specialize the structure for.
///
/// @param args The run-time arguments of the operation.
///
/// @note This function is explicitly deleted to generate compile errors
/// for missing implementations.
///
/// @see Outputs
/// @see UniqueOutputs
/// @see alloc_buffer()
/// @see alloc_tensor_buffer()
template <auto SIGNATURE>
    requires ValidUniqueOutputs<SIGNATURE>
UniqueInputs<SIGNATURE> alloc_outputs(const Args<SIGNATURE>& args) = delete;

/// @brief Compare device operation outputs.
///
/// This function implements the main comparison functionality, used to compare
/// the output of one implementation for a particular `SIGNATURE` with that of
/// another. Usually, the `expected` output should be computed by a reference
/// implementation.
///
/// The implementation of this function generates a "report", which includes
/// detailed information about which tensors are different, how many elements
/// were incorrect, and where (a subset of) those elements are located within
/// the tensor. See `ValidationReport` for more information about the report.
///
/// @tparam SIGNATURE The signature to specialize the structure for.
///
/// @param args The run-time arguments of the operation.
/// @param actual The actual results, the results of the operation to-be-tested.
/// @param expected The expected results, the results of the reference implementation.
///
/// @note This function is explicitly deleted to generate compile errors
/// for missing implementations.
///
/// @see ValidationReport
template <auto SIGNATURE>
ValidationReport validate(const Args<SIGNATURE>& args,
                          Outputs<SIGNATURE> actual,
                          Outputs<SIGNATURE> expected) = delete;

/// @brief Invoke a device operation created by CK Builder.
///
/// This is the main function used to invoke a particular device operation
/// instance created by the builder. It uses the `args`, `inputs`, and `outputs`
/// to configure the `operation` and invokes it immediately.
///
/// In practice, the `Operation` is usually a CK or CK Tile device operation
/// type, for example `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3`.
/// This function implements the required functionality to invoke any relevant
/// type created by the builder.
///
/// @note Unlike the Args, Inputs, Outputs, and related structures, this function
/// is specialized for the different implementations that the builder may
/// return (see file-level documentation).
///
/// @pre The tensors in `inputs` should be allocated and initialized with the
///   appropriate values to perform the operation.
/// @pre The tensors in `outputs` should be allocated.
/// @post The tensors in `outputs` are overwritten with the outputs of the device
///   operation.
///
/// @tparam SIGNATURE The signature to specialize this function for
/// @tparam Operation the kernel of the operation to invoke. This type should be
///   one that is created using the Builder API.
/// @param operation An instance of the operation to invoke.
/// @param args The run-time arguments of the operation.
/// @param inputs The input tensor data. Will not be modified by this function.
/// @param outputs The output tensor data. The contents will be overwritten by
///   this function.
///
/// @note This function is explicitly deleted to generate compile errors
/// for missing implementations.
template <auto SIGNATURE, typename Operation>
void run(Operation& operation,
         const Args<SIGNATURE>& args,
         const Inputs<SIGNATURE>& inputs,
         const Outputs<SIGNATURE>& outputs) = delete;

} // namespace ck_tile::builder::test
