// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile-time dispatcher for convolution kernel instantiation.
//
// This header provides a centralized factory dispatch mechanism that routes algorithm
// specifications to appropriate convolution kernel implementations at compile-time.
//
// ## Design Overview
//
// The dispatcher operates in two phases:
// 1. **Algorithm Identification**: Six `consteval` predicate functions (`IsReferenceAlgorithm`,
//    `IsXdlV3Algorithm`, `IsXdlAlgorithm`, `IsWmmaAlgorithm`, `IsDlAlgorithm`,
//    `IsLargeTensorAlgorithm`) inspect the algorithm descriptor's structure to determine which
//    kernel variant it satisfies. Each predicate checks a specific set of concept constraints
//    that define a kernel variant.
//
// 2. **Factory Routing**: The main `make_conv_instance()` function uses `if constexpr`
//    to dispatch to the appropriate factory class based on both the convolution direction
//    and the identified algorithm type. All routing decisions occur at compile-time,
//    ensuring zero runtime overhead.
//
// ## Supported Kernel Variants
//
// - **Reference**: Simple reference implementation for validation. Only requires a specialization
//   field set to ConvAlgorithmSpecialization::REFERENCE.
//
// - **XDL V3**: Newer XDL-based pipeline using block GEMM structure. Requires fewer parameters
//   than standard XDL (e.g., uses `SpecifiesBlockGemm` instead of scheduling/prefetch configs).
//
// - **XDL**: Standard XDL-based kernel using AMD XDLops hardware instructions for matrix
//   multiply. Requires full scheduling configuration including prefetch stages and loop scheduler.
//
// - **WMMA**: Wavefront Matrix-Matrix Accumulate variant optimized for WMMA-capable hardware.
//   Requires similar configuration to XDL.
//
// - **DL**: Specialized vectorized dot-product kernel optimized for specific data layouts
//   (NHWC/KYXC/NHWK). The "DL" label just indicates this does not use XDLops instructions.
//
// - **Large Tensor**: XDL-based kernel with extended tensor support. Wraps a base XDL algorithm
//   and adds large tensor capabilities.
//
// ## Current Limitations
//
// Currently only forward convolution is supported. Backward data and backward weight convolution
// directions will fail at compile-time with informative static_assert messages.
//
// ## Usage Example
//
// ```
// auto kernel = make_conv_instance<SIGNATURE, ALGORITHM>();
// ```

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"

// Include all factory implementations
#include "ck_tile/builder/factory/conv_fwd_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_wmma_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_dl_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_large_tensor_factory.hpp"
#include "ck_tile/builder/factory/reference_factory.hpp"
#include "ck_tile/builder/factory/conv_tile_factory.hpp"

namespace ck_tile::builder::factory {

// This dispatch logic is rigid and confusing for users. Further, hides most of
// the great error messages from our concepts.
//
// Requirements for a good design:
// 1. Fall through is bad: inputs should get directly to an implementation
//    if we are going to have good compiler errors.
// 2. Logic should be easy for library users to understand.
// 3. Logic should be easy to test, maintain, and extend.
//
// We should probably add explicit tags to the algorithm descriptors, at least
// for the initial implemenation.
//
// To avoid changing behavior too much during refactoring, we leave the explicit
// dispatch logic here for now, just changing it from SFINAE to consteval + if constexpr.
// There may be some subtle behavior changes, but build failure messages will be more
// clear.
//
// TODO: Make this dispatch logic much more robust and clear for users.

// Reference algorithm (simplest implementation for validation)
template <typename T>
concept IsReferenceAlgorithm = ConvAlgorithmDescriptor<T> && requires {
    { T::specialization } -> std::convertible_to<ConvAlgorithmSpecialization>;
    requires T::specialization == ConvAlgorithmSpecialization::REFERENCE;
};

// CK Tile kernel
template <typename T>
concept IsTileAlgorithm = ConvAlgorithmDescriptor<T> && SpecifiesTileThreadBlock<T> &&
                          SpecifiesTileTransfer<T> && SpecifiesTileConvSpecialization<T> &&
                          SpecifiesTileBlockGemm<T> && SpecifiesTileOptimizations<T>;

// XDL-based kernel with V3 pipeline structure (newer block GEMM pipeline)
template <typename T>
concept IsXdlV3Algorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesGridwiseXdlGemm<T> &&
    SpecifiesBlockTransfer<T> && SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesFwdConvSpecialization<T> &&
    SpecifiesGemmSpecialization<T> && SpecifiesBlockGemm<T>;

// Standard XDL-based kernel (uses XDLops hardware instructions for matrix multiply)
template <typename T>
concept IsXdlAlgorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesGridwiseXdlGemm<T> &&
    SpecifiesBlockTransfer<T> && SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesFwdConvSpecialization<T> &&
    SpecifiesGemmSpecialization<T> && SpecifiesNumPrefetchStages<T> &&
    SpecifiesNumGroupsToMerge<T> && SpecifiesLoopScheduler<T>;

// WMMA-based kernel (uses Wavefront Matrix-Matrix Accumulate instructions)
template <typename T>
concept IsWmmaAlgorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesGridwiseWmmaGemm<T> &&
    SpecifiesBlockTransfer<T> && SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesFwdConvSpecialization<T> &&
    SpecifiesGemmSpecialization<T> && SpecifiesNumPrefetchStages<T> && SpecifiesLoopScheduler<T>;

// Specialized DL kernel for specific NHWC/KYXC/NHWK data layouts
template <typename T>
concept IsDlAlgorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesFwdConvSpecialization<T> &&
    SpecifiesGemmSpecialization<T> && SpecifiesDlThreadConfig<T> && SpecifiesDlThreadCluster<T> &&
    SpecifiesDlBlockTransfer<T> && SpecifiesDlEpilogue<T>;

// XDL-based kernel with large tensor support
template <typename T>
concept IsLargeTensorAlgorithm =
    IsXdlAlgorithm<decltype(T::base_algorithm)> && SpecifiesLargeTensorSupport<T>;

template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
constexpr auto make_conv_instance()
{
    using AlgoType = std::remove_const_t<decltype(ALGORITHM)>;

    // Reference algorithm supports all directions
    if constexpr(IsReferenceAlgorithm<AlgoType>)
    {
        return typename ReferenceFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
    }
    // CK Tile supports common factory for each direction
    else if constexpr(IsTileAlgorithm<AlgoType>)
    {
        return typename ConvTileFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
    }
    // Forward direction (supports most algorithm variants)
    else if constexpr(ConvDirectionIsForward<SIGNATURE>)
    {
        if constexpr(IsXdlV3Algorithm<AlgoType>)
        {
            return typename ConvFwdXdlV3Factory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(IsXdlAlgorithm<AlgoType>)
        {
            return typename ConvFwdXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(IsWmmaAlgorithm<AlgoType>)
        {
            return typename ConvFwdWmmaFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(IsDlAlgorithm<AlgoType>)
        {
            return typename ConvFwdDlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(IsLargeTensorAlgorithm<AlgoType>)
        {
            return typename ConvFwdLargeTensorFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else
        {
            static_assert(
                false,
                "No suitable forward convolution kernel factory found for the provided ALGORITHM. "
                "The ALGORITHM must satisfy requirements for one of: Reference, Tile, XDL V3, XDL, "
                "WMMA, DL (NHWC layout), or Large Tensor variant.");
        }
    }
    // Backward data direction (will expand with more algorithms in the future)
    else if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
    {
        static_assert(false,
                      "Backward data convolution: Only reference and tile algorithms supported "
                      "currently. "
                      "Optimized kernels (XDL, WMMA, etc.) not yet implemented.");
    }
    // Backward weight direction (will expand with more algorithms in the future)
    else if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
    {
        static_assert(false,
                      "Backward weight convolution: Only reference and tile algorithms "
                      "supported currently. "
                      "Optimized kernels (XDL, WMMA, etc.) not yet implemented.");
    }
    else
    {
        static_assert(false,
                      "Invalid or unsupported convolution direction. "
                      "The SIGNATURE must specify a valid ConvDirection: FORWARD, BACKWARD_DATA, "
                      "or BACKWARD_WEIGHT.");
    }
}

} // namespace ck_tile::builder::factory
