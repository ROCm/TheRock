// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

#include "ck_tile/builder/factory/conv_dispatcher.hpp"
#include "ck_tile/builder/versions.hpp"

namespace ck_tile::builder {

/**
 * @brief Top-level builder for creating convolution kernel instances.
 *
 * This struct serves as the main entry point for generating a convolution kernel.
 * It uses a dispatcher function based on the provided signature, algorithm, and version
 * to construct the appropriate kernel instance.
 *
 * @tparam SIGNATURE The convolution signature, which describes the mathematical functionality of
 * the algorithm (e.g., data types, layouts, direction).
 * @tparam ALGORITHM The specific convolution algorithm to be used for the implementation.
 * @tparam VERSION The version of the builder implementation.
 */
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION = LATEST_API_VERSION>
    requires SupportedVersion<VERSION> && ValidConvSignature<SIGNATURE>
struct ConvBuilder
{
    static constexpr auto kVersion = VERSION;
    // Output: The kernel class instance created via the dispatcher.
    using Instance = decltype(factory::make_conv_instance<SIGNATURE, ALGORITHM, VERSION>());
};

} // namespace ck_tile::builder
