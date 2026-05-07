// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file
/// @brief Provides a base class for generating human-readable descriptions of kernel instances.
///
/// This file contains the Description base class that defines a common interface for
/// all descriptor types. Derived classes implement specific formatting and explanation
/// logic for different kernel types (e.g., convolution, GEMM, etc.).

#pragma once

#include <string>

namespace ck_tile::reflect {

/// @brief Base class for generating human-readable descriptions of kernel instances
/// Defines a common interface for all descriptor types with methods for generating
/// descriptions at various levels of detail.
class Description
{
    public:
    Description()                              = default;
    Description(const Description&)            = default;
    Description(Description&&)                 = default;
    Description& operator=(const Description&) = default;
    Description& operator=(Description&&)      = default;
    /// @brief Virtual destructor for proper cleanup of derived classes
    virtual ~Description() = default;

    /// @brief Generate a brief one-line summary
    /// @return A concise description of the kernel configuration
    virtual std::string brief() const = 0;

    /// @brief Generate a detailed hierarchical description
    /// @return A multi-line tree-formatted description covering all configuration details
    virtual std::string detailed() const = 0;

    /// @brief Generate a string representation of the instance
    /// @return A string that represents the instance
    virtual std::string instance_string() const = 0;
};

/// @brief A specialized Description that only supports instance_string()
/// This is a helper class for kernels that don't yet have full ConvDescription support.
/// The brief() and detailed() methods return "not supported" placeholders.
class InstanceStringDescription : public Description
{
    public:
    /// @brief Construct with an instance string
    /// @param instance The instance string to store
    explicit InstanceStringDescription(std::string instance) : instance_(std::move(instance)) {}

    /// @brief Returns "not supported" as brief descriptions are not implemented
    /// @return A placeholder string indicating the feature is not supported
    std::string brief() const override { return "not supported"; }

    /// @brief Returns "not supported" as detailed descriptions are not implemented
    /// @return A placeholder string indicating the feature is not supported
    std::string detailed() const override { return "not supported"; }

    /// @brief Returns the stored instance string
    /// @return The instance string provided during construction
    std::string instance_string() const override { return instance_; }

    private:
    std::string instance_; ///< The stored instance string
};

} // namespace ck_tile::reflect
