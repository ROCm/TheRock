// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <source_location>
#include <stdexcept>
#include <sstream>

/// This file defines some utilities for dealing with HIP errors. In the CK-Builder
/// testing code, we'd like to just turn them into exceptions: This cleans up testing
/// code as we don't need to think about returning error codes, but its still much
/// cleaner than just creating a hard crash and thereby possibly interrupting other
/// units in the same test. The testing framework can catch these exceptions where
/// necessary.
///
/// While the exceptions defined in this file are in principle suitable for general
/// usage, HIP functions which return HIP error codes (`hipError_t`) should be
/// checked using the `check_hip` function.

namespace ck_tile::builder::test {

/// @brief Generic HIP exception.
///
/// This is a derivation of `std::runtime_error` which represents a HIP error code.
///
/// @see std::runtime_error
/// @see hipError_t
struct HipError : std::runtime_error
{
    /// @brief Utility for formatting HIP error messages
    ///
    /// Returns a human-readable description of a HIP error. Given a description of the
    /// activity that the user tried to perform, this function appends the HIP-specific
    /// information such as the stringified version of the error code, and the error
    /// code itself (for reference).
    ///
    /// @param user_msg User-given message about the activity at time of error.
    /// @param code The status to report.
    /// @param src The location where this error was discovered.
    static std::string
    format_error(std::string_view user_msg, hipError_t code, std::source_location src)
    {
        std::stringstream msg;
        msg << user_msg << ": " << hipGetErrorString(code) << " (" << code << ")";
        if(src.function_name())
            msg << " in function '" << src.function_name();
        msg << "' at " << src.file_name() << ":" << src.line() << ":" << src.column();
        return msg.str();
    }

    /// @brief Construct a generic HIP error.
    ///
    /// @param msg User-given message about the activity at time of error.
    /// @param code The status to report.
    /// @param src The location where this error was discovered. Defaults to the caller's
    /// location.
    HipError(std::string_view msg,
             hipError_t code,
             std::source_location src = std::source_location::current())
        : std::runtime_error(format_error(msg, code, src)), code_(code)
    {
    }

    /// @brief Retrieve the inner error code.
    ///
    /// This function returns the status code that was encountered while checking an
    /// operation for errors.
    hipError_t code() const { return code_; }

    private:
    hipError_t code_;
};

/// @brief HIP out of memory error.
///
/// This a derivation of `HipError` which is specialized for Out-of-memory errors. This
/// makes it easier to attach additional context, and to match on these errors while
/// using `catch` blocks.
///
/// @see HipError
struct OutOfDeviceMemoryError : HipError
{
    /// @brief Construct an out-of-device-memory error.
    ///
    /// @param msg User-given message about the activity at time of error.
    /// @param src The location where this error was discovered. Defaults to the caller's
    /// location.
    OutOfDeviceMemoryError(std::string_view msg     = "failed to allocate device memory",
                           std::source_location src = std::source_location::current())
        : HipError(msg, hipErrorOutOfMemory, src)
    {
    }
};

/// @brief Check HIP status for errors.
///
/// This function checks a HIP status code (obtained from a HIP function call) for any
/// errors. If the status `code` is not `hipSuccess`, this function throws an instance of
/// `HipError`. The exact type thats thrown depends on the status. If `code` represents
/// an out-of-memory error `hipErrorOutOfMemory`, then `OutOfDeviceMemoryError` will be
/// thrown instead.
///
/// @param msg User-given message about the activity at possible time of error.
/// @param code The HIP status code to examine.
/// @param src The location where this status was set. Defaults to the caller's location.
///
/// @throws HipError if `code` is not `hipSuccess`.
///
/// @see HipError
/// @see OutOfDeviceMemoryError
inline void check_hip(std::string_view msg,
                      hipError_t code,
                      std::source_location src = std::source_location::current())
{
    // -Wswitch-enum throws a warning if this code is changed into a switch, even with
    // the `default` label...

    if(code == hipSuccess)
        // When you beat the error allegations
        return;
    else if(code == hipErrorOutOfMemory)
        throw OutOfDeviceMemoryError(msg, src);
    else
        throw HipError(msg, code, src);
}

/// @brief Check HIP status for errors.
///
/// This function is similar to `check_hip(std::string_view, hipError_t)`, except that a
/// default message is given.
///
/// @param code The HIP status code to examine.
/// @param src The location where this status was set. Defaults to the caller's location.
///
/// @throws HipError if `code` is not `hipSuccess`.
///
/// @see HipError
/// @see OutOfDeviceMemoryError
/// @see check_hip(std::string_view, hipError_t)
inline void check_hip(hipError_t code, std::source_location src = std::source_location::current())
{
    check_hip(code == hipErrorOutOfMemory ? "failed to allocate device memory"
                                          : "HIP runtime error",
              code,
              src);
}

} // namespace ck_tile::builder::test
