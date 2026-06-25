// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/error.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <memory>
#include <sstream>

/// This file deals with tensor memory management and allocation. The main
/// item is the `DeviceBuffer`: An owned piece of device memory, which is
/// automatically freed when it goes out of scope.

namespace ck_tile::builder::test {

/// @brief Automatic deleter for GPU memory.
///
/// This structure implements a C++ functor which can be used to configure
/// `std::unique_ptr` to automatically delete memory using `hipFree`.
///
/// @see DeviceBuffer
struct DeviceMemoryDeleter
{
    /// @brief Deleter callback.
    ///
    /// This function is invoked by `std::unique_ptr` when memory that the
    /// pointer represents should be freed. In our implementation, we just
    /// pass it directly to `hipFree`.
    void operator()(std::byte* ptr) const
    {
        if(ptr)
            (void)hipFree(ptr);
    }
};

/// @brief Automatically managed GPU memory.
///
/// The `DeviceBuffer` is an automatically managed pointer for GPU memory. When
/// adopting a device pointer into a `DeviceBuffer`, it will automatically be
/// free'd when the pointer goes out of scope. Memory can be allocated directly
/// into a `DeviceBuffer` using `alloc_buffer()` or `alloc_tensor_buffer()`.
///
/// Since this type is just an alias of `std::unique_ptr`, you can use that type's
/// functionality to manage memory further, such as `.reset()` to release the
/// memory.
///
/// @see alloc_buffer()
/// @see alloc_tensor_buffer()
using DeviceBuffer = std::unique_ptr<std::byte[], DeviceMemoryDeleter>;

/// @brief Allocate automatically managed GPU memory.
///
/// This function essentially acts like a managed version of hipMalloc -
/// allocating GPU memory on the currently active device - except that this
/// version returns an automatically managed pointer.
///
/// @param size The amount of memory to allocate in bytes.
/// @throws OutOfDeviceMemoryError if memory allocation failed.
///
/// @see DeviceBuffer
/// @see OutOfDeviceMemoryError
/// @see hipMalloc()
inline DeviceBuffer alloc_buffer(size_t size)
{
    std::byte* d_buf = nullptr;
    if(const auto status = hipMalloc(&d_buf, size); status != hipSuccess)
    {
        // Add some additional context

        size_t free, total;
        check_hip("failed to get HIP memory info", hipMemGetInfo(&free, &total));

        std::stringstream ss;
        ss << "failed to allocate device memory (tried to allocate " << size << " bytes with only "
           << free << " available)";

        throw OutOfDeviceMemoryError(ss.str());
    }
    return DeviceBuffer(d_buf);
}

} // namespace ck_tile::builder::test
