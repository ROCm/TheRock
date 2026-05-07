// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using ck_tile::test::HipError;
using ck_tile::test::HipSuccess;
using ::testing::Eq;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Throws;

TEST(DeviceBuffer, DefaultToNull)
{
    ckt::DeviceBuffer buffer;
    EXPECT_THAT(buffer.get(), IsNull());
}

TEST(DeviceBuffer, AllocBuffer)
{
    const auto size = 12345;
    auto buffer     = ckt::alloc_buffer(size);

    // Pointer should be non-null
    EXPECT_THAT(buffer.get(), NotNull());

    // Actually, the pointer should be a device pointer
    hipPointerAttribute_t attr;
    EXPECT_THAT(hipPointerGetAttributes(&attr, buffer.get()), HipSuccess());

    EXPECT_THAT(attr.devicePointer, NotNull());
    EXPECT_THAT(attr.type, Eq(hipMemoryTypeDevice));

    // Memory should be writable without error
    EXPECT_THAT(hipMemset(buffer.get(), 0xFF, size), HipSuccess());
}

TEST(DeviceBuffer, AutoFree)
{
    const auto size = 12345;
    std::byte* ptr  = nullptr;

    {
        auto buffer = ckt::alloc_buffer(size);
        ptr         = buffer.get();
    }

    // Trying to use a pointer after freeing should return en error in HIP.
    EXPECT_THAT(hipMemset(ptr, 0xFF, size), HipError(hipErrorInvalidValue));

    // Reset internal HIP error state.
    // Otherwise, the error may leak into other tests, triggering anything that
    // checks the output of hipGetLastError();
    (void)hipGetLastError();
}

TEST(DeviceBuffer, ThrowsOnOom)
{
    const auto size = size_t{1} << 60; // 1 exabyte

    auto check = [] { auto buffer = ckt::alloc_buffer(size); };
    EXPECT_THAT(check, Throws<ckt::OutOfDeviceMemoryError>());

    // Reset internal HIP error state.
    // Otherwise, the error may leak into other tests, triggering anything that
    // checks the output of hipGetLastError();
    (void)hipGetLastError();
}

TEST(DeviceBuffer, AllocTensorBuffer)
{
    ckt::TensorDescriptor<ckb::DataType::FP32, 3> descriptor({128, 128, 128}, {128 * 128, 128, 1});

    auto buffer = ckt::alloc_tensor_buffer(descriptor);

    // Pointer should be non-null
    EXPECT_THAT(buffer.get(), NotNull());

    // Memory should be writable without error
    EXPECT_THAT(hipMemset(buffer.get(), 0xFF, descriptor.get_element_space_size_in_bytes()),
                HipSuccess());
}
