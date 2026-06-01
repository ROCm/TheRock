// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/error.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace ckt = ck_tile::builder::test;

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Throws;
using ::testing::ThrowsMessage;

[[noreturn]] void throw_error() { throw ckt::HipError("test error", hipErrorInvalidValue); }

TEST(HipError, SourceInfo)
{
    EXPECT_THAT(throw_error,
                ThrowsMessage<ckt::HipError>(AllOf(
                    // The error message should include...
                    // ...the user message
                    HasSubstr("test error"),
                    // ...the HIP message
                    HasSubstr("invalid argument"),
                    // ...the HIP status code,
                    HasSubstr("(1)"),
                    // ...the filename
                    HasSubstr("experimental/builder/test/unit_error.cpp"),
                    // ...the function name
                    HasSubstr("throw_error")
                    // Note: Don't include the row/column so that we can move
                    // stuff around in this file.
                    )));
}

TEST(CheckHip, BasicUsage)
{
    EXPECT_THAT([] { ckt::check_hip(hipSuccess); }, Not(Throws<ckt::HipError>()));
    EXPECT_THAT([] { ckt::check_hip(hipErrorNotMapped); }, Throws<ckt::HipError>());
    EXPECT_THAT([] { ckt::check_hip(hipErrorOutOfMemory); }, Throws<ckt::OutOfDeviceMemoryError>());
    EXPECT_THAT([] { ckt::check_hip("test message", hipErrorAlreadyMapped); },
                ThrowsMessage<ckt::HipError>(HasSubstr("test message")));
}
