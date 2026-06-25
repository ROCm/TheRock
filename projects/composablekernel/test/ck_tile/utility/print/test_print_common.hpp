// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "ck_tile/core/utility/print.hpp"

class PrintTest : public ::testing::Test
{
    protected:
    void SetUp() override {}
    void TearDown() override {}
    // Helper function to capture and return the output of a print function
    template <typename T>
    std::string CapturePrintOutput(const T& type)
    {
        using namespace ck_tile;
        testing::internal::CaptureStdout();
        print(type);
        return testing::internal::GetCapturedStdout();
    }
};
