// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace ck_tile::builder::test_utils {
using namespace test;

// Common test implementation
template <typename Builder>
constexpr void run_test(const std::vector<std::string>& kernel_instance_components)
{
    auto instance = typename Builder::Instance{};

    const auto kernel_string = instance.GetInstanceString();
    std::cout << "Generated kernel: " << kernel_string << std::endl;
    EXPECT_GT(kernel_string.size(), 0);

    const auto invoker_ptr = instance.MakeInvokerPointer();
    EXPECT_NE(invoker_ptr, nullptr);

    for(const auto& component : kernel_instance_components)
    {
        EXPECT_THAT(kernel_string, ::testing::HasSubstr(component));
    }
}

// Common CK Tile test implementation
template <typename Builder>
constexpr void run_ck_tile_test(const std::vector<std::string>& kernel_instance_components)
{
    auto instance = typename Builder::Instance{};

    const auto kernel_string = instance.GetTypeString();
    std::cout << "Generated kernel: " << kernel_string << std::endl;
    EXPECT_GT(kernel_string.size(), 0);
    std::cout << kernel_string << std::endl;
    for(const auto& component : kernel_instance_components)
    {
        EXPECT_THAT(kernel_string, ::testing::HasSubstr(component));
    }
}

} // namespace ck_tile::builder::test_utils
