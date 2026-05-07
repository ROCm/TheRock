// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_print_common.hpp"
#include "ck_tile/core/tensor/buffer_view.hpp"
#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

class PrintBufferViewTest : public PrintTest
{
};

TEST_F(PrintBufferViewTest, PrintGenericBufferView)
{
    // Test printing generic address space buffer_view
    float data[4] = {100.f, 200.f, 300.f, 400.f};
    auto bv       = make_buffer_view<address_space_enum::generic>(&data, 4);

    std::string output = CapturePrintOutput(bv);

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("buffer_view{AddressSpace: generic") != std::string::npos);
    EXPECT_TRUE(output.find("p_data_:") != std::string::npos);
    EXPECT_TRUE(output.find("buffer_size_:") != std::string::npos);
    EXPECT_TRUE(output.find("invalid_element_value_:") != std::string::npos);
    EXPECT_TRUE(output.find("}") != std::string::npos);
}

TEST_F(PrintBufferViewTest, PrintGlobalBufferView)
{
    // Test printing global address space buffer_view
    float data[4] = {100.f, 200.f, 300.f, 400.f};
    auto bv       = make_buffer_view<address_space_enum::global>(&data, 4);

    std::string output = CapturePrintOutput(bv);

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("buffer_view{AddressSpace: global") != std::string::npos);
    EXPECT_TRUE(output.find("p_data_:") != std::string::npos);
    EXPECT_TRUE(output.find("buffer_size_:") != std::string::npos);
    EXPECT_TRUE(output.find("invalid_element_value_:") != std::string::npos);
    EXPECT_TRUE(output.find("}") != std::string::npos);
}

TEST_F(PrintBufferViewTest, PrintLdsBufferView)
{
    // Test printing LDS address space buffer_view
    float data[4] = {100.f, 200.f, 300.f, 400.f};
    auto bv       = make_buffer_view<address_space_enum::lds>(data, 4);

    std::string output = CapturePrintOutput(bv);

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("buffer_view{AddressSpace: lds") != std::string::npos);
    EXPECT_TRUE(output.find("p_data_:") != std::string::npos);
    EXPECT_TRUE(output.find("buffer_size_:") != std::string::npos);
    EXPECT_TRUE(output.find("invalid_element_value_:") != std::string::npos);
    EXPECT_TRUE(output.find("}") != std::string::npos);
}

TEST_F(PrintBufferViewTest, PrintVgprBufferView)
{
    // Test printing VGPR address space buffer_view
    float data[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    auto bv       = make_buffer_view<address_space_enum::vgpr>(data, 4);

    std::string output = CapturePrintOutput(bv);

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("buffer_view{AddressSpace: vgpr") != std::string::npos);
    EXPECT_TRUE(output.find("p_data_:") != std::string::npos);
    EXPECT_TRUE(output.find("buffer_size_:") != std::string::npos);
    EXPECT_TRUE(output.find("invalid_element_value_:") != std::string::npos);
    EXPECT_TRUE(output.find("}") != std::string::npos);
}

} // namespace ck_tile
