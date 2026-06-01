// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_print_common.hpp"
#include "ck_tile/core/algorithm/static_encoding_pattern.hpp"
#include "ck_tile/core/utility/print.hpp"

#include <sstream>

namespace ck_tile {

class PrintStaticEncodingPatternTest : public PrintTest
{
    protected:
    template <typename T0, typename T1, typename T2>
    void TestY0Y1Y2(const std::string& output, T0 Y0, T1 Y1, T2 Y2)
    {
        std::stringstream expected;
        expected << "<Y0, Y1, Y2>: <" << Y0 << ", " << Y1 << ", " << Y2 << ">";
        EXPECT_TRUE(output.find(expected.str()) != std::string::npos);
    }
    template <typename T0, typename T1>
    void TestX0X1(const std::string& output, T0 X0, T1 X1)
    {
        std::stringstream expected;
        expected << "<X0, X1>: <" << X0 << ", " << X1 << ">";
        EXPECT_TRUE(output.find(expected.str()) != std::string::npos);
    }
};

TEST_F(PrintStaticEncodingPatternTest, PrintThreadRakedPattern)
{
    // Test printing thread raked pattern
    using PatternType =
        tile_distribution_encoding_pattern_2d<64,
                                              8,
                                              16,
                                              4,
                                              tile_distribution_pattern::thread_raked>;
    PatternType pattern;

    std::string output = CapturePrintOutput(pattern);

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("tile_distribution_encoding_pattern_2d") != std::string::npos);
    EXPECT_TRUE(output.find("BlockSize:64") != std::string::npos);
    EXPECT_TRUE(output.find("YPerTile:8") != std::string::npos);
    EXPECT_TRUE(output.find("XPerTile:16") != std::string::npos);
    EXPECT_TRUE(output.find("VecSize:4") != std::string::npos);
    EXPECT_TRUE(output.find("thread_raked") != std::string::npos);
    TestY0Y1Y2(output, PatternType::Y0, PatternType::Y1, PatternType::Y2);
    TestX0X1(output, PatternType::X0, PatternType::X1);
}

TEST_F(PrintStaticEncodingPatternTest, PrintWarpRakedPattern)
{
    // Test printing warp raked pattern
    using PatternType =
        tile_distribution_encoding_pattern_2d<128,
                                              16,
                                              32,
                                              8,
                                              tile_distribution_pattern::warp_raked>;
    PatternType pattern;

    std::string output = CapturePrintOutput(pattern);

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("tile_distribution_encoding_pattern_2d") != std::string::npos);
    EXPECT_TRUE(output.find("BlockSize:128") != std::string::npos);
    EXPECT_TRUE(output.find("YPerTile:16") != std::string::npos);
    EXPECT_TRUE(output.find("XPerTile:32") != std::string::npos);
    EXPECT_TRUE(output.find("VecSize:8") != std::string::npos);
    EXPECT_TRUE(output.find("warp_raked") != std::string::npos);
    TestY0Y1Y2(output, PatternType::Y0, PatternType::Y1, PatternType::Y2);
    TestX0X1(output, PatternType::X0, PatternType::X1);
}

TEST_F(PrintStaticEncodingPatternTest, PrintBlockRakedPattern)
{
    // Test printing block raked pattern
    using PatternType =
        tile_distribution_encoding_pattern_2d<256,
                                              32,
                                              64,
                                              16,
                                              tile_distribution_pattern::block_raked>;
    PatternType pattern;

    std::string output = CapturePrintOutput(pattern);

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("tile_distribution_encoding_pattern_2d") != std::string::npos);
    EXPECT_TRUE(output.find("BlockSize:256") != std::string::npos);
    EXPECT_TRUE(output.find("YPerTile:32") != std::string::npos);
    EXPECT_TRUE(output.find("XPerTile:64") != std::string::npos);
    EXPECT_TRUE(output.find("VecSize:16") != std::string::npos);
    EXPECT_TRUE(output.find("block_raked") != std::string::npos);
    TestY0Y1Y2(output, PatternType::Y0, PatternType::Y1, PatternType::Y2);
    TestX0X1(output, PatternType::X0, PatternType::X1);
}

} // namespace ck_tile
