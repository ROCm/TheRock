// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_print_common.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

class PrintCoordinateTransformTest : public PrintTest
{
};

TEST_F(PrintCoordinateTransformTest, PrintPassThrough)
{
    // Test printing pass_through transform
    auto pt = make_pass_through_transform(number<32>{});

    std::string output = CapturePrintOutput(pt);

    // Verify it contains the pass_through identifier and some structure
    EXPECT_TRUE(output.find("pass_through{") == 0);
    EXPECT_TRUE(output.find("up_lengths_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

TEST_F(PrintCoordinateTransformTest, PrintEmbed)
{
    // Test printing embed transform
    auto embed_transform = make_embed_transform(make_tuple(number<4>{}, number<8>{}),
                                                make_tuple(number<1>{}, number<4>{}));

    std::string output = CapturePrintOutput(embed_transform);

    // Verify it contains the embed identifier and key fields
    EXPECT_TRUE(output.find("embed{") == 0);
    EXPECT_TRUE(output.find("up_lengths_") != std::string::npos);
    EXPECT_TRUE(output.find("coefficients_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

TEST_F(PrintCoordinateTransformTest, PrintMerge)
{
    // Test printing merge transform
    auto merge_transform = make_merge_transform(make_tuple(number<4>{}, number<8>{}));

    std::string output = CapturePrintOutput(merge_transform);

    // Verify it contains merge identifier and key fields
    EXPECT_TRUE(output.find("merge") ==
                0); // Could be merge_v2_magic_division or merge_v3_division_mod
    EXPECT_TRUE(output.find("low_lengths_") != std::string::npos ||
                output.find("up_lengths_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

TEST_F(PrintCoordinateTransformTest, PrintUnmerge)
{
    // Test printing unmerge transform
    auto unmerge_transform = make_unmerge_transform(make_tuple(number<4>{}, number<8>{}));

    std::string output = CapturePrintOutput(unmerge_transform);

    // Verify it contains the unmerge identifier and key fields
    EXPECT_TRUE(output.find("unmerge{") == 0);
    EXPECT_TRUE(output.find("up_lengths_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

TEST_F(PrintCoordinateTransformTest, PrintFreeze)
{
    // Test printing freeze transform
    auto freeze_transform = make_freeze_transform(number<5>{});

    std::string output = CapturePrintOutput(freeze_transform);

    // Verify it contains the freeze identifier and key fields
    EXPECT_TRUE(output.find("freeze{") == 0);
    EXPECT_TRUE(output.find("low_idx_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

} // namespace ck_tile
