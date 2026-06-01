// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_print_common.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

class PrintArrayTest : public PrintTest
{
};

TEST_F(PrintArrayTest, PrintIntArray)
{
    // Test printing array<int, 3>
    array<int, 3> arr{10, 20, 30};

    std::string output = CapturePrintOutput(arr);

    // The expected format should match the array print function implementation
    EXPECT_EQ(output, "array{size: 3, data: [10, 20, 30]}");
}

TEST_F(PrintArrayTest, PrintSingleElementArray)
{
    // Test printing array<int, 1>
    array<int, 1> arr{42};

    std::string output = CapturePrintOutput(arr);

    EXPECT_EQ(output, "array{size: 1, data: [42]}");
}

TEST_F(PrintArrayTest, PrintEmptyArray)
{
    // Test printing array<int, 0> (empty array)
    array<int, 0> arr{};

    std::string output = CapturePrintOutput(arr);

    EXPECT_EQ(output, "array{size: 0, data: []}");
}

TEST_F(PrintArrayTest, PrintFloatArray)
{
    // Test printing array with float values
    array<float, 2> arr{3.14f, 2.71f};

    std::string output = CapturePrintOutput(arr);

    // Note: float printing format may vary, so we'll test for basic structure
    EXPECT_TRUE(output.find("array{size: 2, data: [") == 0);
    EXPECT_TRUE(output.find("3.14") != std::string::npos);
    EXPECT_TRUE(output.find("2.71") != std::string::npos);
    EXPECT_TRUE(output.find("]}") == output.length() - 2);
}

} // namespace ck_tile
