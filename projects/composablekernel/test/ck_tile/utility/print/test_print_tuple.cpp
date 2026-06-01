// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_print_common.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

class PrintTupleTest : public PrintTest
{
};

TEST_F(PrintTupleTest, PrintSimpleTuple)
{
    // Test printing tuple with numbers
    auto tup = make_tuple(number<1>{}, number<5>{}, number<8>{});

    std::string output = CapturePrintOutput(tup);

    // Verify the output format matches tuple print implementation
    EXPECT_TRUE(output.find("tuple<") == 0);
    EXPECT_TRUE(output.find("1") != std::string::npos);
    EXPECT_TRUE(output.find("5") != std::string::npos);
    EXPECT_TRUE(output.find("8") != std::string::npos);
    EXPECT_TRUE(output.back() == '>');
}

TEST_F(PrintTupleTest, PrintSingleElementTuple)
{
    // Test printing tuple with single element
    auto tup = make_tuple(number<42>{});

    std::string output = CapturePrintOutput(tup);

    EXPECT_TRUE(output.find("tuple<") == 0);
    EXPECT_TRUE(output.find("42") != std::string::npos);
    EXPECT_TRUE(output.back() == '>');
}

TEST_F(PrintTupleTest, PrintEmptyTuple)
{
    // Test printing empty tuple
    auto tup = make_tuple();

    std::string output = CapturePrintOutput(tup);

    EXPECT_EQ(output, "tuple<>");
}

TEST_F(PrintTupleTest, PrintMixedTypeTuple)
{
    // Test printing tuple with mixed types (numbers and constants)
    auto tup = make_tuple(number<10>{}, constant<20>{}, number<30>{});

    std::string output = CapturePrintOutput(tup);

    EXPECT_TRUE(output.find("tuple<") == 0);
    EXPECT_TRUE(output.find("10") != std::string::npos);
    EXPECT_TRUE(output.find("20") != std::string::npos);
    EXPECT_TRUE(output.find("30") != std::string::npos);
    EXPECT_TRUE(output.back() == '>');
}

} // namespace ck_tile
