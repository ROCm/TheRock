// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_print_common.hpp"
#include "ck_tile/core/utility/print.hpp"
#include "ck_tile/core/container/sequence.hpp"

namespace ck_tile {

class PrintSequenceTest : public PrintTest
{
};

TEST_F(PrintSequenceTest, PrintSimpleSequence)
{
    // Test printing sequence<1, 5, 8>
    constexpr auto seq = sequence<1, 5, 8>{};

    std::string output = CapturePrintOutput(seq);

    // Verify the output format
    EXPECT_EQ(output, "sequence<1, 5, 8>");
}

TEST_F(PrintSequenceTest, PrintSingleElementSequence)
{
    // Test printing sequence<42>
    constexpr auto seq = sequence<42>{};

    std::string output = CapturePrintOutput(seq);

    EXPECT_EQ(output, "sequence<42>");
}

TEST_F(PrintSequenceTest, PrintEmptySequence)
{
    // Test printing sequence<> (empty sequence)
    constexpr auto seq = sequence<>{};

    std::string output = CapturePrintOutput(seq);

    EXPECT_EQ(output, "sequence<>");
}

} // namespace ck_tile
