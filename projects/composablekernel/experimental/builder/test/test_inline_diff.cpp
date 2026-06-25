// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "testing_utils.hpp"

using ck_tile::test::inlineDiff;

TEST(InlineDiff, simpleColorDiff)
{
    std::string str1{"hello"};
    std::string str2{"hello"};
    std::string str3{"world"};

    // some easy tests
    // you can veryfy the ungodly strings are meaningful by running echo -e "<string>"
    EXPECT_THAT(inlineDiff(str1, str2, true), "hello");
    EXPECT_THAT(inlineDiff(str1, str3, true),
                "[\x1B[36mwor\x1B[0m|\x1B[35mhel\x1B[0m]l[\x1B[36md\x1B[0m|\x1B[35mo\x1B[0m]");
}

TEST(InlineDiff, noColorDiff)
{
    std::string str1{"hello"};
    std::string str2{"hello"};
    std::string str3{"world"};

    // some easy tests without color
    EXPECT_THAT(inlineDiff(str1, str2, false), "hello");
    EXPECT_THAT(inlineDiff(str1, str3, false), "[wor|hel]l[d|o]");
}

TEST(InlineDiff, complexColorDiff)
{

    // now something more interesting
    std::string str4{"this part has changed, this part has been left out, this part, this part has "
                     "an extra letter"};
    std::string str5{
        "this part has degeahc, this part has, this part added, this part has ana extra letter"};

    EXPECT_THAT(
        inlineDiff(str5, str4, true),
        "this part has [\x1B[36mchanged\x1B[0m|\x1B[35mdegeahc\x1B[0m], this part has[\x1B[36m "
        "been left out\x1B[0m|\x1B[35m\x1B[0m], this part[\x1B[36m\x1B[0m|\x1B[35m added\x1B[0m], "
        "this part has an[\x1B[36m\x1B[0m|\x1B[35ma\x1B[0m] extra letter");
};
