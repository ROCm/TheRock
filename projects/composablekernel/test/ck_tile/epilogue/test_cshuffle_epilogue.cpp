// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_cshuffle_epilogue_util.hpp"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

using namespace ck_tile;

class CShuffleEpilogueTest : public ::testing::Test
{
    protected:
    void SetUp() override {}
};

TEST_F(CShuffleEpilogueTest, BasicHalfTest)
{
    // Basic test configuration with half_t data types
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using ODataType   = ck_tile::half_t;

    constexpr index_t kMPerBlock = 256;
    constexpr index_t kNPerBlock = 256;
    constexpr index_t MWave      = 2;
    constexpr index_t NWave      = 2;
    constexpr index_t MPerXdl    = 32;
    constexpr index_t NPerXdl    = 32;
    constexpr index_t KPerXdl    = 8;

    using TestProblem = SimpleCShuffleEpilogueProblem<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      ODataType,
                                                      kMPerBlock,
                                                      kNPerBlock,
                                                      MWave,
                                                      NWave,
                                                      MPerXdl,
                                                      NPerXdl,
                                                      KPerXdl>;

    auto result = run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>(ScaleType::None);
    EXPECT_FLOAT_EQ(result[0], 2.0F) << "Basic CShuffleEpilogue test failed";
}

TEST_F(CShuffleEpilogueTest, BasicHalfTestWithScale)
{
    // Basic test configuration with half_t data types
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using ODataType   = ck_tile::half_t;

    constexpr index_t kMPerBlock = 256;
    constexpr index_t kNPerBlock = 256;
    constexpr index_t MWave      = 2;
    constexpr index_t NWave      = 2;
    constexpr index_t MPerXdl    = 32;
    constexpr index_t NPerXdl    = 32;
    constexpr index_t KPerXdl    = 8;

    using TestProblem = SimpleCShuffleEpilogueProblem<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      ODataType,
                                                      kMPerBlock,
                                                      kNPerBlock,
                                                      MWave,
                                                      NWave,
                                                      MPerXdl,
                                                      NPerXdl,
                                                      KPerXdl>;

    auto result =
        run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>(ScaleType::RowCol);
    EXPECT_FLOAT_EQ(result[0], 2.0F) << "RowCol CShuffleEpilogue test failed: first element not 2";
    EXPECT_FLOAT_EQ(result[1], 4.0F)
        << "RowCol CShuffleEpilogue test failed: second element not 2*2";
}

TEST_F(CShuffleEpilogueTest, BasicHalfTestWithTensorScale)
{
    // Basic test configuration with half_t data types
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using ODataType   = ck_tile::half_t;

    constexpr index_t kMPerBlock = 256;
    constexpr index_t kNPerBlock = 256;
    constexpr index_t MWave      = 2;
    constexpr index_t NWave      = 2;
    constexpr index_t MPerXdl    = 32;
    constexpr index_t NPerXdl    = 32;
    constexpr index_t KPerXdl    = 8;

    using TestProblem = SimpleCShuffleEpilogueProblem<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      ODataType,
                                                      kMPerBlock,
                                                      kNPerBlock,
                                                      MWave,
                                                      NWave,
                                                      MPerXdl,
                                                      NPerXdl,
                                                      KPerXdl>;

    auto result =
        run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>(ScaleType::Tensor);
    EXPECT_FLOAT_EQ(result[0], 4.0F)
        << "TensorScale CShuffleEpilogue test failed: first element not 2*2=4";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
