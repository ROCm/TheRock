// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "profiler/profile_layernorm_bwd_gamma_beta_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

static ck::index_t length_mask    = 0xffff;
static ck::index_t instance_index = -1;

template <typename Tuple>
class TestLayernorm2dBwdGammaBeta : public ::testing::Test
{
    protected:
    using DYDataType         = std::tuple_element_t<0, Tuple>;
    using XDataType          = std::tuple_element_t<1, Tuple>;
    using MeanInvStdDataType = std::tuple_element_t<2, Tuple>;
    using ComputeDataType    = std::tuple_element_t<3, Tuple>;
    using DGammaDataType     = std::tuple_element_t<4, Tuple>;
    using DBetaDataType      = std::tuple_element_t<5, Tuple>;

    void Run()
    {
        // Bwd data: [N, D], reduce D
        std::vector<std::vector<ck::index_t>> lengths = {
            {4, 256}, {8, 511}, {9, 1032}, {4, 2048}, {1, 8192}, {4000, 2000}};

        for(size_t i = 0; i < lengths.size(); i++)
        {
            if((length_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto length  = lengths[i];
            bool success = ck::profiler::profile_layernorm_bwd_gamma_beta_impl<DYDataType,
                                                                               XDataType,
                                                                               MeanInvStdDataType,
                                                                               ComputeDataType,
                                                                               DGammaDataType,
                                                                               DBetaDataType,
                                                                               2>(
                true, 2, false, false, length, instance_index);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<
    // DYDataType XDataType, MeanInvStdDataType, ComputeDataType, DGammaDataType, DBetaDataType>
    std::tuple<F32, F32, F32, F32, F32, F32>>;

TYPED_TEST_SUITE(TestLayernorm2dBwdGammaBeta, KernelTypes);
TYPED_TEST(TestLayernorm2dBwdGammaBeta, Test_FP32) { this->Run(); }

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        length_mask    = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: length_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
