// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "profiler/profile_max_pool3d_bwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

template <typename Tuple>
class TestMaxPool3dBwd : public ::testing::Test
{
    protected:
    using DOutDataType  = std::tuple_element_t<0, Tuple>;
    using DInDataType   = std::tuple_element_t<1, Tuple>;
    using IndexDataType = std::tuple_element_t<2, Tuple>;

    using InDataType  = DInDataType;
    using OutDataType = DOutDataType;

    std::vector<PoolingParam> params;

    void Run()
    {
        for(size_t i = 0; i < this->params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param = this->params[i];
            bool success =
                ck::profiler::profile_max_pool3d_bwd_impl<InDataType,
                                                          OutDataType,
                                                          IndexDataType,
                                                          DOutDataType,
                                                          DInDataType,
                                                          false>(true,
                                                                 2,
                                                                 false,
                                                                 false,
                                                                 param.length_,
                                                                 param.window_spatial_lengths_,
                                                                 param.window_strides_,
                                                                 param.window_dilations_,
                                                                 param.input_left_pads_,
                                                                 param.input_right_pads_,
                                                                 instance_index);
            EXPECT_TRUE(success);
        }
    }
};

#if defined(CK_ENABLE_FP16) && defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP32)
using KernelTypes = ::testing::Types<std::tuple<F16, F16, I32, NDHWC, NDHWC>,
                                     std::tuple<BF16, BF16, I32, NDHWC, NDHWC>,
                                     std::tuple<F32, F32, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP32)
using KernelTypes = ::testing::Types<std::tuple<F16, F16, I32, NDHWC, NDHWC>,
                                     std::tuple<F32, F32, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP32)
using KernelTypes = ::testing::Types<std::tuple<BF16, BF16, I32, NDHWC, NDHWC>,
                                     std::tuple<F32, F32, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_FP16) && defined(CK_ENABLE_BF16)
using KernelTypes = ::testing::Types<std::tuple<F16, F16, I32, NDHWC, NDHWC>,
                                     std::tuple<BF16, BF16, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_FP16)
using KernelTypes = ::testing::Types<std::tuple<F16, F16, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_BF16)
using KernelTypes = ::testing::Types<std::tuple<BF16, BF16, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_FP32)
using KernelTypes = ::testing::Types<std::tuple<F32, F32, I32, NDHWC, NDHWC>>;
#endif

TYPED_TEST_SUITE(TestMaxPool3dBwd, KernelTypes);
TYPED_TEST(TestMaxPool3dBwd, Test_Pool)
{
    // length, window_length, window_stride, window_dilation, left_pad, right_pad
    this->params = {{{1, 1, 1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 16, 64, 64, 64}, {4, 4, 4}, {4, 4, 4}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 32, 30, 30, 30}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};

    // this->params = {{{2, 32, 30, 30, 30}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1,
    // 1}}};

    this->Run();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        param_mask     = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: param_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
