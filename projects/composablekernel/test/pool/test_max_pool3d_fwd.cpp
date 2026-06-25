// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "profiler/profile_pool3d_fwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

template <typename Tuple>
class TestMaxPool3dFwd : public ::testing::Test
{
    protected:
    using InDataType      = std::tuple_element_t<0, Tuple>;
    using OutDataType     = std::tuple_element_t<1, Tuple>;
    using ComputeDataType = std::tuple_element_t<2, Tuple>;
    using IndexDataType   = std::tuple_element_t<3, Tuple>;

    std::vector<PoolingParam> params;

    ck::profiler::PoolFwdInputParams in_params_max_pool{true, 2, false, false, false, 0};
    ck::profiler::PoolFwdInputParams in_params_max_pool_indexed{true, 2, false, false, true, 0};

    void Run()
    {
        for(size_t i = 0; i < this->params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param = this->params[i];

            ck::profiler::PoolFwdKernelParams kernel_params{param.length_,
                                                            param.window_spatial_lengths_,
                                                            param.window_strides_,
                                                            param.window_dilations_,
                                                            param.input_left_pads_,
                                                            param.input_right_pads_};

            // max pool
            bool success =
                ck::profiler::profile_pool3d_fwd_impl<InDataType,
                                                      OutDataType,
                                                      ComputeDataType,
                                                      IndexDataType,
                                                      ck::tensor_layout::convolution::NDHWC,
                                                      ck::tensor_layout::convolution::NDHWC,
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      false>(
                    in_params_max_pool, kernel_params, instance_index);
            EXPECT_TRUE(success);

            // max pool + index
            success = ck::profiler::profile_pool3d_fwd_impl<InDataType,
                                                            OutDataType,
                                                            ComputeDataType,
                                                            IndexDataType,
                                                            ck::tensor_layout::convolution::NDHWC,
                                                            ck::tensor_layout::convolution::NDHWC,
                                                            ck::ReduceTensorOp::MAX,
                                                            false,
                                                            true>(
                in_params_max_pool_indexed, kernel_params, instance_index);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<std::tuple<I8, I8, I8, I32>,
                                     std::tuple<F8, F8, F8, I32>,
                                     std::tuple<F16, F16, F16, I32>,
                                     std::tuple<BF16, BF16, BF16, I32>,
                                     std::tuple<F32, F32, F32, I32>>;

TYPED_TEST_SUITE(TestMaxPool3dFwd, KernelTypes);
TYPED_TEST(TestMaxPool3dFwd, Test_Pool)
{
    // length, window_length, window_stride, window_dilation, left_pad, right_pad
    this->params = {{{1, 1, 1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 16, 64, 64, 64}, {64, 64, 64}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 16, 64, 64, 64}, {4, 4, 4}, {4, 4, 4}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 32, 30, 30, 30}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};

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
