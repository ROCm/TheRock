// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_batched_gemm_impl.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm.hpp"
static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;
struct GemmParams
{
    ck::index_t M;
    ck::index_t N;
    ck::index_t K;
    ck::index_t BatchCount;
};

class TestBatchedGemm : public ::testing::Test
{
    protected:
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    std::vector<GemmParams> params;

    template <typename DataType>
    void Run()
    {
        using namespace ck::tensor_operation::device;

        bool pass = true;
        for(size_t i = 0; i < params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param           = params[i];
            const auto M          = param.M;
            const auto N          = param.N;
            const auto K          = param.K;
            const auto BatchCount = param.BatchCount;

            pass = pass && ck::profiler::profile_batched_gemm_impl<DataType,
                                                                   DataType,
                                                                   DataType,
                                                                   Row,
                                                                   Row,
                                                                   Row,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   DeviceBatchedGemm<Row,
                                                                                     Row,
                                                                                     Row,
                                                                                     DataType,
                                                                                     DataType,
                                                                                     DataType,
                                                                                     PassThrough,
                                                                                     PassThrough,
                                                                                     PassThrough>>(
                               true,
                               1,
                               false,
                               1,
                               M,
                               N,
                               K,
                               K,
                               N,
                               N,
                               M * K,
                               K * N,
                               M * N,
                               BatchCount,
                               instance_index);

            pass = pass && ck::profiler::profile_batched_gemm_impl<DataType,
                                                                   DataType,
                                                                   DataType,
                                                                   Row,
                                                                   Col,
                                                                   Row,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   DeviceBatchedGemm<Row,
                                                                                     Col,
                                                                                     Row,
                                                                                     DataType,
                                                                                     DataType,
                                                                                     DataType,
                                                                                     PassThrough,
                                                                                     PassThrough,
                                                                                     PassThrough>>(
                               true,
                               1,
                               false,
                               1,
                               M,
                               N,
                               K,
                               K,
                               K,
                               N,
                               M * K,
                               K * N,
                               M * N,
                               BatchCount,
                               instance_index);

            pass = pass && ck::profiler::profile_batched_gemm_impl<DataType,
                                                                   DataType,
                                                                   DataType,
                                                                   Col,
                                                                   Row,
                                                                   Row,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   DeviceBatchedGemm<Col,
                                                                                     Row,
                                                                                     Row,
                                                                                     DataType,
                                                                                     DataType,
                                                                                     DataType,
                                                                                     PassThrough,
                                                                                     PassThrough,
                                                                                     PassThrough>>(
                               true,
                               1,
                               false,
                               1,
                               M,
                               N,
                               K,
                               M,
                               N,
                               N,
                               M * K,
                               K * N,
                               M * N,
                               BatchCount,
                               instance_index);

            pass = pass && ck::profiler::profile_batched_gemm_impl<DataType,
                                                                   DataType,
                                                                   DataType,
                                                                   Col,
                                                                   Col,
                                                                   Row,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   DeviceBatchedGemm<Col,
                                                                                     Col,
                                                                                     Row,
                                                                                     DataType,
                                                                                     DataType,
                                                                                     DataType,
                                                                                     PassThrough,
                                                                                     PassThrough,
                                                                                     PassThrough>>(
                               true,
                               1,
                               false,
                               1,
                               M,
                               N,
                               K,
                               M,
                               K,
                               N,
                               M * K,
                               K * N,
                               M * N,
                               BatchCount,
                               instance_index);
        }
        EXPECT_TRUE(pass);
    }
};

// #ifdef CK_ENABLE_INT8
// TEST_F(TestBatchedGemm, i8)
// {
//     this->params.push_back({64, 64, 64, 2});
//     this->params.push_back({64, 64, 64, 1});
//     this->params.push_back({60, 60, 60, 2});
//     this->params.push_back({68, 68, 68, 2});
//     this->params.push_back({40, 40, 40, 2});
//     this->params.push_back({256, 256, 128, 3});
//     this->template Run<int8_t>();
// }
// #endif

#ifdef CK_ENABLE_BF16
TEST_F(TestBatchedGemm, bf16)
{
    this->params.push_back({64, 64, 64, 2});
    this->params.push_back({64, 64, 64, 1});
    this->params.push_back({40, 40, 40, 2});
    this->params.push_back({256, 256, 128, 3});

    // Tests with larger MNK
    this->params.push_back({512, 256, 128, 1});
    this->params.push_back({256, 240, 192, 2});
    this->params.push_back({256, 256, 128, 3});
    this->params.push_back({240, 128, 128, 5});
    this->template Run<ck::bhalf_t>();
}
#endif

#ifdef CK_ENABLE_FP16
TEST_F(TestBatchedGemm, fp16)
{
    this->params.push_back({64, 64, 64, 2});
    this->params.push_back({64, 64, 64, 1});
    this->params.push_back({40, 40, 40, 2});
    this->params.push_back({256, 256, 128, 3});

    // Tests with larger MNK
    this->params.push_back({512, 256, 128, 1});
    this->params.push_back({256, 240, 192, 2});
    this->params.push_back({256, 256, 128, 3});
    this->params.push_back({240, 128, 128, 5});
    this->template Run<ck::half_t>();
}
#endif

// #ifdef CK_ENABLE_FP32
// TEST_F(TestBatchedGemm, fp32)
// {
//     this->params.push_back({64, 64, 64, 2});
//     this->params.push_back({64, 64, 64, 1});
//     this->params.push_back({60, 60, 60, 2});
//     this->params.push_back({68, 68, 68, 2});
//     this->params.push_back({40, 40, 40, 2});
//     this->params.push_back({256, 256, 128, 3});
//     this->template Run<float>();
// }
// #endif

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
