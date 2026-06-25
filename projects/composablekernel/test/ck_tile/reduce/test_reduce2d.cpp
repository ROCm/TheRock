// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/host/kernel_launch.hpp"

template <typename Tuple>
class TestCkTileReduce : public ::testing::Test
{
    protected:
    using XDataType       = std::tuple_element_t<0, Tuple>;
    using ComputeDataType = std::tuple_element_t<1, Tuple>;
    using YDataType       = std::tuple_element_t<2, Tuple>;
    using ReduceOpType    = std::tuple_element_t<3, Tuple>;
    using BlockWarps_     = std::tuple_element_t<4, Tuple>;
    using BlockTile_      = std::tuple_element_t<5, Tuple>;
    using WarpTile_       = std::tuple_element_t<6, Tuple>;
    using ThreadTile_     = std::tuple_element_t<7, Tuple>;

    using TestReduce2dShape =
        ck_tile::Reduce2dShape<BlockWarps_, BlockTile_, WarpTile_, ThreadTile_>;

    template <std::size_t InputDim, typename KeptDimSeq, typename ReduceDimSeq>
    void RunGenericTest(const std::vector<ck_tile::index_t>& input_shape,
                        const std::vector<ck_tile::index_t>& input_strides,
                        const std::vector<ck_tile::index_t>& output_shape,
                        const std::vector<ck_tile::index_t>& output_strides,
                        ck_tile::index_t kept_dim_len_prod,
                        ck_tile::index_t total_reduce_elements,
                        KeptDimSeq kept_dims,
                        ReduceDimSeq reduce_dims)
    {
        ck_tile::HostTensor<XDataType> h_x(input_shape, input_strides);
        ck_tile::HostTensor<YDataType> h_y(output_shape, output_strides);
        ck_tile::HostTensor<YDataType> h_y_ref(output_shape, output_strides);

        ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(h_x);
        h_y.SetZero();
        h_y_ref.SetZero();

        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_y_mem(h_y.get_element_space_size_in_bytes());

        d_x_mem.ToDevice(h_x.data());
        d_y_mem.ToDevice(h_y.data()); // Initialize device output buffer

        // Problem and kernel setup
        using Problem = ck_tile::Reduce2dProblem<XDataType,
                                                 ComputeDataType,
                                                 YDataType,
                                                 TestReduce2dShape,
                                                 ReduceOpType,
                                                 KeptDimSeq,
                                                 ReduceDimSeq,
                                                 InputDim>;

        using Kernel = ck_tile::ReduceKernel<Problem>;

        // Launch configuration
        const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = 1;

        ck_tile::index_t kGridSize =
            (kept_dim_len_prod + TestReduce2dShape::Block_M - 1) / TestReduce2dShape::Block_M;

        // Generic helper to create tuple from vector based on compile-time size
        auto make_shape_tuple = []<std::size_t N>(const std::vector<ck_tile::index_t>& vec) {
            return [&vec]<std::size_t... I>(std::index_sequence<I...>) {
                return ck_tile::make_tuple(vec[I]...);
            }(std::make_index_sequence<N>{});
        };

        auto input_shape_tuple   = make_shape_tuple.template operator()<InputDim>(input_shape);
        auto input_strides_tuple = make_shape_tuple.template operator()<InputDim>(input_strides);

        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                              kGridSize,
                                              kBlockSize,
                                              0,
                                              static_cast<XDataType*>(d_x_mem.GetDeviceBuffer()),
                                              static_cast<YDataType*>(d_y_mem.GetDeviceBuffer()),
                                              input_shape_tuple,
                                              input_strides_tuple));

        // Get results back
        d_y_mem.FromDevice(h_y.data());

        // Reference computation
        ck_tile::reference_reduce<XDataType, ComputeDataType, YDataType>(
            h_x, h_y_ref, ReduceOpType{}, kept_dims, reduce_dims);

        // Calculate proper error thresholds based on data types and number of accumulations
        const auto rtol = ck_tile::get_relative_threshold<XDataType, YDataType, ComputeDataType>(
            total_reduce_elements);
        const auto atol = ck_tile::get_absolute_threshold<XDataType, YDataType, ComputeDataType>(
            5.0f, total_reduce_elements);

        bool result =
            ck_tile::check_err(h_y, h_y_ref, "Error: Incorrect reduce results!", rtol, atol);
        EXPECT_TRUE(result);
    }

    // Convenience functions for specific dimensional patterns
    void RunTest2D_KeepDim0_ReduceDim1(ck_tile::index_t dim0, ck_tile::index_t dim1)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0>{};
        constexpr auto reduce_dims = ck_tile::sequence<1>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {dim0, dim1};
        std::vector<ck_tile::index_t> input_strides = {dim1, 1};

        // Output shape and strides (keep dim0)
        std::vector<ck_tile::index_t> output_shape   = {dim0};
        std::vector<ck_tile::index_t> output_strides = {1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = dim0;
        ck_tile::index_t total_reduce_elements = dim1;

        RunGenericTest<2>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }

    void RunTest3D_KeepDim0_ReduceDim12(ck_tile::index_t dim0,
                                        ck_tile::index_t dim1,
                                        ck_tile::index_t dim2)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0>{};
        constexpr auto reduce_dims = ck_tile::sequence<1, 2>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {dim0, dim1, dim2};
        std::vector<ck_tile::index_t> input_strides = {dim1 * dim2, dim2, 1};

        // Output shape and strides (keep dim0)
        std::vector<ck_tile::index_t> output_shape   = {dim0};
        std::vector<ck_tile::index_t> output_strides = {1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = dim0;        // product of kept dimensions
        ck_tile::index_t total_reduce_elements = dim1 * dim2; // product of reduced dimensions

        RunGenericTest<3>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }

    void RunTest3D_KeepDim01_ReduceDim2(ck_tile::index_t dim0,
                                        ck_tile::index_t dim1,
                                        ck_tile::index_t dim2)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0, 1>{};
        constexpr auto reduce_dims = ck_tile::sequence<2>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {dim0, dim1, dim2};
        std::vector<ck_tile::index_t> input_strides = {dim1 * dim2, dim2, 1};

        // Output shape and strides (keep dim0)
        std::vector<ck_tile::index_t> output_shape   = {dim0, dim1};
        std::vector<ck_tile::index_t> output_strides = {dim1, 1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = dim0 * dim1; // product of kept dimensions
        ck_tile::index_t total_reduce_elements = dim2;        // product of reduced dimensions

        RunGenericTest<3>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }

    void RunTest4D_KeepDim01_ReduceDim23(ck_tile::index_t N,
                                         ck_tile::index_t C,
                                         ck_tile::index_t H,
                                         ck_tile::index_t W)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0, 1>{};
        constexpr auto reduce_dims = ck_tile::sequence<2, 3>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {N, C, H, W};
        std::vector<ck_tile::index_t> input_strides = {C * H * W, H * W, W, 1};

        // Output shape and strides (keep dim0, dim1)
        std::vector<ck_tile::index_t> output_shape   = {N, C};
        std::vector<ck_tile::index_t> output_strides = {C, 1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = N * C; // product of kept dimensions
        ck_tile::index_t total_reduce_elements = H * W; // product of reduced dimensions

        RunGenericTest<4>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }

    void RunTest4D_KeepDim03_ReduceDim12(ck_tile::index_t N,
                                         ck_tile::index_t H,
                                         ck_tile::index_t W,
                                         ck_tile::index_t C)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0, 3>{};
        constexpr auto reduce_dims = ck_tile::sequence<1, 2>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {N, H, W, C};
        std::vector<ck_tile::index_t> input_strides = {H * W * C, W * C, C, 1};

        // Output shape and strides (keep dim0, dim1)
        std::vector<ck_tile::index_t> output_shape   = {N, C};
        std::vector<ck_tile::index_t> output_strides = {C, 1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = N * C; // product of kept dimensions
        ck_tile::index_t total_reduce_elements = H * W; // product of reduced dimensions

        RunGenericTest<4>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }
};

// Shape parameters for different test configurations
using Shape1_BlockWarps = ck_tile::sequence<4, 1>;
using Shape1_BlockTile  = ck_tile::sequence<128, 128>;
using Shape1_WarpTile   = ck_tile::sequence<32, 128>;
using Shape1_ThreadTile = ck_tile::sequence<8, 8>;

using Shape2_BlockWarps = ck_tile::sequence<2, 2>; // Cross-warp reduction test
using Shape2_BlockTile  = ck_tile::sequence<2, 1024>;
using Shape2_WarpTile   = ck_tile::sequence<1, 512>;
using Shape2_ThreadTile = ck_tile::sequence<1, 8>;

// Test configurations for different data types and operations
using TestConfig_F32_Add = std::tuple<float,
                                      float,
                                      float,
                                      ck_tile::ReduceOp::Add,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_ThreadTile>;

using TestConfig_F16_Add = std::tuple<ck_tile::half_t,
                                      float,
                                      ck_tile::half_t,
                                      ck_tile::ReduceOp::Add,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_ThreadTile>;

using TestConfig_F32_CrossWarp = std::tuple<float,
                                            float,
                                            float,
                                            ck_tile::ReduceOp::Add,
                                            Shape2_BlockWarps,
                                            Shape2_BlockTile,
                                            Shape2_WarpTile,
                                            Shape2_ThreadTile>;

using TestConfig_F32_Max = std::tuple<float,
                                      float,
                                      float,
                                      ck_tile::ReduceOp::Max,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_ThreadTile>;

using TestTypes = ::testing::
    Types<TestConfig_F32_Add, TestConfig_F16_Add, TestConfig_F32_CrossWarp, TestConfig_F32_Max>;

TYPED_TEST_SUITE(TestCkTileReduce, TestTypes);

// 2D Tests - Keep dim0, reduce dim1
TYPED_TEST(TestCkTileReduce, Test2D_KeepDim0_ReduceDim1_64x32)
{
    this->RunTest2D_KeepDim0_ReduceDim1(64, 32);
}

TYPED_TEST(TestCkTileReduce, Test2D_KeepDim0_ReduceDim1_1024x512)
{
    this->RunTest2D_KeepDim0_ReduceDim1(1024, 512);
}

// 3D Tests - Keep dim0, reduce dim1,2
TYPED_TEST(TestCkTileReduce, Test3D_KeepDim0_ReduceDim12_128x128x1)
{
    this->RunTest3D_KeepDim0_ReduceDim12(128, 128, 8);
}
// 3D Tests - Keep dim0,1, reduce dim1
TYPED_TEST(TestCkTileReduce, Test3D_KeepDim01_ReduceDim2_512x1024x16)
{
    this->RunTest3D_KeepDim01_ReduceDim2(512, 1024, 16);
}

// 4D Tests - Keep dim0,1, reduce dim2,3 (NCHW -> NC)
TYPED_TEST(TestCkTileReduce, Test4D_KeepDim01_ReduceDim23_32x256x16x16)
{
    this->RunTest4D_KeepDim01_ReduceDim23(32, 256, 16, 16);
}
// 4D Tests - Keep dim0,3, reduce dim1,2 (NHWC -> NC)
TYPED_TEST(TestCkTileReduce, Test4D_KeepDim03_ReduceDim12_16x32x32x128)
{
    this->RunTest4D_KeepDim03_ReduceDim12(16, 32, 32, 128);
}
