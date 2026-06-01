// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include "ck_tile/host/kernel_launch.hpp"

template <typename Tuple>
class TestCkTilePooling : public ::testing::Test
{
    protected:
    using InDataType      = std::tuple_element_t<0, Tuple>;
    using OutDataType     = std::tuple_element_t<1, Tuple>;
    using ComputeDataType = std::tuple_element_t<2, Tuple>;
    using ReduceOpType    = std::tuple_element_t<3, Tuple>;
    using BlockWarps_     = std::tuple_element_t<4, Tuple>;
    using BlockTile_      = std::tuple_element_t<5, Tuple>;
    using WarpTile_       = std::tuple_element_t<6, Tuple>;
    using ThreadTile_     = std::tuple_element_t<7, Tuple>;

    using TestPoolShape = ck_tile::PoolShape<BlockWarps_, BlockTile_, WarpTile_, ThreadTile_>;

    // 2D pooling configuration (NHWC)
    struct Config2D
    {
        ck_tile::index_t N, H, W, C;
        ck_tile::index_t Y, X;
        ck_tile::index_t Sy, Sx;
        ck_tile::index_t Dy, Dx;
        ck_tile::index_t LeftPy, LeftPx;
        ck_tile::index_t RightPy, RightPx;
        std::string name;
    };

    // 3D pooling configuration (NDHWC)
    struct Config3D
    {
        ck_tile::index_t N, D, H, W, C;
        ck_tile::index_t Z, Y, X;
        ck_tile::index_t Sz, Sy, Sx;
        ck_tile::index_t Dz, Dy, Dx;
        ck_tile::index_t LeftPz, LeftPy, LeftPx;
        ck_tile::index_t RightPz, RightPy, RightPx;
        std::string name;
    };

    bool RunPool2D(const Config2D& config)
    {
        std::cout << "Testing 2D: " << config.name << " ... ";

        const ck_tile::index_t Ys = (config.Y - 1) * config.Dy + 1;
        const ck_tile::index_t Xs = (config.X - 1) * config.Dx + 1;
        const ck_tile::index_t Ho =
            (config.H + config.LeftPy + config.RightPy - Ys) / config.Sy + 1;
        const ck_tile::index_t Wo =
            (config.W + config.LeftPx + config.RightPx - Xs) / config.Sx + 1;

        using IndexDataType = ck_tile::index_t;

        // Host tensors
        ck_tile::HostTensor<InDataType> h_in({config.N, config.H, config.W, config.C});
        ck_tile::HostTensor<OutDataType> h_out({config.N, Ho, Wo, config.C});
        ck_tile::HostTensor<OutDataType> h_out_ref({config.N, Ho, Wo, config.C});
        ck_tile::HostTensor<IndexDataType> h_out_index({config.N, Ho, Wo, config.C});
        ck_tile::HostTensor<IndexDataType> h_out_ref_index({config.N, Ho, Wo, config.C});

        // Initialize input with random data
        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);

        // Device memory
        ck_tile::DeviceMem d_in_mem(h_in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out_mem(h_out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out_index_mem(h_out_index.get_element_space_size_in_bytes());

        d_in_mem.ToDevice(h_in.data());
        d_out_mem.ToDevice(h_out.data());
        d_out_index_mem.ToDevice(h_out_index.data());

        constexpr ck_tile::index_t kBlockPerCu = 1;

        using Problem = ck_tile::PoolProblem<InDataType,
                                             OutDataType,
                                             ComputeDataType,
                                             IndexDataType,
                                             ReduceOpType,
                                             true,  // OutputIndex
                                             false, // PropagateNan
                                             TestPoolShape>;
        using Kernel  = ck_tile::PoolKernel<Problem>;

        const ck_tile::index_t kBlockSize = Kernel::BlockSize();

        // Shapes and strides (NHWC)
        const auto input_shape  = ck_tile::make_tuple(config.N, config.H, config.W, config.C);
        const auto output_shape = ck_tile::make_tuple(config.N, Ho, Wo, config.C);
        const auto input_strides =
            ck_tile::make_tuple(config.H * config.W * config.C, config.W * config.C, config.C, 1);
        const auto output_strides =
            ck_tile::make_tuple(Ho * Wo * config.C, Wo * config.C, config.C, 1);
        const auto window_spatial_lengths = ck_tile::make_tuple(config.Y, config.X);
        const auto window_strides         = ck_tile::make_tuple(config.Sy, config.Sx);
        const auto window_dilations       = ck_tile::make_tuple(config.Dy, config.Dx);
        const auto input_left_pads        = ck_tile::make_tuple(config.LeftPy, config.LeftPx);
        const auto input_right_pads       = ck_tile::make_tuple(config.RightPy, config.RightPx);

        auto host_args =
            ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_spatial_lengths)>{
                static_cast<InDataType*>(d_in_mem.GetDeviceBuffer()),
                static_cast<OutDataType*>(d_out_mem.GetDeviceBuffer()),
                static_cast<IndexDataType*>(d_out_index_mem.GetDeviceBuffer()),
                input_shape,
                output_shape,
                input_strides,
                output_strides,
                window_spatial_lengths,
                window_strides,
                window_dilations,
                input_left_pads,
                input_right_pads};

        auto kernel_args                 = Kernel::MakeKernelArgs(host_args);
        const ck_tile::index_t kGridSize = Kernel::CalculateGridSize(kernel_args);

        if(!Kernel::IsSupportedArgument(kernel_args))
        {
            return true;
        }

        // Run kernel
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, kGridSize, kBlockSize, 0, kernel_args));

        // Run reference
        ck_tile::reference_pool2d<InDataType,
                                  ComputeDataType,
                                  OutDataType,
                                  IndexDataType,
                                  ReduceOpType,
                                  decltype(input_shape),
                                  decltype(window_spatial_lengths),
                                  true>(
            h_in, h_out_ref, h_out_ref_index, kernel_args, ReduceOpType{});

        d_out_mem.FromDevice(h_out.data());
        d_out_index_mem.FromDevice(h_out_index.data());

        // Validate results
        bool pass_value =
            ck_tile::check_err(h_out, h_out_ref, "Error: Incorrect values!", 1e-5, 1e-5);
        bool pass_index = ck_tile::check_err(
            h_out_index, h_out_ref_index, "Error: Incorrect indices!", 1e-5, 1e-5);

        std::cout << (pass_value && pass_index ? "PASS" : "FAIL") << std::endl;
        return pass_value && pass_index;
    }

    bool RunPool3D(const Config3D& config)
    {
        std::cout << "Testing 3D: " << config.name << " ... ";

        const ck_tile::index_t Zs = (config.Z - 1) * config.Dz + 1;
        const ck_tile::index_t Ys = (config.Y - 1) * config.Dy + 1;
        const ck_tile::index_t Xs = (config.X - 1) * config.Dx + 1;
        const ck_tile::index_t Do =
            (config.D + config.LeftPz + config.RightPz - Zs) / config.Sz + 1;
        const ck_tile::index_t Ho =
            (config.H + config.LeftPy + config.RightPy - Ys) / config.Sy + 1;
        const ck_tile::index_t Wo =
            (config.W + config.LeftPx + config.RightPx - Xs) / config.Sx + 1;

        const auto input_shape =
            ck_tile::make_tuple(config.N, config.D, config.H, config.W, config.C);
        const auto output_shape   = ck_tile::make_tuple(config.N, Do, Ho, Wo, config.C);
        const auto input_strides  = ck_tile::make_tuple(config.D * config.H * config.W * config.C,
                                                       config.H * config.W * config.C,
                                                       config.W * config.C,
                                                       config.C,
                                                       1);
        const auto output_strides = ck_tile::make_tuple(
            Do * Ho * Wo * config.C, Ho * Wo * config.C, Wo * config.C, config.C, 1);
        const auto window_spatial_lengths = ck_tile::make_tuple(config.Z, config.Y, config.X);
        const auto window_strides         = ck_tile::make_tuple(config.Sz, config.Sy, config.Sx);
        const auto window_dilations       = ck_tile::make_tuple(config.Dz, config.Dy, config.Dx);
        const auto input_left_pads =
            ck_tile::make_tuple(config.LeftPz, config.LeftPy, config.LeftPx);
        const auto input_right_pads =
            ck_tile::make_tuple(config.RightPz, config.RightPy, config.RightPx);

        using IndexDataType = ck_tile::index_t;

        ck_tile::HostTensor<InDataType> h_in({config.N, config.D, config.H, config.W, config.C},
                                             {config.D * config.H * config.W * config.C,
                                              config.H * config.W * config.C,
                                              config.W * config.C,
                                              config.C,
                                              1});
        ck_tile::HostTensor<OutDataType> h_out(
            {config.N, Do, Ho, Wo, config.C},
            {Do * Ho * Wo * config.C, Ho * Wo * config.C, Wo * config.C, config.C, 1});
        ck_tile::HostTensor<OutDataType> h_out_ref(
            {config.N, Do, Ho, Wo, config.C},
            {Do * Ho * Wo * config.C, Ho * Wo * config.C, Wo * config.C, config.C, 1});
        ck_tile::HostTensor<IndexDataType> h_out_index(
            {config.N, Do, Ho, Wo, config.C},
            {Do * Ho * Wo * config.C, Ho * Wo * config.C, Wo * config.C, config.C, 1});
        ck_tile::HostTensor<IndexDataType> h_out_ref_index(
            {config.N, Do, Ho, Wo, config.C},
            {Do * Ho * Wo * config.C, Ho * Wo * config.C, Wo * config.C, config.C, 1});

        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);
        h_out.SetZero();
        h_out_ref.SetZero();

        ck_tile::DeviceMem d_in_mem(h_in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out_mem(h_out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out_index_mem(h_out_index.get_element_space_size_in_bytes());

        d_in_mem.ToDevice(h_in.data());
        d_out_mem.ToDevice(h_out.data());
        d_out_index_mem.ToDevice(h_out_index.data());

        using Problem = ck_tile::PoolProblem<InDataType,
                                             OutDataType,
                                             ComputeDataType,
                                             IndexDataType,
                                             ReduceOpType,
                                             true,  // OutputIndex
                                             false, // PropagateNan
                                             TestPoolShape>;
        using Kernel  = ck_tile::PoolKernel<Problem>;

        constexpr ck_tile::index_t kBlockPerCu = 1;
        const ck_tile::index_t kBlockSize      = Kernel::BlockSize();

        auto host_args =
            ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_spatial_lengths)>{
                static_cast<InDataType*>(d_in_mem.GetDeviceBuffer()),
                static_cast<OutDataType*>(d_out_mem.GetDeviceBuffer()),
                static_cast<IndexDataType*>(d_out_index_mem.GetDeviceBuffer()),
                input_shape,
                output_shape,
                input_strides,
                output_strides,
                window_spatial_lengths,
                window_strides,
                window_dilations,
                input_left_pads,
                input_right_pads};

        auto kernel_args = Kernel::MakeKernelArgs(host_args);

        const ck_tile::index_t kGridSize = Kernel::CalculateGridSize(kernel_args);

        if(!Kernel::IsSupportedArgument(kernel_args))
        {
            return true;
        }

        // Run kernel
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, kGridSize, kBlockSize, 0, kernel_args));

        // Run reference implementation
        ck_tile::reference_pool3d<InDataType,
                                  ComputeDataType,
                                  OutDataType,
                                  IndexDataType,
                                  ReduceOpType,
                                  decltype(input_shape),
                                  decltype(window_spatial_lengths),
                                  true>(
            h_in, h_out_ref, h_out_ref_index, kernel_args, ReduceOpType{});

        d_out_mem.FromDevice(h_out.data());
        d_out_index_mem.FromDevice(h_out_index.data());

        // Validate results
        bool pass_value =
            ck_tile::check_err(h_out, h_out_ref, "Error: Incorrect values!", 1e-5, 1e-5);
        bool pass_index = ck_tile::check_err(
            h_out_index, h_out_ref_index, "Error: Incorrect indices!", 1e-5, 1e-5);

        std::cout << (pass_value && pass_index ? "PASS" : "FAIL") << std::endl;
        return pass_value && pass_index;
    }
};

using Shape1_BlockWarps = ck_tile::sequence<1, 1>;
using Shape1_BlockTile  = ck_tile::sequence<128, 1>;
using Shape1_WarpTile   = ck_tile::sequence<128, 1>;
using Shape1_ThreadTile = ck_tile::sequence<2, 1>;

// Cross-warp configuration
using Shape2_BlockWarps = ck_tile::sequence<2, 2>;
using Shape2_BlockTile  = ck_tile::sequence<2, 1024>;
using Shape2_WarpTile   = ck_tile::sequence<1, 512>;
using Shape2_ThreadTile = ck_tile::sequence<1, 8>;

// Test configurations for different data types and operations
using TestConfig_F32_Max = std::tuple<float,
                                      float,
                                      float,
                                      ck_tile::ReduceOp::Max,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_ThreadTile>;

using TestConfig_F16_Max = std::tuple<ck_tile::half_t,
                                      ck_tile::half_t,
                                      float,
                                      ck_tile::ReduceOp::Max,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_ThreadTile>;

using TestConfig_F32_CrossWarp = std::tuple<float,
                                            float,
                                            float,
                                            ck_tile::ReduceOp::Max,
                                            Shape2_BlockWarps,
                                            Shape2_BlockTile,
                                            Shape2_WarpTile,
                                            Shape2_ThreadTile>;

using TestTypes =
    ::testing::Types<TestConfig_F32_Max, TestConfig_F16_Max, TestConfig_F32_CrossWarp>;

TYPED_TEST_SUITE(TestCkTilePooling, TestTypes);

// 2D Pooling Tests (NHWC)
TYPED_TEST(TestCkTilePooling, Pool2D_2x2)
{
    typename TestFixture::Config2D config = {1,  // N - batch size
                                             8,  // H - height dimension
                                             8,  // W - width dimension
                                             32, // C - channel dimension
                                             2,  // Y - pooling window height
                                             2,  // X - pooling window width
                                             2,  // Sy - window stride height
                                             2,  // Sx - window stride width
                                             1,  // Dy - window dilation height
                                             1,  // Dx - window dilation width
                                             0,  // LeftPy - left padding height
                                             0,  // LeftPx - left padding width
                                             0,  // RightPy - right padding height
                                             0,  // RightPx - right padding width
                                             "2x2 pooling NHWC"};
    bool pass                             = this->RunPool2D(config);
    EXPECT_TRUE(pass);
}

TYPED_TEST(TestCkTilePooling, Pool2D_3x3_WithPadding)
{
    typename TestFixture::Config2D config = {2,  // N - batch size
                                             16, // H - height dimension
                                             16, // W - width dimension
                                             32, // C - channel dimension
                                             3,  // Y - pooling window height
                                             3,  // X - pooling window width
                                             2,  // Sy - window stride height
                                             2,  // Sx - window stride width
                                             1,  // Dy - window dilation height
                                             1,  // Dx - window dilation width
                                             1,  // LeftPy - left padding height
                                             1,  // LeftPx - left padding width
                                             1,  // RightPy - right padding height
                                             1,  // RightPx - right padding width
                                             "3x3 pooling NHWC with padding"};
    bool pass                             = this->RunPool2D(config);
    EXPECT_TRUE(pass);
}

// 3D Pooling Tests (NDHWC)
TYPED_TEST(TestCkTilePooling, Pool3D_2x2x2)
{
    typename TestFixture::Config3D config = {1,  // N - batch size
                                             4,  // D - depth dimension
                                             4,  // H - height dimension
                                             4,  // W - width dimension
                                             32, // C - channel dimension
                                             2,  // Z - pooling window depth
                                             2,  // Y - pooling window height
                                             2,  // X - pooling window width
                                             2,  // Sz - window stride depth
                                             2,  // Sy - window stride height
                                             2,  // Sx - window stride width
                                             1,  // Dz - window dilation depth
                                             1,  // Dy - window dilation height
                                             1,  // Dx - window dilation width
                                             0,  // LeftPz - left padding depth
                                             0,  // LeftPy - left padding height
                                             0,  // LeftPx - left padding width
                                             0,  // RightPz - right padding depth
                                             0,  // RightPy - right padding height
                                             0,  // RightPx - right padding width
                                             "2x2x2 pooling NDHWC"};
    bool pass                             = this->RunPool3D(config);
    EXPECT_TRUE(pass);
}

TYPED_TEST(TestCkTilePooling, Pool3D_3x3x3)
{
    typename TestFixture::Config3D config = {2,   // N - batch size
                                             16,  // D - depth dimension
                                             16,  // H - height dimension
                                             16,  // W - width dimension
                                             128, // C - channel dimension
                                             3,   // Z - pooling window depth
                                             3,   // Y - pooling window height
                                             3,   // X - pooling window width
                                             2,   // Sz - window stride depth
                                             2,   // Sy - window stride height
                                             2,   // Sx - window stride width
                                             1,   // Dz - window dilation depth
                                             1,   // Dy - window dilation height
                                             1,   // Dx - window dilation width
                                             1,   // LeftPz - left padding depth
                                             1,   // LeftPy - left padding height
                                             1,   // LeftPx - left padding width
                                             1,   // RightPz - right padding depth
                                             1,   // RightPy - right padding height
                                             1,   // RightPx - right padding width
                                             "3x3x3 pooling NDHWC with padding"};
    bool pass                             = this->RunPool3D(config);
    EXPECT_TRUE(pass);
}
