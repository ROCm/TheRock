// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

#include "ck_tile/ops/batched_transpose.hpp"

enum class PipelineTag : ck_tile::index_t
{
    Universal,
    LDSLoadTranspose,
};

template <PipelineTag kPipelineId_>
struct PipelineSelector
{
};

template <>
struct PipelineSelector<PipelineTag::Universal>
{
    template <typename DataType, typename BlockTile, typename WarpLayout, bool kPadM, bool kPadN>
    using Problem = ck_tile::BatchedTransposeProblem<DataType, BlockTile, WarpLayout, kPadM, kPadN>;

    using Policy = ck_tile::BatchedTransposePolicy;

    template <typename Problem_>
    using Pipeline = ck_tile::BatchedTransposePipeline<Problem_, Policy>;
};

template <>
struct PipelineSelector<PipelineTag::LDSLoadTranspose>
{
    template <typename DataType, typename BlockTile, typename WarpLayout, bool kPadM, bool kPadN>
    using Problem =
        ck_tile::BatchedTransposeLdsProblem<DataType, BlockTile, WarpLayout, kPadM, kPadN>;

    using Policy = ck_tile::BatchedTransposeLdsPolicy;

    template <typename Problem_>
    using Pipeline = ck_tile::BatchedTransposeLdsPipeline<Problem_, Policy>;
};

template <typename DataType_,
          PipelineTag kPipelineId_     = PipelineTag::Universal,
          ck_tile::index_t kBlockX_    = 64,
          ck_tile::index_t kBlockY_    = 64,
          ck_tile::index_t kNumWarpsX_ = 1,
          ck_tile::index_t kNumWarpsY_ = 1,
          bool kPadM_                  = true,
          bool kPadN_                  = true>
struct PipelineConfig
{
    using DataType                               = DataType_;
    using BlockTile                              = ck_tile::sequence<kBlockX_, kBlockY_>;
    using WarpLayout                             = ck_tile::sequence<kNumWarpsX_, kNumWarpsY_>;
    static constexpr bool kPadM                  = kPadM_;
    static constexpr bool kPadN                  = kPadN_;
    static constexpr PipelineTag kPipelineId     = kPipelineId_;
    static constexpr ck_tile::index_t kBlockX    = kBlockX_;
    static constexpr ck_tile::index_t kBlockY    = kBlockY_;
    static constexpr ck_tile::index_t kNumWarpsX = kNumWarpsX_;
    static constexpr ck_tile::index_t kNumWarpsY = kNumWarpsY_;

    using Problem = typename PipelineSelector<
        kPipelineId_>::template Problem<DataType, BlockTile, WarpLayout, kPadM, kPadN>;
    using Pipeline = typename PipelineSelector<kPipelineId_>::template Pipeline<Problem>;
    using Kernel   = ck_tile::BatchedTransposeKernel<Pipeline>;
};

template <typename Config>
class TestCkTileBatchedTranspose //              N    C    H    W    layout_in==NCHW
    : public ::testing::TestWithParam<std::tuple<int, int, int, int, bool>>
{
    protected:
    void Run(std::tuple<int, int, int, int, bool> param)
    {
        using DataType                     = typename Config::DataType;
        const auto [N, C, H, W, nchw2nhwc] = param;
        const std::string layout_in        = nchw2nhwc ? "NCHW" : "NHWC";
        const std::string layout_out       = nchw2nhwc ? "NHWC" : "NCHW";
        const auto X_dim = nchw2nhwc ? std::array{N, C, H, W} : std::array{N, H, W, C};
        const auto X_stride =
            nchw2nhwc ? std::array{C * H * W, H * W, W, 1} : std::array{C * H * W, C * W, C, 1};
        ck_tile::HostTensor<DataType> x_host(X_dim, X_stride);
        const auto Y_dim = nchw2nhwc ? std::array{N, H, W, C} : std::array{N, C, H, W};
        const auto Y_stride =
            nchw2nhwc ? std::array{C * H * W, C * W, C, 1} : std::array{C * H * W, H * W, W, 1};
        ck_tile::HostTensor<DataType> y_host(Y_dim, Y_stride);
        ck_tile::HostTensor<DataType> y_ref(Y_dim, Y_stride);

        ck_tile::FillUniformDistribution<DataType>{-.5f, .5f}(x_host);
        ck_tile::FillConstant<DataType>{-37}(y_host);

        ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem y_dev(y_host.get_element_space_size_in_bytes());
        x_dev.ToDevice(x_host.data());
        y_dev.ToDevice(y_host.data());

        using Kernel = typename Config::Kernel;

        const ck_tile::index_t height = nchw2nhwc ? C : H * W;
        const ck_tile::index_t width  = nchw2nhwc ? H * W : C;

        if(height % Config::kBlockX != 0 && !Config::kPadM)
        {
            GTEST_SKIP_("Input cannot be covered with block tiles and Kernel does not force height "
                        "padding");
        }

        if(width % Config::kBlockY != 0 && !Config::kPadN)
        {
            GTEST_SKIP_(
                "Input cannot be covered with block tiles and Kernel does not force width padding");
        }

        const auto device_name = ck_tile::get_device_name();

        if(Config::kPipelineId == PipelineTag::LDSLoadTranspose &&
           device_name.find("gfx950") == std::string::npos)
        {
            GTEST_SKIP_("LDS Load Transpose cannot be launched with this device");
        }

        const auto host_args = ck_tile::BatchedTransposeHostArgs{x_dev.GetDeviceBuffer(),
                                                                 y_dev.GetDeviceBuffer(),
                                                                 N,
                                                                 height,
                                                                 width,
                                                                 height * width,
                                                                 Config::BlockTile::at(0),
                                                                 Config::BlockTile::at(1)};
        auto kargs           = Kernel::MakeKargs(host_args);

        auto sc               = ck_tile::stream_config{};
        const dim3 grid_size  = Kernel::GridSize(host_args);
        const dim3 block_size = Kernel::BlockSize();
        ck_tile::launch_kernel(sc,
                               ck_tile::make_kernel<1>(Kernel{}, grid_size, block_size, 0, kargs));

        y_dev.FromDevice(y_host.data());
        ck_tile::reference_batched_transpose<DataType>(x_host, y_ref, layout_in, layout_out);

        std::ostringstream message;
        message << "N=" << N << " C=" << C << " H=" << H << " W=" << W << " layout_in=" << layout_in
                << " layout_out=" << layout_out << " grid_size={" << grid_size.x << ", "
                << grid_size.y << ", " << grid_size.z << "} block_size=" << block_size.x
                << " device_name=" << device_name;

        // NB: order of output and reference matters
        bool pass = ck_tile::check_err(
            /* out */ y_host,
            /* ref */ y_ref,
            message.str(),
            /* rtol */ 0,
            /* atol */ 0,
            /* allow inf */ false);

        EXPECT_TRUE(pass);
    }
};

// clang-format off
// the default indent is not sane
static const auto kTestingValues = ::testing::Values(
//             N  C   H  W   layout_in==NCHW    
    std::tuple{1, 32, 1, 32, true},
    std::tuple{1, 64, 1, 64, true},
    std::tuple{1, 32, 1, 64, true},
    std::tuple{1, 64, 1, 32, true},
    std::tuple{2, 12, 1, 32, false},
    std::tuple{3, 1334, 1, 37, false},
    std::tuple{4, 27, 1, 32, true},
    std::tuple{5, 1234, 1, 12, true},
    std::tuple{1, 1, 1, 1, true},
    std::tuple{1, 1, 1, 1, false},
    std::tuple{17, 1024, 64, 64, true},
    std::tuple{17, 1024, 64, 64, false},
    std::tuple{16, 64, 32, 128, true},
    std::tuple{16, 64, 128, 32, false},
    std::tuple{1, 2048, 1, 1, true},
    std::tuple{1, 2048, 1, 1, false},
    std::tuple{1, 1, 1024, 1024, true},
    std::tuple{1, 1, 1024, 1024, false},
    std::tuple{8, 16, 8, 16, true},
    std::tuple{8, 16, 8, 16, false},
    std::tuple{1, 64, 1, 1024, true},
    std::tuple{1, 64, 1024, 1, false}
);
// clang-format on

class CaseHalf : public TestCkTileBatchedTranspose<PipelineConfig<ck_tile::half_t>>
{
};

class CaseByte : public TestCkTileBatchedTranspose<PipelineConfig<ck_tile::fp8_t>>
{
};

class CaseWord : public TestCkTileBatchedTranspose<PipelineConfig<float>>
{
};

class CaseHalfLoadTranspose : public TestCkTileBatchedTranspose<
                                  PipelineConfig<ck_tile::half_t, PipelineTag::LDSLoadTranspose>>
{
};

class CaseByteLoadTranspose : public TestCkTileBatchedTranspose<
                                  PipelineConfig<ck_tile::fp8_t, PipelineTag::LDSLoadTranspose>>
{
};

class CaseHalfPad
    : public TestCkTileBatchedTranspose<
          PipelineConfig<ck_tile::half_t, PipelineTag::Universal, 64, 64, 1, 1, false, false>>
{
};

class CaseHalfPadLoadTranspose
    : public TestCkTileBatchedTranspose<PipelineConfig<ck_tile::half_t,
                                                       PipelineTag::LDSLoadTranspose,
                                                       64,
                                                       64,
                                                       1,
                                                       1,
                                                       false,
                                                       false>>
{
};

class CaseHalfPadMultiWarp
    : public TestCkTileBatchedTranspose<
          PipelineConfig<ck_tile::half_t, PipelineTag::Universal, 64, 64, 2, 2, false, false>>
{
};

class CaseHalfPadMultiWarpLoadTranspose
    : public TestCkTileBatchedTranspose<PipelineConfig<ck_tile::half_t,
                                                       PipelineTag::LDSLoadTranspose,
                                                       64,
                                                       64,
                                                       2,
                                                       2,
                                                       false,
                                                       false>>
{
};

class CaseHalfPadMultiWarp128MNLoadTranspose
    : public TestCkTileBatchedTranspose<PipelineConfig<ck_tile::half_t,
                                                       PipelineTag::LDSLoadTranspose,
                                                       128,
                                                       128,
                                                       2,
                                                       2,
                                                       false,
                                                       false>>
{
};

class CaseHalfPadMultiWarp128MN
    : public TestCkTileBatchedTranspose<
          PipelineConfig<ck_tile::half_t, PipelineTag::Universal, 128, 128, 2, 2, false, false>>
{
};

class CaseHalfPadRectTile1
    : public TestCkTileBatchedTranspose<
          PipelineConfig<ck_tile::half_t, PipelineTag::Universal, 32, 64, 1, 1, false, false>>
{
};

class CaseHalfPadRectTile2
    : public TestCkTileBatchedTranspose<
          PipelineConfig<ck_tile::half_t, PipelineTag::Universal, 64, 32, 1, 1, false, false>>
{
};

class CaseHalfPadRectTile1LoadTranspose
    : public TestCkTileBatchedTranspose<PipelineConfig<ck_tile::half_t,
                                                       PipelineTag::LDSLoadTranspose,
                                                       32,
                                                       64,
                                                       1,
                                                       1,
                                                       false,
                                                       false>>
{
};

class CaseHalfPadRectTile2LoadTranspose
    : public TestCkTileBatchedTranspose<PipelineConfig<ck_tile::half_t,
                                                       PipelineTag::LDSLoadTranspose,
                                                       64,
                                                       32,
                                                       1,
                                                       1,
                                                       false,
                                                       false>>
{
};

class CaseBytePadRectTile
    : public TestCkTileBatchedTranspose<
          PipelineConfig<ck_tile::fp8_t, PipelineTag::Universal, 256, 32, 2, 2, false, false>>
{
};

TEST_P(CaseHalf, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseByte, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseWord, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfLoadTranspose, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseByteLoadTranspose, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPad, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPadLoadTranspose, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPadMultiWarp, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPadMultiWarpLoadTranspose, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPadMultiWarp128MN, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPadMultiWarp128MNLoadTranspose, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPadRectTile1, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPadRectTile1LoadTranspose, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPadRectTile2, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseHalfPadRectTile2LoadTranspose, TestCorrectness) { this->Run(GetParam()); }
TEST_P(CaseBytePadRectTile, TestCorrectness) { this->Run(GetParam()); }

// clang-format off
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalf, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseByte, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseWord, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfLoadTranspose, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseByteLoadTranspose, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPad, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPadLoadTranspose, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPadMultiWarp, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPadMultiWarpLoadTranspose, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPadMultiWarp128MN, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPadMultiWarp128MNLoadTranspose, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPadRectTile1, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPadRectTile1LoadTranspose, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPadRectTile2, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseHalfPadRectTile2LoadTranspose, kTestingValues);
INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite, CaseBytePadRectTile, kTestingValues);

// clang-format on
