// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <vector>
#include "gtest/gtest.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "test_grouped_gemm_util.hpp"
#include "test_grouped_gemm_interface_xdl.hpp"

class TestGGemmSplitKInterface_MKNKMN : public ::testing::Test
{
    protected:
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using ALayout = Row;
    using BLayout = Col;
    using ELayout = Row;

    static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

    template <ck::tensor_operation::device::GemmSpecialization GemmSpec,
              ck::index_t KPerBlock,
              ck::index_t K1,
              ck::index_t ABlockTransferSrcScalarPerVector,
              ck::index_t BBlockTransferSrcScalarPerVector,
              ck::index_t CDEBlockTransferScalarPerVector_NPerBlock>
    using GGemmInstance =
        ck::test::DeviceGroupedGemmSplitkInstanceWrapper<ALayout,
                                                         BLayout,
                                                         ELayout,
                                                         GemmSpec,
                                                         KPerBlock,
                                                         K1,
                                                         ABlockTransferSrcScalarPerVector,
                                                         BBlockTransferSrcScalarPerVector,
                                                         CDEBlockTransferScalarPerVector_NPerBlock>;

    using DefaultGGemmInstance = GGemmInstance<GemmDefault, 64, 16, 4, 8, 4>;
};

TEST_F(TestGGemmSplitKInterface_MKNKMN, TileSize)
{
    std::vector<int> Ms{128, 256, 188, 512};
    constexpr int N = 256;
    constexpr int K = 128;

    std::vector<int> Ns(Ms.size(), N);
    std::vector<int> Ks(Ms.size(), K);
    std::vector<int> StrideAs(Ms.size(), K);
    std::vector<int> StrideBs(Ms.size(), K);
    std::vector<int> StrideCs(Ms.size(), N);

    // M % MPerBlock
    EXPECT_FALSE(DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));

    Ms = std::vector<int>{256, 128, 128, 512};
    Ns = std::vector<int>{256, 177, 128, 512};
    // N % NPerBlock
    EXPECT_FALSE(DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));
}

TEST_F(TestGGemmSplitKInterface_MKNKMN, VectorLoadWidth)
{
    static constexpr auto GemmMNKPadding =
        ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    using PaddedGGemmInstance = GGemmInstance<GemmMNKPadding, 64, 16, 4, 8, 4>;

    std::vector<int> Ms{128, 256, 256, 512};
    constexpr int N = 256;
    constexpr int K = 512;

    std::vector<int> Ns(Ms.size(), N);
    std::vector<int> Ks(Ms.size(), K);
    std::vector<int> StrideAs(Ms.size(), K);
    std::vector<int> StrideBs(Ms.size(), K);
    std::vector<int> StrideCs(Ms.size(), N);

    // K % ABlockTransferSrcScalarPerVector
    Ks = std::vector<int>{256, 177, 128, 512};
    EXPECT_FALSE(PaddedGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));

    Ks = std::vector<int>{256, 164, 128, 512};
    // K % BBlockTransferSrcScalarPerVector
    EXPECT_FALSE(PaddedGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));

    Ks = std::vector<int>(4, 128);
    Ns = std::vector<int>{256, 127, 128, 512};
    // N % CBlockTransferScalarPerVector_NWaveNPerXDL
    EXPECT_FALSE(PaddedGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));
}

TEST_F(TestGGemmSplitKInterface_MKNKMN, KLoops)
{
    std::vector<int> Ms{128, 256, 256, 512};
    constexpr int N      = 256;
    constexpr int K      = 128;
    constexpr int kbatch = 4;

    std::vector<int> Ns(Ms.size(), N);
    std::vector<int> Ks(Ms.size(), K);
    std::vector<int> StrideAs(Ms.size(), K);
    std::vector<int> StrideBs(Ms.size(), K);
    std::vector<int> StrideCs(Ms.size(), N);

    // kloops % 2
    Ks = std::vector<int>{256, 512, 320, 768};
    EXPECT_FALSE(
        DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, kbatch));

    if(!ck::is_gfx11_supported())
    {
        Ks = std::vector<int>{256, 512, 768, 1536};
        EXPECT_TRUE(
            DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, kbatch));

        // Not all gemms have same value for main_k0_block_loop!
        Ks = std::vector<int>{256, 512, 512, 512};
        EXPECT_THROW(DefaultGGemmInstance{}.Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, kbatch),
                     std::runtime_error);
    }
}

class TestGGemmSplitKInterface_KMKNNM : public ::testing::Test
{
    protected:
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using ALayout = Col;
    using BLayout = Row;
    using ELayout = Col;

    static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

    template <ck::tensor_operation::device::GemmSpecialization GemmSpec,
              ck::index_t KPerBlock,
              ck::index_t K1,
              ck::index_t ABlockTransferSrcScalarPerVector,
              ck::index_t BBlockTransferSrcScalarPerVector,
              ck::index_t CDEBlockTransferScalarPerVector_NPerBlock>
    using GGemmInstance =
        ck::test::DeviceGroupedGemmSplitkInstanceWrapper<ALayout,
                                                         BLayout,
                                                         ELayout,
                                                         GemmSpec,
                                                         KPerBlock,
                                                         K1,
                                                         ABlockTransferSrcScalarPerVector,
                                                         BBlockTransferSrcScalarPerVector,
                                                         CDEBlockTransferScalarPerVector_NPerBlock>;

    using DefaultGGemmInstance = GGemmInstance<GemmDefault, 64, 16, 4, 8, 4>;
};

TEST_F(TestGGemmSplitKInterface_KMKNNM, TileSize)
{
    std::vector<int> Ms{128, 256, 188, 512};
    constexpr int N = 256;
    constexpr int K = 128;

    std::vector<int> Ns(Ms.size(), N);
    std::vector<int> Ks(Ms.size(), K);
    std::vector<int> StrideAs(Ms.size(), K);
    std::vector<int> StrideBs(Ms.size(), K);
    std::vector<int> StrideCs(Ms.size(), N);

    // M % MPerBlock
    EXPECT_FALSE(DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));

    Ms = std::vector<int>{128, 256, 256, 512};
    Ns = std::vector<int>{256, 177, 128, 512};
    // N % NPerBlock
    EXPECT_FALSE(DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));
}

TEST_F(TestGGemmSplitKInterface_KMKNNM, VectorLoadWidth)
{
    static constexpr auto GemmMNKPadding =
        ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    using PaddedGGemmInstance = GGemmInstance<GemmMNKPadding, 64, 16, 2, 8, 4>;

    std::vector<int> Ms{128, 256, 256, 512};
    constexpr int N = 256;
    constexpr int K = 512;

    std::vector<int> Ns(Ms.size(), N);
    std::vector<int> Ks(Ms.size(), K);
    std::vector<int> StrideAs(Ms.size(), K);
    std::vector<int> StrideBs(Ms.size(), K);
    std::vector<int> StrideCs(Ms.size(), N);

    // M % ABlockTransferSrcScalarPerVector
    Ms = std::vector<int>{256, 177, 128, 512};
    EXPECT_FALSE(PaddedGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));

    Ms = std::vector<int>{128, 256, 256, 512};
    Ns = std::vector<int>{256, 164, 128, 512};
    // N % BBlockTransferSrcScalarPerVector
    EXPECT_FALSE(PaddedGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));

    Ns = std::vector<int>{128, 256, 256, 512};
    Ms = std::vector<int>{256, 130, 128, 512};
    // M % CBlockTransferScalarPerVector_NWaveNPerXDL
    EXPECT_FALSE(PaddedGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));
}
