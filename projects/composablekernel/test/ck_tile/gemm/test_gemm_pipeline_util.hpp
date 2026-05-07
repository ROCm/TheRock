// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

enum struct GemmPipelineType
{
    Mem,
    CompV3,
    CompV4,
    CompV6,
    CompAsync
};

template <GemmPipelineType PT, typename Problem>
struct GemmPipelineTypeSelector;

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::Mem, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrMem<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrMem<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrMem"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV3, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompV3"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV4, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompV4<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompV4"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV6, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV6<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompV6<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompV6"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompAsync, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompAsync<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompAsync<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompAsync"; }
};

template <typename Tuple, typename Derived>
class TestCkTileGemmPipeline : public ::testing::Test
{
    public:
    using ALayout                      = std::tuple_element_t<0, Tuple>;
    using BLayout                      = std::tuple_element_t<1, Tuple>;
    using CLayout                      = std::tuple_element_t<2, Tuple>;
    using ADataType                    = std::tuple_element_t<3, Tuple>;
    using BDataType                    = std::tuple_element_t<4, Tuple>;
    using AccDataType                  = std::tuple_element_t<5, Tuple>;
    using CDataType                    = std::tuple_element_t<6, Tuple>;
    static constexpr auto Scheduler    = std::tuple_element_t<13, Tuple>::value;
    static constexpr auto PipelineType = std::tuple_element_t<14, Tuple>::value;

    static constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, Tuple>{};
    static constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, Tuple>{};
    static constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, Tuple>{};

    static constexpr ck_tile::index_t M_Warp_Tile = std::tuple_element_t<10, Tuple>{};
    static constexpr ck_tile::index_t N_Warp_Tile = std::tuple_element_t<11, Tuple>{};
    static constexpr ck_tile::index_t K_Warp_Tile = std::tuple_element_t<12, Tuple>{};

    using DsLayout   = ck_tile::tuple<>;
    using DsDataType = ck_tile::tuple<>;

    static constexpr bool Persistent =
        ck_tile::tuple_element_or_default_t<Tuple, 15, std::false_type>::value;

    protected:
    template <bool PadM, bool PadN, bool PadK, bool Preshuffle>
    void invoke_gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr bool kPadM      = PadM;
        constexpr bool kPadN      = PadN;
        constexpr bool kPadK      = PadK;
        constexpr bool preshuffle = Preshuffle;

        constexpr bool DoubleSmemBuffer          = (PipelineType == GemmPipelineType::CompV4 ||
                                           PipelineType == GemmPipelineType::CompAsync);
        constexpr bool TransposeC                = false;
        static constexpr bool StructuredSparsity = false;
        static constexpr bool NumWaveGroup       = 1;

        // TODO: For now - but this should also be a test parameter

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        // ===============================================

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC,
                                                                     StructuredSparsity,
                                                                     Persistent,
                                                                     NumWaveGroup,
                                                                     preshuffle>;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           Scheduler>;

        using GemmPipeline =
            typename GemmPipelineTypeSelector<PipelineType, UniversalGemmProblem>::pipeline;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             1,     /*kNumWaveGroups_*/
                                             false, /*FixedVectorSize_*/
                                             1,     /*VectorSizeC_*/
                                             false, /*TiledMMAPermuteN_*/
                                             1,     /*BlockedXDLN_PerWarp_*/
                                             DoubleSmemBuffer /*DoubleSmemBuffer*/>>;

        using Kernel     = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        const auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 blocks = Kernel::BlockSize();
        dim3 grids;
        if constexpr(Persistent)
        {
            grids = Kernel::MaxOccupancyGridSize(s);
        }
        else
        {
            grids = Kernel::GridSize(args.M, args.N, args.k_batch);
        }

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:" << " grid: {" << grids.x << ", " << grids.y
                      << ", " << grids.z << "}" << ", blocks: {" << blocks.x << ", " << blocks.y
                      << ", " << blocks.z << "}" << std::endl;
        }

        ck_tile::ignore = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }

    public:
    std::vector<int> k_batches_;

    void SetUp() override
    {
        if constexpr(!Derived::check_data_type())
        {
            GTEST_SKIP() << "Unsupported data type combination for gemm pipeline test.";
        }
        if constexpr(PipelineType == GemmPipelineType::CompV4 ||
                     std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Only do k_batch = 1 when pipeline is CompV4, or BDataType is I4
            k_batches_ = {1};
        }
        else
        {
            // Otherwise, use k_batch = 1 and 2
            k_batches_ = {1, 2};
        }
    }

    template <bool PadM = true, bool PadN = true, bool PadK = true, bool Preshuffle = false>
    void Run(const int M,
             const int N,
             const int K,
             const int StrideA = 0,
             const int StrideB = 0,
             const int StrideC = 0)
    {
        // Some unsupported tests don't compile, so we check here before attempting to.
        if constexpr(Derived::check_data_type())
        {
            for(auto kb : k_batches_)
            {
                RunSingle<PadM, PadN, PadK, Preshuffle>(M, N, K, StrideA, StrideB, StrideC, kb);
            }
        }
    }

    template <bool PadM, bool PadN, bool PadK, bool Preshuffle>
    void RunSingle(const int M,
                   const int N,
                   const int K,
                   const int StrideA,
                   const int StrideB,
                   const int StrideC,
                   int kbatch = 1)
    {
        ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, StrideA, is_row_major(ALayout{}));
        ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, StrideB, is_row_major(BLayout{}));
        ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, StrideC, is_row_major(CLayout{}));

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, 11939}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, 11940}(b_k_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
            permute_vectors_i4x4_b(b_k_n_dev);
            b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
        }
        else
        {
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::GemmHostArgs args = {a_m_k_dev_buf.GetDeviceBuffer(),
                                      b_k_n_dev_buf.GetDeviceBuffer(),
                                      c_m_n_dev_buf.GetDeviceBuffer(),
                                      kbatch,
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      stride_C};

        invoke_gemm<PadM, PadN, PadK, Preshuffle>(args, ck_tile::stream_config{nullptr, false});

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool pass = true;

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));
        EXPECT_TRUE(pass);
    }
};
