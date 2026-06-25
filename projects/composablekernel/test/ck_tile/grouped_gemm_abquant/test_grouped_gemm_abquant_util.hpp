// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <sstream>
#include <gtest/gtest.h>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

template <typename Tuple>
class TestCkTileGroupedGemmABQuant : public ::testing::Test
{
    protected:
    using ALayout                    = std::tuple_element_t<0, Tuple>;
    using BLayout                    = std::tuple_element_t<1, Tuple>;
    using CLayout                    = std::tuple_element_t<2, Tuple>;
    using ADataType                  = std::tuple_element_t<3, Tuple>;
    using AQDataType                 = std::tuple_element_t<4, Tuple>;
    using BDataType                  = std::tuple_element_t<5, Tuple>;
    using BQDataType                 = std::tuple_element_t<6, Tuple>;
    using AccDataType                = std::tuple_element_t<7, Tuple>;
    using CDataType                  = std::tuple_element_t<8, Tuple>;
    using AQuantGroupSize            = std::tuple_element_t<9, Tuple>;
    using BQuantGroupSize            = std::tuple_element_t<10, Tuple>;
    static constexpr bool Persistent = std::tuple_element_t<11, Tuple>::value;

    using Row      = ck_tile::tensor_layout::gemm::RowMajor;
    using Col      = ck_tile::tensor_layout::gemm::ColumnMajor;
    using AQLayout = Row;
    using BQLayout = Col;

    static constexpr auto QuantMode = ck_tile::QuantType::ABQuantGrouped;

    struct GemmConfig
    {
        static constexpr bool kPadM = false;
        static constexpr bool kPadN = false;
        static constexpr bool kPadK = false;

        static constexpr int kBlockPerCu         = 1;
        static constexpr ck_tile::index_t M_Tile = 128;
        static constexpr ck_tile::index_t N_Tile = 128;
        static constexpr ck_tile::index_t K_Tile = 128 / sizeof(ADataType);

        static constexpr ck_tile::index_t M_Warp = 1;
        static constexpr ck_tile::index_t N_Warp = 4;
        static constexpr ck_tile::index_t K_Warp = 1;

        static constexpr ck_tile::index_t M_Warp_Tile = 16;
        static constexpr ck_tile::index_t N_Warp_Tile = 16;
        static constexpr ck_tile::index_t K_Warp_Tile =
            ck_tile::get_k_warp_tile<ADataType, M_Warp_Tile>();

        static constexpr bool PreshuffleB      = false;
        static constexpr bool TransposeC       = false;
        static constexpr bool DoubleSmemBuffer = false;
        static constexpr auto Scheduler        = ck_tile::GemmPipelineScheduler::Intrawave;

        static constexpr bool IsPersistent = Persistent;
    };

    using grouped_gemm_kargs = ck_tile::QuantGroupedGemmHostArgs;

    std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
    {
        return gemm_descs.size() * sizeof(ck_tile::QuantGemmTransKernelArg);
    }

    template <typename Layout>
    static constexpr inline auto is_row_major(Layout layout_)
    {
        return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                     ck_tile::tensor_layout::gemm::RowMajor>>{};
    }

    auto calculate_rtol_atol(const ck_tile::index_t K,
                             const ck_tile::index_t kbatch,
                             const float max_accumulated_value)
    {
        using ComputeType =
            std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
        const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
            ck_tile::integer_divide_ceil(K, kbatch));
        const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
            max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
        const auto rtol_split_k =
            ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
        const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
            max_accumulated_value, kbatch);
        return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
    }

    template <typename Config>
    float invoke_grouped_gemm_abquant(const std::vector<grouped_gemm_kargs>& gemm_descs,
                                      const ck_tile::stream_config& s,
                                      void* kargs_ptr)
    {
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
            ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
            ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using Traits = ck_tile::
            TileGemmTraits<Config::kPadM, Config::kPadN, Config::kPadK, ALayout, BLayout, CLayout>;
        using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<Config::kPadM,
                                                                 Config::kPadN,
                                                                 Config::kPadK,
                                                                 false,
                                                                 Config::PreshuffleB,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 QuantMode,
                                                                 AQLayout,
                                                                 BQLayout,
                                                                 Config::TransposeC,
                                                                 Config::DoubleSmemBuffer,
                                                                 Config::IsPersistent>;
        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

        const ck_tile::index_t k_grain = gemm_descs[0].k_batch * Config::K_Tile;
        const ck_tile::index_t K_split = (gemm_descs[0].K + k_grain - 1) / k_grain * Config::K_Tile;

        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        float ave_time{0};

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr auto scheduler      = Config::Scheduler;

            using QuantGemmProblem = ck_tile::GemmABQuantPipelineProblem<ADataType,
                                                                         AQDataType,
                                                                         BDataType,
                                                                         BQDataType,
                                                                         AccDataType,
                                                                         GemmShape,
                                                                         GemmUniversalTraits,
                                                                         AQuantGroupSize,
                                                                         BQuantGroupSize,
                                                                         Config::TransposeC,
                                                                         BDataType,
                                                                         scheduler,
                                                                         has_hot_loop_v,
                                                                         tail_number_v>;

            using GemmPipeline = ck_tile::ABQuantGemmPipelineAgBgCrCompV3<QuantGemmProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 Config::M_Warp,
                                                 Config::N_Warp,
                                                 Config::M_Warp_Tile,
                                                 Config::N_Warp_Tile,
                                                 Config::K_Warp_Tile,
                                                 QuantGemmProblem::TransposeC>>;

            using Kernel = ck_tile::QuantGroupedGemmKernel<TilePartitioner,
                                                           GemmPipeline,
                                                           GemmEpilogue,
                                                           GemmUniversalTraits::kQuantType>;
            auto kargs   = Kernel::MakeKargs(gemm_descs);
            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Kernel arguments not supported!");
            }

            const dim3 blocks = Kernel::BlockSize();
            const dim3 grids  = Kernel::GridSize(gemm_descs);

            HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                                kargs.data(),
                                                get_workspace_size(gemm_descs),
                                                hipMemcpyHostToDevice,
                                                s.stream_id_));

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel: " << Kernel::GetName()
                          << " with args:" << " grid: {" << grids.x << ", " << grids.y << ", "
                          << grids.z << "}" << ", blocks: {" << blocks.x << ", " << blocks.y << ", "
                          << blocks.z << "}" << std::endl;
            }

            return ave_time = ck_tile::launch_kernel(
                       s,
                       ck_tile::make_kernel<Config::kBlockPerCu>(
                           Kernel{},
                           grids,
                           blocks,
                           0,
                           ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                           gemm_descs.size()));
        };

        return ave_time = BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }

    template <typename Config>
    void invoke_grouped_gemm_persistent(const ck_tile::stream_config& s,
                                        const ck_tile::index_t num_groups,
                                        void* kargs_ptr)
    {
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
            ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
            ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<Config::kPadM,
                                                                 Config::kPadN,
                                                                 Config::kPadK,
                                                                 false,
                                                                 Config::PreshuffleB,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 QuantMode,
                                                                 AQLayout,
                                                                 BQLayout,
                                                                 Config::TransposeC,
                                                                 Config::DoubleSmemBuffer,
                                                                 Config::IsPersistent>;

        using QuantGemmProblem = ck_tile::GemmABQuantPipelineProblem<ADataType,
                                                                     AQDataType,
                                                                     BDataType,
                                                                     BQDataType,
                                                                     AccDataType,
                                                                     GemmShape,
                                                                     GemmUniversalTraits,
                                                                     AQuantGroupSize,
                                                                     BQuantGroupSize,
                                                                     Config::TransposeC>;

        using GemmPipeline = ck_tile::ABQuantGemmPipelineAgBgCrCompV3<QuantGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             Config::M_Warp,
                                             Config::N_Warp,
                                             Config::M_Warp_Tile,
                                             Config::N_Warp_Tile,
                                             Config::K_Warp_Tile,
                                             QuantGemmProblem::TransposeC>>;

        using Kernel      = ck_tile::QuantGroupedGemmKernel<TilePartitioner,
                                                            GemmPipeline,
                                                            GemmEpilogue,
                                                            GemmUniversalTraits::kQuantType>;
        const dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::MaxOccupancyGridSize(s);

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ck_tile::launch_kernel(s,
                               ck_tile::make_kernel<Config::kBlockPerCu>(
                                   Kernel{},
                                   grids,
                                   blocks,
                                   0,
                                   ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                   num_groups));
    }

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             std::vector<int>& stride_As,
             std::vector<int>& stride_Bs,
             std::vector<int>& stride_Cs,
             std::vector<int>& stride_AQs,
             std::vector<int>& stride_BQs,
             const int group_count = 8)
    {
        ck_tile::index_t AQK, BQK;

        std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
        std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
        std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;
        std::vector<ck_tile::HostTensor<AQDataType>> aq_tensors;
        std::vector<ck_tile::HostTensor<BQDataType>> bq_tensors;

        a_m_k_tensors.reserve(group_count);
        b_k_n_tensors.reserve(group_count);
        c_m_n_tensors.reserve(group_count);
        aq_tensors.reserve(group_count);
        bq_tensors.reserve(group_count);

        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> aq_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> bq_dev_buf;

        a_m_k_dev_buf.reserve(group_count);
        b_k_n_dev_buf.reserve(group_count);
        c_m_n_dev_buf.reserve(group_count);
        aq_dev_buf.reserve(group_count);
        bq_dev_buf.reserve(group_count);

        std::vector<grouped_gemm_kargs> gemm_descs;
        gemm_descs.reserve(group_count);

        for(int i = 0; i < group_count; ++i)
        {
            const ck_tile::index_t M = Ms[i];
            const ck_tile::index_t N = Ns[i];
            const ck_tile::index_t K = Ks[i];

            AQK = K / AQuantGroupSize::kK;
            BQK = K / BQuantGroupSize::kK;

            if(K % AQuantGroupSize::kK != 0)
            {
                throw std::runtime_error(
                    "K must be divisible by AQuantGroupSize::kK for ABQuantGrouped mode");
            }
            if(K % BQuantGroupSize::kK != 0)
            {
                throw std::runtime_error(
                    "K must be divisible by BQuantGroupSize::kK for ABQuantGrouped mode");
            }

            stride_As[i] = ck_tile::get_default_stride(M, K, stride_As[i], is_row_major(ALayout{}));
            stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(BLayout{}));
            stride_Cs[i] = ck_tile::get_default_stride(M, N, stride_Cs[i], is_row_major(CLayout{}));
            stride_AQs[i] =
                ck_tile::get_default_stride(M, AQK, stride_AQs[i], is_row_major(AQLayout{}));
            stride_BQs[i] =
                ck_tile::get_default_stride(BQK, N, stride_BQs[i], is_row_major(BQLayout{}));

            a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
                ck_tile::host_tensor_descriptor(M, K, stride_As[i], is_row_major(ALayout{}))));
            b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
                ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(BLayout{}))));
            c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
                ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));
            aq_tensors.push_back(ck_tile::HostTensor<AQDataType>(
                ck_tile::host_tensor_descriptor(M, AQK, stride_AQs[i], is_row_major(AQLayout{}))));
            bq_tensors.push_back(ck_tile::HostTensor<BQDataType>(
                ck_tile::host_tensor_descriptor(BQK, N, stride_BQs[i], is_row_major(BQLayout{}))));

            std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
                      << " b_k_n: " << b_k_n_tensors[i].mDesc
                      << " c_m_n: " << c_m_n_tensors[i].mDesc << " aq: " << aq_tensors[i].mDesc
                      << " bq: " << bq_tensors[i].mDesc << std::endl;

            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);
            ck_tile::FillUniformDistribution<AQDataType>{-1.f, 1.f}(aq_tensors[i]);
            ck_tile::FillUniformDistribution<BQDataType>{-1.f, 1.f}(bq_tensors[i]);

            a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                a_m_k_tensors[i].get_element_space_size_in_bytes()));
            b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                b_k_n_tensors[i].get_element_space_size_in_bytes()));
            c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                c_m_n_tensors[i].get_element_space_size_in_bytes()));
            aq_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                aq_tensors[i].get_element_space_size_in_bytes()));
            bq_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                bq_tensors[i].get_element_space_size_in_bytes()));

            a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
            b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
            aq_dev_buf[i]->ToDevice(aq_tensors[i].data());
            bq_dev_buf[i]->ToDevice(bq_tensors[i].data());
            c_m_n_dev_buf[i]->SetZero();
            c_m_n_tensors[i].SetZero();

            const void* p_a  = a_m_k_dev_buf[i]->GetDeviceBuffer();
            const void* p_b  = b_k_n_dev_buf[i]->GetDeviceBuffer();
            void* p_c        = c_m_n_dev_buf[i]->GetDeviceBuffer();
            const void* p_aq = aq_dev_buf[i]->GetDeviceBuffer();
            const void* p_bq = bq_dev_buf[i]->GetDeviceBuffer();

            gemm_descs.push_back({p_a,
                                  p_b,
                                  p_c,
                                  p_aq,
                                  p_bq,
                                  1, // k_batch
                                  M,
                                  N,
                                  K,
                                  AQK,
                                  BQK,
                                  stride_As[i],
                                  stride_Bs[i],
                                  stride_Cs[i],
                                  stride_AQs[i],
                                  stride_BQs[i]});
        }

        ck_tile::DeviceMem gemm_workspace;
        gemm_workspace.Realloc(get_workspace_size(gemm_descs));
        void* kargs_ptr = gemm_workspace.GetDeviceBuffer();

        if constexpr(Persistent)
        {
            std::vector<ck_tile::QuantGemmTransKernelArg> kargs;
            for(const auto& arg : gemm_descs)
            {
                kargs.emplace_back(ck_tile::QuantGroupedGemmKernelArgs{arg.a_ptr,
                                                                       arg.b_ptr,
                                                                       arg.aq_ptr,
                                                                       arg.bq_ptr,
                                                                       arg.e_ptr,
                                                                       arg.M,
                                                                       arg.N,
                                                                       arg.K,
                                                                       arg.QK_A,
                                                                       arg.QK_B,
                                                                       arg.stride_A,
                                                                       arg.stride_B,
                                                                       arg.stride_E,
                                                                       arg.stride_AQ,
                                                                       arg.stride_BQ,
                                                                       arg.k_batch});
            }
            const auto stream = ck_tile::stream_config{nullptr, false, 1};
            ck_tile::hip_check_error(
                hipMemcpyWithStream(kargs_ptr,
                                    kargs.data(),
                                    kargs.size() * sizeof(ck_tile::QuantGemmTransKernelArg),
                                    hipMemcpyHostToDevice,
                                    stream.stream_id_));
            invoke_grouped_gemm_persistent<GemmConfig>(stream, group_count, kargs_ptr);
        }
        else
        {
            const auto stream = ck_tile::stream_config{nullptr, false, 1};
            invoke_grouped_gemm_abquant<GemmConfig>(gemm_descs, stream, kargs_ptr);
        }

        // Copy results back to host for validation
        for(int i = 0; i < group_count; i++)
        {
            c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
        }

        bool pass{true};
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::HostTensor<CDataType> c_m_n_host_ref(ck_tile::host_tensor_descriptor(
                Ms[i], Ns[i], stride_Cs[i], is_row_major(CLayout{})));
            c_m_n_host_ref.SetZero();

            ck_tile::reference_gemm_abquant<ADataType,
                                            AQDataType,
                                            BDataType,
                                            BQDataType,
                                            AccDataType,
                                            CDataType,
                                            AQuantGroupSize,
                                            BQuantGroupSize>(
                a_m_k_tensors[i], aq_tensors[i], b_k_n_tensors[i], bq_tensors[i], c_m_n_host_ref);

            const float max_accumulated_value =
                *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
            const auto rtol_atol = calculate_rtol_atol(Ks[i], 1, max_accumulated_value);
            pass &=
                ck_tile::check_err(c_m_n_tensors[i],
                                   c_m_n_host_ref,
                                   "Error: Incorrect results! in group [" + std::to_string(i) + "]",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));
            std::cout << "gemm[" << i
                      << "] Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;

        EXPECT_TRUE(pass);
    }
};

// Aliases for split test files
template <typename Tuple>
using TestCkTileGroupedGemmABQuant_1x1x128 = TestCkTileGroupedGemmABQuant<Tuple>;

template <typename Tuple>
using TestCkTileGroupedGemmABQuant_1x128x128 = TestCkTileGroupedGemmABQuant<Tuple>;
