// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

template <typename Tuple>
class TestCkTileGroupedGemmPreshuffle : public ::testing::Test
{
    protected:
    using ALayout     = typename Tuple::ALayoutType;
    using BLayout     = typename Tuple::BLayoutType;
    using CLayout     = typename Tuple::CLayoutType;
    using ADataType   = typename Tuple::ADataType;
    using BDataType   = typename Tuple::BDataType;
    using AccDataType = typename Tuple::AccDataType;
    using CDataType   = typename Tuple::CDataType;

    using DsLayout   = ck_tile::tuple<>; // not used
    using DsDataType = ck_tile::tuple<>; // not used

    // Get the persistent value from ck_tile::bool_constant
    using PersistentType             = typename Tuple::Persistent;
    static constexpr bool Persistent = PersistentType::value;

    static const bool kPadM = false;
    static const bool kPadN = false;
    static const bool kPadK = true; // preshuffle pipeline requires k padding

    static const int kBlockPerCu = Tuple::BlockPerCu_;

    // Tile dimensions from tuple
    static const ck_tile::index_t M_Tile = Tuple::M_Tile_;
    static const ck_tile::index_t N_Tile = Tuple::N_Tile_;
    static const ck_tile::index_t K_Tile = Tuple::K_Tile_;

    static const ck_tile::index_t M_Warp = 1;
    static const ck_tile::index_t N_Warp = 4;
    static const ck_tile::index_t K_Warp = 1;

    static const ck_tile::index_t M_Warp_Tile = 16;
    static const ck_tile::index_t N_Warp_Tile = 16;
    static const ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<BDataType, M_Warp_Tile, true>();

    static constexpr bool DoubleSmemBuffer = true;  // preshuffle v2 uses ping-pong smem
    static constexpr bool TransposeC       = false; // transpose c is not supported
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;

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

    using grouped_gemm_kargs = ck_tile::GroupedGemmHostArgs<>;
    inline std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
    {
        return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArg<>);
    }

    template <typename ALayout, typename BLayout, typename CLayout>
    void invoke_grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                             const ck_tile::stream_config& s,
                             void* kargs_ptr)
    {
        constexpr ck_tile::index_t WaveSize     = 32;
        constexpr ck_tile::index_t MIterPerWarp = M_Tile / (M_Warp * M_Warp_Tile);
        constexpr bool SupportVectorSize16 =
            (M_Warp_Tile * K_Warp_Tile * sizeof(ADataType) * MIterPerWarp / WaveSize) % 16 == 0;
        constexpr int VectorSize = SupportVectorSize16 ? 16 : 8;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        // for testing purposes, we can hardcode the values here as we what is compatible with
        // pipeline
        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<kPadM,
                                             kPadN,
                                             kPadK,
                                             DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             CLayout,
                                             TransposeC,
                                             /*UseStructuredSparsity*/ false,
                                             /*Persistent*/ false,
                                             /*NumWaveGroups*/ 1,
                                             /*Preshuffle*/ true,
                                             VectorSize>;

        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  ck_tile::GemmPipelineScheduler::Default>;
        using GemmPipeline =
            ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2<UniversalGemmProblem>;

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
                                             UniversalGemmProblem::TransposeC>>;
        using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKargs(gemm_descs);
        EXPECT_TRUE(Kernel::IsSupportedArgument(kargs));
        const dim3 grids  = Kernel::GridSize(gemm_descs);
        const dim3 blocks = Kernel::BlockSize();

        ck_tile::hip_check_error(hipMemcpyWithStream(kargs_ptr,
                                                     kargs.data(),
                                                     get_workspace_size(gemm_descs),
                                                     hipMemcpyHostToDevice,
                                                     s.stream_id_));

        ck_tile::ignore =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       gemm_descs.size()));
    }

    private:
    template <typename ALayout, typename BLayout, typename CLayout>
    void invoke_grouped_gemm_persistent(const std::vector<grouped_gemm_kargs>& gemm_descs,
                                        const ck_tile::stream_config& s,
                                        void* kargs_ptr)
    {
        constexpr ck_tile::index_t WaveSize     = 32;
        constexpr ck_tile::index_t MIterPerWarp = M_Tile / (M_Warp * M_Warp_Tile);
        constexpr bool SupportVectorSize16 =
            (M_Warp_Tile * K_Warp_Tile * sizeof(ADataType) * MIterPerWarp / WaveSize) % 16 == 0;
        constexpr int VectorSize = SupportVectorSize16 ? 16 : 8;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        // Enable persistent mode for preshuffle
        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits</*kPadM*/ true,
                                             /*kPadN*/ true,
                                             /*kPadK*/ true,
                                             DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             CLayout,
                                             TransposeC,
                                             /*UseStructuredSparsity*/ false,
                                             /*Persistent*/ true, // Enable persistent mode
                                             /*NumWaveGroups*/ 1,
                                             /*Preshuffle*/ true,
                                             VectorSize>;

        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  ck_tile::GemmPipelineScheduler::Default>;
        using GemmPipeline =
            ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2<UniversalGemmProblem>;

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
                                             UniversalGemmProblem::TransposeC>>;
        using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKargs(gemm_descs);
        EXPECT_TRUE(Kernel::IsSupportedArgument(kargs));
        const dim3 grids  = Kernel::GridSize(gemm_descs);
        const dim3 blocks = Kernel::BlockSize();

        ck_tile::hip_check_error(hipMemcpyWithStream(kargs_ptr,
                                                     kargs.data(),
                                                     get_workspace_size(gemm_descs),
                                                     hipMemcpyHostToDevice,
                                                     s.stream_id_));

        ck_tile::ignore =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       gemm_descs.size()));
    }

    struct BShuffleGemmConfig
    {
        static constexpr ck_tile::index_t N_Warp_Tile =
            TestCkTileGroupedGemmPreshuffle::N_Warp_Tile;
        static constexpr ck_tile::index_t K_Warp_Tile =
            TestCkTileGroupedGemmPreshuffle::K_Warp_Tile;
    };

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             std::vector<int>& stride_As,
             std::vector<int>& stride_Bs,
             std::vector<int>& stride_Cs,
             const int kbatch      = 1,
             const int group_count = 16)
    {

        using namespace ck_tile::literals;
        auto f_host_tensor_descriptor = [](std::size_t row,
                                           std::size_t col,
                                           std::size_t stride,
                                           auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

        auto f_get_default_stride =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(stride == 0)
                {
                    if constexpr(std::is_same_v<decltype(layout),
                                                ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        return col;
                    }
                    else
                    {
                        return row;
                    }
                }
                else
                    return stride;
            };

        std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
        std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
        std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

        a_m_k_tensors.reserve(group_count);
        b_k_n_tensors.reserve(group_count);
        c_m_n_tensors.reserve(group_count);

        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

        a_m_k_dev_buf.reserve(group_count);
        b_k_n_dev_buf.reserve(group_count);
        c_m_n_dev_buf.reserve(group_count);

        std::vector<grouped_gemm_kargs> gemm_descs;
        gemm_descs.reserve(group_count);

        for(int i = 0; i < group_count; ++i)
        {
            const ck_tile::index_t M = Ms[i];
            const ck_tile::index_t N = Ns[i];
            const ck_tile::index_t K = Ks[i];

            stride_As[i] = f_get_default_stride(M, K, stride_As[i], ALayout{});
            stride_Bs[i] = f_get_default_stride(K, N, stride_Bs[i], BLayout{});
            stride_Cs[i] = f_get_default_stride(M, N, stride_Cs[i], CLayout{});

            a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
                f_host_tensor_descriptor(M, K, stride_As[i], ALayout{})));
            b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
                f_host_tensor_descriptor(K, N, stride_Bs[i], BLayout{})));
            c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
                f_host_tensor_descriptor(M, N, stride_Cs[i], CLayout{})));

            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);

            // Host-side preshuffle of B
            auto b_shuffle_host = ck_tile::shuffle_b<BShuffleGemmConfig>(b_k_n_tensors[i]);

            a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                a_m_k_tensors[i].get_element_space_size_in_bytes()));
            b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                b_shuffle_host.get_element_space_size_in_bytes()));
            c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                c_m_n_tensors[i].get_element_space_size_in_bytes()));

            a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
            b_k_n_dev_buf[i]->ToDevice(b_shuffle_host.data());
            c_m_n_dev_buf[i]->SetZero();
            c_m_n_tensors[i].SetZero();

            const void* p_a = a_m_k_dev_buf[i]->GetDeviceBuffer();
            const void* p_b = b_k_n_dev_buf[i]->GetDeviceBuffer();
            void* p_c       = c_m_n_dev_buf[i]->GetDeviceBuffer();

            gemm_descs.push_back({p_a,
                                  p_b,
                                  {/*ds_ptr*/},
                                  p_c,
                                  kbatch,
                                  M,
                                  N,
                                  K,
                                  stride_As[i],
                                  stride_Bs[i],
                                  {/*stride_Ds*/},
                                  stride_Cs[i]});
        }

        ck_tile::DeviceMem gemm_workspace;
        gemm_workspace.Realloc(get_workspace_size(gemm_descs));

        if constexpr(Persistent)
        {
            invoke_grouped_gemm_persistent<ALayout, BLayout, CLayout>(
                gemm_descs,
                ck_tile::stream_config{nullptr, false, 1},
                gemm_workspace.GetDeviceBuffer());
        }
        else
        {
            invoke_grouped_gemm<ALayout, BLayout, CLayout>(
                gemm_descs,
                ck_tile::stream_config{nullptr, false, 1},
                gemm_workspace.GetDeviceBuffer());
        }

        // Copy results back to host for validation
        for(int i = 0; i < group_count; i++)
        {
            c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
        }

        bool pass{true};
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::HostTensor<CDataType> c_m_n_host_ref(
                f_host_tensor_descriptor(Ms[i], Ns[i], stride_Cs[i], CLayout{}));
            c_m_n_host_ref.SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
            const float max_accumulated_value =
                *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
            const auto rtol_atol =
                calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                    Ks[i], kbatch, max_accumulated_value);
            pass &= ck_tile::check_err(c_m_n_tensors[i],
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));
        }
        EXPECT_TRUE(pass);
    }
};
