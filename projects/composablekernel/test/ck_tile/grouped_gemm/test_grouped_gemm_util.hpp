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
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

template <typename Tuple>
class TestCkTileGroupedGemm : public ::testing::Test
{
    protected:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = std::tuple_element_t<2, Tuple>;
    using ADataType   = std::tuple_element_t<3, Tuple>;
    using BDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = std::tuple_element_t<5, Tuple>;
    using CDataType   = std::tuple_element_t<6, Tuple>;
    using DsLayout    = ck_tile::tuple<>;
    using DsDataType  = ck_tile::tuple<>;

    // Get the persistent value from ck_tile::bool_constant
    using PersistentType             = std::tuple_element_t<7, Tuple>;
    static constexpr bool Persistent = PersistentType::value;

    struct GroupedGemKernelParam_Mfma
    {
        static const bool kPadM = false;
        static const bool kPadN = false;
        static const bool kPadK = false;

        static const int kBlockPerCu         = 1;
        static const ck_tile::index_t M_Tile = 256;
        static const ck_tile::index_t N_Tile = 256;
        static const ck_tile::index_t K_Tile = 64;

        static const ck_tile::index_t M_Warp = 2;
        static const ck_tile::index_t N_Warp = 2;
        static const ck_tile::index_t K_Warp = 1;

        static const ck_tile::index_t M_Warp_Tile = 32;
        static const ck_tile::index_t N_Warp_Tile = 32;
        static const ck_tile::index_t K_Warp_Tile = 16;
    };

    struct GroupedGemKernelParam_Wmma : public GroupedGemKernelParam_Mfma
    {
        static const ck_tile::index_t M_Tile = 128;
        static const ck_tile::index_t N_Tile = 128;
        static const ck_tile::index_t K_Tile = 64;

        static const ck_tile::index_t M_Warp_Tile = 16;
        static const ck_tile::index_t N_Warp_Tile = 16;
        static const ck_tile::index_t K_Warp_Tile = 16;
    };

    using grouped_gemm_kargs = ck_tile::GroupedGemmHostArgs<>;
    std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
    {
        return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArg<>);
    }

    template <typename GroupedGemKernelParam, typename ALayout, typename BLayout, typename CLayout>
    void invoke_grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                             const ck_tile::stream_config& s,
                             void* kargs_ptr)
    {
        constexpr bool DoubleSmemBuffer = false;
        constexpr bool TransposeC       = false;

        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<GroupedGemKernelParam::M_Tile,
                                                     GroupedGemKernelParam::N_Tile,
                                                     GroupedGemKernelParam::K_Tile>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp,
                                                     GroupedGemKernelParam::N_Warp,
                                                     GroupedGemKernelParam::K_Warp>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp_Tile,
                                                     GroupedGemKernelParam::N_Warp_Tile,
                                                     GroupedGemKernelParam::K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GroupedGemKernelParam::kPadM,
                                                                     GroupedGemKernelParam::kPadN,
                                                                     GroupedGemKernelParam::kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC>;

        constexpr auto scheduler   = ck_tile::GemmPipelineScheduler::Intrawave;
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

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
                                             GroupedGemKernelParam::M_Warp,
                                             GroupedGemKernelParam::N_Warp,
                                             GroupedGemKernelParam::M_Warp_Tile,
                                             GroupedGemKernelParam::N_Warp_Tile,
                                             GroupedGemKernelParam::K_Warp_Tile,
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

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ck_tile::ignore =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GroupedGemKernelParam::kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       gemm_descs.size()));
    }

    template <typename GroupedGemKernelParam, typename ALayout, typename BLayout, typename CLayout>
    void invoke_grouped_gemm_persistent(const ck_tile::stream_config& s,
                                        const ck_tile::index_t num_groups,
                                        void* kargs_ptr)
    {
        constexpr bool TransposeC       = false;
        constexpr bool DoubleSmemBuffer = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<GroupedGemKernelParam::M_Tile,
                                                     GroupedGemKernelParam::N_Tile,
                                                     GroupedGemKernelParam::K_Tile>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp,
                                                     GroupedGemKernelParam::N_Warp,
                                                     GroupedGemKernelParam::K_Warp>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp_Tile,
                                                     GroupedGemKernelParam::N_Warp_Tile,
                                                     GroupedGemKernelParam::K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits =
            ck_tile::PersistentTileGemmUniversalTraits<GroupedGemKernelParam::kPadM,
                                                       GroupedGemKernelParam::kPadN,
                                                       GroupedGemKernelParam::kPadK,
                                                       DoubleSmemBuffer,
                                                       ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       TransposeC>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        // We create the GEMM pipeline without specifying hotloop or tailnumber.
        // These are automatically run inside the kernel based on the given input data.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;
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
                                             GroupedGemKernelParam::M_Warp,
                                             GroupedGemKernelParam::N_Warp,
                                             GroupedGemKernelParam::M_Warp_Tile,
                                             GroupedGemKernelParam::N_Warp_Tile,
                                             GroupedGemKernelParam::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC>>;
        using Kernel      = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        const dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::MaxOccupancyGridSize(s);

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ck_tile::ignore =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       num_groups));
    }

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

            stride_As[i] = f_get_default_stride(M, N, stride_As[i], ALayout{});
            stride_Bs[i] = f_get_default_stride(K, N, stride_Bs[i], BLayout{});
            stride_Cs[i] = f_get_default_stride(M, N, stride_Cs[i], CLayout{});

            a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
                f_host_tensor_descriptor(M, K, stride_As[i], ALayout{})));
            b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
                f_host_tensor_descriptor(K, N, stride_Bs[i], BLayout{})));
            c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
                f_host_tensor_descriptor(M, N, stride_Cs[i], CLayout{})));

            std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
                      << " b_k_n: " << b_k_n_tensors[i].mDesc
                      << " c_m_n: " << c_m_n_tensors[i].mDesc << " KBatch: " << kbatch << std::endl;

            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);

            a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                a_m_k_tensors[i].get_element_space_size_in_bytes()));
            b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                b_k_n_tensors[i].get_element_space_size_in_bytes()));
            c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                c_m_n_tensors[i].get_element_space_size_in_bytes()));

            a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
            b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
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
            // Generate kernel arguments
            std::vector<ck_tile::GemmTransKernelArg<>> kargs;
            void* kargs_ptr = gemm_workspace.GetDeviceBuffer();
            for(const auto& arg : gemm_descs)
            {
                kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{{arg.a_ptr},
                                                                      {arg.b_ptr},
                                                                      {/*arg.ds_ptr*/},
                                                                      arg.e_ptr,
                                                                      arg.M,
                                                                      arg.N,
                                                                      arg.K,
                                                                      {arg.stride_A},
                                                                      {arg.stride_B},
                                                                      {/*arg.stride_Ds*/},
                                                                      arg.stride_E,
                                                                      arg.k_batch});
            }
            const auto stream = ck_tile::stream_config{nullptr, false, 1};
            ck_tile::hip_check_error(
                hipMemcpyWithStream(kargs_ptr,
                                    kargs.data(),
                                    kargs.size() * sizeof(ck_tile::GemmTransKernelArg<>),
                                    hipMemcpyHostToDevice,
                                    stream.stream_id_));
#if CK_TILE_USE_WMMA
            invoke_grouped_gemm_persistent<GroupedGemKernelParam_Wmma, ALayout, BLayout, CLayout>(
                stream, group_count, kargs_ptr);
#else
            invoke_grouped_gemm_persistent<GroupedGemKernelParam_Mfma, ALayout, BLayout, CLayout>(
                stream, group_count, kargs_ptr);
#endif
        }
        else
        {
#if CK_TILE_USE_WMMA
            invoke_grouped_gemm<GroupedGemKernelParam_Wmma, ALayout, BLayout, CLayout>(
                gemm_descs,
                ck_tile::stream_config{nullptr, false, 1},
                gemm_workspace.GetDeviceBuffer());
#else
            invoke_grouped_gemm<GroupedGemKernelParam_Mfma, ALayout, BLayout, CLayout>(
                gemm_descs,
                ck_tile::stream_config{nullptr, false, 1},
                gemm_workspace.GetDeviceBuffer());
#endif
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
            const auto rtol_atol = calculate_rtol_atol(Ks[i], kbatch, max_accumulated_value);
            pass &= ck_tile::check_err(c_m_n_tensors[i],
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));
        }
        EXPECT_TRUE(pass);
    }
};
