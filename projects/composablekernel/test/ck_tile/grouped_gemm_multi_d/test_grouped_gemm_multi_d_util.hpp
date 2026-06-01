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

enum class PipelineType
{
    Memory = 0,
    CompV3 = 1,
    CompV4 = 2
};

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

struct MultiplyMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const D0& d0, const D1& d1) const -> void
    {
        const float x0_f = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
                           ck_tile::type_convert<float>(d1);

        e = ck_tile::type_convert<E>(x0_f);
    }
};

template <typename Config>
class TestCkTileGroupedGemmMultiD : public ::testing::Test
{
    protected:
    using ALayout     = typename Config::ALayoutType;
    using BLayout     = typename Config::BLayoutType;
    using ELayout     = typename Config::ELayoutType;
    using DsLayout    = typename Config::DsLayoutType;
    using ADataType   = typename Config::ADataType;
    using BDataType   = typename Config::BDataType;
    using AccDataType = typename Config::AccDataType;
    using EDataType   = typename Config::EDataType;
    using PrecType    = BDataType;
    using DsDataType  = typename Config::DsDataType;
    using D0DataType  = std::tuple_element_t<0, DsDataType>;
    using D1DataType  = std::tuple_element_t<1, DsDataType>;
    using D0Layout    = std::tuple_element_t<0, DsLayout>;
    using D1Layout    = std::tuple_element_t<1, DsLayout>;

    static const bool kPadM = false;
    static const bool kPadN = false;
    static const bool kPadK = false;

    static constexpr bool TransposeC = false; // transpose c is not supported
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;

    auto calculate_rtol_atol(const ck_tile::index_t K,
                             const ck_tile::index_t kbatch,
                             const float max_accumulated_value)
    {
        using ComputeTypeAB =
            std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

        using ComputeType = std::
            conditional_t<sizeof(ComputeTypeAB) < sizeof(D0DataType), ComputeTypeAB, D0DataType>;
        // Calculate thresholds
        const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType, AccDataType>(
            ck_tile::integer_divide_ceil(K, kbatch));

        const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType, AccDataType>(
            max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

        // Calculate error due to split_k accumulation
        const auto rtol_split_k =
            ck_tile::get_relative_threshold<EDataType, EDataType, EDataType>(kbatch);

        const auto atol_split_k = ck_tile::get_absolute_threshold<EDataType, EDataType, EDataType>(
            max_accumulated_value, kbatch);

        // Use higher threshold
        return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
    }

    using grouped_gemm_kargs = ck_tile::GroupedGemmHostArgs<DsDataType::size()>;
    inline std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
    {
        return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArg<DsDataType::size()>);
    }

    void invoke_grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                             const ck_tile::stream_config& s,
                             void* kargs_ptr)
    {
        EXPECT_TRUE(gemm_descs[0].k_batch == 1);
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<Config::M_Tile_, Config::N_Tile_, Config::K_Tile_>,
            ck_tile::sequence<Config::M_Warp_, Config::N_Warp_, Config::K_Warp_>,
            ck_tile::sequence<Config::M_Warp_Tile_, Config::N_Warp_Tile_, Config::K_Warp_Tile_>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        // for testing purposes, we can hardcode the values here as we what is compatible with
        // pipeline
        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<kPadM,
                                             kPadN,
                                             kPadK,
                                             Config::DoubleSmemBuffer_,
                                             ALayout,
                                             BLayout,
                                             ELayout,
                                             TransposeC,
                                             /*UseStructuredSparsity*/ false,
                                             /*Persistent*/ false,
                                             /*NumWaveGroups*/ 1,
                                             /*Preshuffle*/ false>;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           Config::Scheduler_>;

        using GemmPipeline = std::conditional_t<
            Config::Pipeline_ == (PipelineType::Memory),
            ck_tile::GemmPipelineAgBgCrMem<UniversalGemmProblem>,
            std::conditional_t<Config::Pipeline_ == (PipelineType::CompV3),
                               ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>,
                               ck_tile::GemmPipelineAgBgCrCompV4<UniversalGemmProblem>>>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             DsLayout,
                                             ELayout,
                                             MultiplyMultiply,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             Config::M_Warp_,
                                             Config::N_Warp_,
                                             Config::M_Warp_Tile_,
                                             Config::N_Warp_Tile_,
                                             Config::K_Warp_Tile_,
                                             UniversalGemmProblem::TransposeC>>;
        using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKargs(gemm_descs);
        EXPECT_TRUE(Kernel::IsSupportedArgument(kargs));
        const dim3 grids  = Kernel::GridSize(gemm_descs);
        const dim3 blocks = Kernel::BlockSize();

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ck_tile::hip_check_error(hipMemcpyWithStream(kargs_ptr,
                                                     kargs.data(),
                                                     get_workspace_size(gemm_descs),
                                                     hipMemcpyHostToDevice,
                                                     s.stream_id_));

        ck_tile::ignore =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<Config::BlockPerCu_>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       gemm_descs.size()));
    }

    void invoke_grouped_gemm_persistent(const ck_tile::stream_config& s,
                                        const ck_tile::index_t num_groups,
                                        void* kargs_ptr)
    {
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<Config::M_Tile_, Config::N_Tile_, Config::K_Tile_>,
            ck_tile::sequence<Config::M_Warp_, Config::N_Warp_, Config::K_Warp_>,
            ck_tile::sequence<Config::M_Warp_Tile_, Config::N_Warp_Tile_, Config::K_Warp_Tile_>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;
        using GemmUniversalTraits =
            ck_tile::PersistentTileGemmUniversalTraits<kPadM,
                                                       kPadN,
                                                       kPadK,
                                                       Config::DoubleSmemBuffer_,
                                                       ALayout,
                                                       BLayout,
                                                       ELayout>;

        // We create the GEMM pipeline without specifying hotloop or tailnumber.
        // These are automatically run inside the kernel based on the given input data.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           Config::Scheduler_>;

        using GemmPipeline = std::conditional_t<
            Config::Pipeline_ == (PipelineType::Memory),
            ck_tile::GemmPipelineAgBgCrMem<UniversalGemmProblem>,
            std::conditional_t<Config::Pipeline_ == (PipelineType::CompV3),
                               ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>,
                               ck_tile::GemmPipelineAgBgCrCompV4<UniversalGemmProblem>>>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             DsLayout,
                                             ELayout,
                                             MultiplyMultiply,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             Config::M_Warp_,
                                             Config::N_Warp_,
                                             Config::M_Warp_Tile_,
                                             Config::N_Warp_Tile_,
                                             Config::K_Warp_Tile_,
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
                                   ck_tile::make_kernel<Config::BlockPerCu_>(
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
             std::vector<int>& stride_Es,
             std::vector<int>& stride_D0,
             std::vector<int>& stride_D1,
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
        std::vector<ck_tile::HostTensor<EDataType>> e_m_n_tensors;
        std::vector<ck_tile::HostTensor<D0DataType>> d0_m_n_tensors;
        std::vector<ck_tile::HostTensor<D1DataType>> d1_m_n_tensors;

        a_m_k_tensors.reserve(group_count);
        b_k_n_tensors.reserve(group_count);
        e_m_n_tensors.reserve(group_count);
        d0_m_n_tensors.reserve(group_count);
        d1_m_n_tensors.reserve(group_count);

        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> e_m_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> d0_m_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> d1_m_n_dev_buf;

        a_m_k_dev_buf.reserve(group_count);
        b_k_n_dev_buf.reserve(group_count);
        e_m_n_dev_buf.reserve(group_count);
        d0_m_n_dev_buf.reserve(group_count);
        d1_m_n_dev_buf.reserve(group_count);

        std::vector<grouped_gemm_kargs> gemm_descs;
        gemm_descs.reserve(group_count);

        for(int i = 0; i < group_count; ++i)
        {
            const ck_tile::index_t M = Ms[i];
            const ck_tile::index_t N = Ns[i];
            const ck_tile::index_t K = Ks[i];

            stride_As[i] = f_get_default_stride(M, K, stride_As[i], ALayout{});
            stride_Bs[i] = f_get_default_stride(K, N, stride_Bs[i], BLayout{});
            stride_Es[i] = f_get_default_stride(M, N, stride_Es[i], ELayout{});
            stride_D0[i] = f_get_default_stride(M, N, stride_D0[i], D0Layout{});
            stride_D1[i] = f_get_default_stride(M, N, stride_D1[i], D1Layout{});

            a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
                f_host_tensor_descriptor(M, K, stride_As[i], ALayout{})));
            b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
                f_host_tensor_descriptor(K, N, stride_Bs[i], BLayout{})));
            e_m_n_tensors.push_back(ck_tile::HostTensor<EDataType>(
                f_host_tensor_descriptor(M, N, stride_Es[i], ELayout{})));
            d0_m_n_tensors.push_back(ck_tile::HostTensor<D0DataType>(
                f_host_tensor_descriptor(M, N, stride_D0[i], D0Layout{})));
            d1_m_n_tensors.push_back(ck_tile::HostTensor<D1DataType>(
                f_host_tensor_descriptor(M, N, stride_D1[i], D1Layout{})));

            std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
                      << " b_k_n: " << b_k_n_tensors[i].mDesc
                      << " e_m_n: " << e_m_n_tensors[i].mDesc
                      << " d0_m_n: " << d0_m_n_tensors[i].mDesc
                      << " d1_m_n: " << d1_m_n_tensors[i].mDesc << std::endl;

            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);
            ck_tile::FillUniformDistribution<D0DataType>{-2.f, 2.f}(d0_m_n_tensors[i]);
            ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n_tensors[i]);

            a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                a_m_k_tensors[i].get_element_space_size_in_bytes()));
            b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                b_k_n_tensors[i].get_element_space_size_in_bytes()));
            e_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                e_m_n_tensors[i].get_element_space_size_in_bytes()));
            d0_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                d0_m_n_tensors[i].get_element_space_size_in_bytes()));
            d1_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                d1_m_n_tensors[i].get_element_space_size_in_bytes()));

            a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
            b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
            e_m_n_dev_buf[i]->SetZero();
            d0_m_n_dev_buf[i]->ToDevice(d0_m_n_tensors[i].data());
            d1_m_n_dev_buf[i]->ToDevice(d1_m_n_tensors[i].data());

            const void* p_a = a_m_k_dev_buf[i]->GetDeviceBuffer();
            const void* p_b = b_k_n_dev_buf[i]->GetDeviceBuffer();
            void* p_e       = e_m_n_dev_buf[i]->GetDeviceBuffer();

            std::array<const void*, DsDataType::size()> ds_ptr_buf = {
                d0_m_n_dev_buf[i]->GetDeviceBuffer(), d1_m_n_dev_buf[i]->GetDeviceBuffer()};
            std::array<ck_tile::index_t, DsDataType::size()> stridesDs = {stride_D0[i],
                                                                          stride_D1[i]};

            gemm_descs.push_back({p_a,
                                  p_b,
                                  ds_ptr_buf,
                                  p_e,
                                  kbatch,
                                  M,
                                  N,
                                  K,
                                  stride_As[i],
                                  stride_Bs[i],
                                  stridesDs,
                                  stride_Es[i]});
        }

        ck_tile::DeviceMem gemm_workspace;
        gemm_workspace.Realloc(get_workspace_size(gemm_descs));

        if constexpr(Config::Persistent_)
        {
            std::vector<ck_tile::GemmTransKernelArg<DsDataType::size()>> kargs;
            void* kargs_ptr = gemm_workspace.GetDeviceBuffer();
            for(const auto& arg : gemm_descs)
            {
                kargs.emplace_back(
                    ck_tile::UniversalGemmKernelArgs<1, 1, DsDataType::size()>{{arg.a_ptr},
                                                                               {arg.b_ptr},
                                                                               arg.ds_ptr,
                                                                               arg.e_ptr,
                                                                               arg.M,
                                                                               arg.N,
                                                                               arg.K,
                                                                               {arg.stride_A},
                                                                               {arg.stride_B},
                                                                               arg.stride_Ds,
                                                                               arg.stride_E,
                                                                               arg.k_batch});
            }
            const auto stream = ck_tile::stream_config{nullptr, false, 1};
            ck_tile::hip_check_error(hipMemcpyWithStream(
                kargs_ptr,
                kargs.data(),
                kargs.size() * sizeof(ck_tile::GemmTransKernelArg<DsDataType::size()>),
                hipMemcpyHostToDevice,
                stream.stream_id_));

            invoke_grouped_gemm_persistent(stream, group_count, kargs_ptr);
        }
        else
        {
            invoke_grouped_gemm(gemm_descs,
                                ck_tile::stream_config{nullptr, false, 1},
                                gemm_workspace.GetDeviceBuffer());
        }

        // Copy results back to host for validation
        for(int i = 0; i < group_count; i++)
        {
            e_m_n_dev_buf[i]->FromDevice(e_m_n_tensors[i].data());
        }

        std::vector<ck_tile::HostTensor<EDataType>> e_m_n_host_refs;
        e_m_n_host_refs.reserve(group_count);

        bool pass{true};
        for(int i = 0; i < group_count; ++i)
        {
            e_m_n_host_refs.push_back(ck_tile::HostTensor<EDataType>(
                f_host_tensor_descriptor(Ms[i], Ns[i], stride_Es[i], ELayout{})));

            e_m_n_host_refs[i].SetZero();

            ck_tile::reference_gemm_multiple_d<ADataType,
                                               BDataType,
                                               DsDataType,
                                               AccDataType,
                                               EDataType,
                                               MultiplyMultiply>(
                a_m_k_tensors[i],
                b_k_n_tensors[i],
                {d0_m_n_tensors[i], d1_m_n_tensors[i]},
                e_m_n_host_refs[i]);
            const float max_accumulated_value =
                *std::max_element(e_m_n_host_refs[i].mData.begin(), e_m_n_host_refs[i].mData.end());

            const auto rtol_atol = calculate_rtol_atol(Ks[i], 1, max_accumulated_value);

            pass &=
                ck_tile::check_err(e_m_n_tensors[i],
                                   e_m_n_host_refs[i],
                                   "Error: Incorrect results! in group [" + std::to_string(i) + "]",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

            std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        EXPECT_TRUE(pass);
    }
};
