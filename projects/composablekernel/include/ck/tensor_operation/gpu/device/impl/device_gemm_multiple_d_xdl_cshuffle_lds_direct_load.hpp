// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle_lds_direct_load.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferScalarPerVector,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferScalarPerVector,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          PipelineVersion PipelineVer = PipelineVersion::v4,
          typename ComputeDataType    = EDataType>
struct DeviceGemmMultipleD_Xdl_CShuffle_LdsDirectLoad
    : public DeviceGemmMultipleD<ALayout,
                                 BLayout,
                                 DsLayout,
                                 ELayout,
                                 ADataType,
                                 BDataType,
                                 DsDataType,
                                 EDataType,
                                 AElementwiseOperation,
                                 BElementwiseOperation,
                                 CDEElementwiseOperation>
{
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();
    static constexpr auto I1            = Number<1>{};
    static constexpr index_t NumDTensor = DsDataType::Size();

    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemmMultipleD_Xdl_CShuffle_LdsDirectLoad<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        ADataType,
        BDataType,
        ComputeDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        GemmSpec,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave_,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferScalarPerVector,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferScalarPerVector,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched,
        PipelineVer>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    using Argument = typename GridwiseGemm64::Argument;

    struct Invoker : public BaseInvoker
    {

        template <typename GridwiseGemm>
        float RunImp(const typename GridwiseGemm::Argument& arg,
                     const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                            arg.b_grid_desc_n_k_,
                                            arg.ds_grid_desc_m_n_,
                                            arg.e_grid_desc_m_n_,
                                            arg.block_2_etile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_gemm_multiple_d_xdl_cshuffle_lds_direct_load<
                    GridwiseGemm,
                    ADataType,
                    BDataType,
                    typename GridwiseGemm::DsGridPointer,
                    EDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    typename GridwiseGemm::AGridDesc_AK0_M_AK1,
                    typename GridwiseGemm::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::Block2ETileMap,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.a_grid_desc_ak0_m_ak1_,
                                              arg.b_grid_desc_bk0_n_bk1_,
                                              arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.block_2_etile_map_);
            };

            const auto K = arg.a_grid_desc_m_k_.GetLength(I1);

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        INVOKER_RUN3_IMPL

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_wmma_supported<ADataType, BDataType, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if(!ck::is_lds_direct_load_supported())
        {
            return false;
        }

        // Check vector load/store.
        {
            using Row = ck::tensor_layout::gemm::RowMajor;
            using Col = ck::tensor_layout::gemm::ColumnMajor;

            // Check vector load of A.
            if constexpr(is_same_v<ALayout, Row> && ABlockTransferSrcVectorDim == 2)
            {
                if(arg.KRaw_ % ABlockTransferScalarPerVector != 0)
                {
                    return false;
                }
            }
            else if constexpr(is_same_v<ALayout, Col> && ABlockTransferSrcVectorDim == 1)
            {
                if(arg.MRaw_ % ABlockTransferScalarPerVector != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // Check vector load of B.
            if constexpr(is_same_v<BLayout, Col> && BBlockTransferSrcVectorDim == 2)
            {
                if(arg.KRaw_ % BBlockTransferScalarPerVector != 0)
                {
                    return false;
                }
            }
            else if constexpr(is_same_v<BLayout, Row> && BBlockTransferSrcVectorDim == 1)
            {
                if(arg.NRaw_ % BBlockTransferScalarPerVector != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // Check vector load of Ds.
            // For now, only the RowMajor layout is supported.
            bool all_valid = true;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                if constexpr(!is_same_v<DLayout, Row>)
                {
                    all_valid = false;
                }
            });

            if(!all_valid)
            {
                return false;
            }

            // Check vector load of E.
            // For now, only the RowMajor layout is supported.
            if constexpr(is_same_v<ELayout, Row>)
            {
                if(arg.NRaw_ % CDEBlockTransferScalarPerVector_NPerBlock != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                return GridwiseGemm64::CheckValidity(arg.a_grid_desc_m_k_,
                                                     arg.b_grid_desc_n_k_,
                                                     arg.ds_grid_desc_m_n_,
                                                     arg.e_grid_desc_m_n_,
                                                     arg.block_2_etile_map_);
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                return GridwiseGemm32::CheckValidity(arg.a_grid_desc_m_k_,
                                                     arg.b_grid_desc_n_k_,
                                                     arg.ds_grid_desc_m_n_,
                                                     arg.e_grid_desc_m_n_,
                                                     arg.block_2_etile_map_);
            }
        }
        return false;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_e,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideE,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        MRaw,
                        NRaw,
                        KRaw,
                        StrideA,
                        StrideB,
                        StrideDs,
                        StrideE,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        index_t MRaw,
                        index_t NRaw,
                        index_t KRaw,
                        index_t StrideA,
                        index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        index_t StrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          MRaw,
                                          NRaw,
                                          KRaw,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideE,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{
            {PipelineVersion::v1, "v1"}, {PipelineVersion::v2, "v2"}, {PipelineVersion::v4, "v4"}};

        // clang-format off
        str << "DeviceGemmMultipleD_Xdl_CShuffle_LdsDirectLoad"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferScalarPerVector << ", "
            << BBlockTransferScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">"
            << " LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
