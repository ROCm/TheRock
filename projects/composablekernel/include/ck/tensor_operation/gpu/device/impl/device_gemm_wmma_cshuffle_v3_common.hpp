// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t BlockSize,
          index_t AK1,
          index_t BK1,
          GemmSpecialization GemmSpec,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename ComputeTypeA,
          typename ComputeTypeB,
          bool IsBPreShuffled = false>
struct DeviceGemm_Wmma_CShuffleV3_Common
{

    using Argument = typename GridwiseGemm::Argument;

    /// @brief  Helper structure responsible for kernel invocation.
    ///
    /// @paragraph  The `Invoker` class is responsible for preparation and invocation of actual GPU
    ///             kernel function. It usually determines the launched grid size prepares kernel
    ///             arguments as well as perform specific kernel configuration selection based on
    ///             runtime arguments.
    ///
    /// @note       If appropriately configured it may measure kernel execution time.
    ///
    struct Invoker : public BaseInvoker
    {
        /// @brief  This function issues GPU kernel execution.
        /// @param arg           The GPU kernel arguments.
        /// @param stream_config The HIP stream configuration helper structure.
        /// @return              The kernel's average execution time (if time measurement is
        ///                      enabled).
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
                GridwiseGemm::BlockwiseGemmPipe::HotLoopInstList::Print();
            }

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N, arg.KBatch);

            float ave_time = 0;

            index_t k_grain = arg.KBatch * KPerBlock;
            index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    Argument arg_ = arg;

                    const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAsGridDescriptor_AK0_M_AK1(
                        arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideAs, arg_.AK0);
                    const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBsGridDescriptor_BK0_N_BK1(
                        arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideBs, arg_.BK0);

                    std::array<std::size_t, GridwiseGemm::NumATensor> size_as_buffers;
                    static_for<0, GridwiseGemm::NumATensor, 1>{}([&](auto i) {
                        using ADataType    = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
                        size_as_buffers[i] = a_grid_desc_ak0_m_ak1[i].GetElementSpaceSize() *
                                             sizeof(ADataType) / GridwiseGemm::APackedSize;
                    });

                    std::array<std::size_t, GridwiseGemm::NumBTensor> size_bs_buffers;
                    static_for<0, GridwiseGemm::NumBTensor, 1>{}([&](auto i) {
                        using BDataType    = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
                        size_bs_buffers[i] = b_grid_desc_bk0_n_bk1[i].GetElementSpaceSize() *
                                             sizeof(BDataType) / GridwiseGemm::BPackedSize;
                    });

                    const auto ds_grid_desc_m_n = GridwiseGemm::MakeDsGridDescriptor_M_N(
                        arg_.M, arg_.MPadded, arg_.N, arg_.NPadded, arg_.StrideDs);

                    std::array<std::size_t, GridwiseGemm::NumDTensor> size_ds_buffers;
                    static_for<0, GridwiseGemm::NumDTensor, 1>{}([&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                        size_ds_buffers[i] =
                            ds_grid_desc_m_n[i].GetElementSpaceSize() * sizeof(DDataType);
                    });

                    ck::utility::
                        RotatingMemWrapperMultiABD<Argument, AsDataType, BsDataType, DsDataType>
                            rotating_mem(arg_,
                                         stream_config.rotating_count,
                                         size_as_buffers,
                                         size_bs_buffers,
                                         size_ds_buffers);
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        // clear c mem
                        if(arg_.KBatch > 1)
                            HIP_CHECK_ERROR(hipMemsetAsync(arg_.p_e_grid,
                                                           0,
                                                           arg_.M * arg_.N * sizeof(EDataType),
                                                           stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg_);
                }
                else
                {
                    if(arg.KBatch > 1)
                        HIP_CHECK_ERROR(hipMemsetAsync(arg.p_e_grid,
                                                       0,
                                                       arg.M * arg.N * sizeof(EDataType),
                                                       stream_config.stream_id_));

                    ave_time = launch_and_time_kernel(
                        stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, arg);
                }
            };

            constexpr index_t minimum_occupancy = []() {
                if constexpr(BlkGemmPipeSched == BlockGemmPipelineScheduler::Interwave)
                {
                    return 2;
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    return (MPerBlock * NPerBlock / BlockSize <= 128) ? 2 : 1;
                }
                else
                {
                    return 1;
                }
            }();

            // ThreadwiseTensorSliceTransfer_v7r3 (used in ThreadGroupTensorSliceTransfer_v7r3) is
            // currently implemented in such a way that all SrcScalarPerVectors must be the same, so
            // if one of D matrices is column-major, then all SrcScalarPerVectors must be 1. On the
            // other hand, Split K for 16-bit outputs uses packed atomics so ScalarPerVectors cannot
            // be odd.
            constexpr bool AtomicsImplementationExists =
                !(std::is_same_v<EDataType, ck::half_t> || std::is_same_v<EDataType, ck::bhalf_t> ||
                  std::is_same_v<EDataType, int8_t>) ||
                (CDEShuffleBlockTransferScalarPerVectors{}[0] % 2 == 0);

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(IsBPreShuffled)
                {
                    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                    {
                        if(arg.KBatch > 1)
                        {
                            if constexpr(AtomicsImplementationExists)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                   TailNumber::Odd)
                                {
                                    const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Odd>;
                                    Run(kernel);
                                }
                                else
                                {
                                    const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Even>;
                                    Run(kernel);
                                }
                            }
                        }
                        else
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                            {
                                const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                                Run(kernel);
                            }
                        }
                    }
                }
                else
                {
                    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                                 BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                    {
                        if(arg.KBatch > 1)
                        {
                            if constexpr(AtomicsImplementationExists)
                            {
                                const auto kernel = kernel_gemm_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy>;
                                Run(kernel);
                            }
                        }
                        else
                        {
                            const auto kernel =
                                kernel_gemm_wmma_cshuffle_v3<GridwiseGemm,
                                                             true,
                                                             InMemoryDataOperationEnum::Set,
                                                             minimum_occupancy>;
                            Run(kernel);
                        }
                    }
                }
            }
            else
            {
                if constexpr(IsBPreShuffled)
                {
                    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                    {
                        if(arg.KBatch > 1)
                        {
                            if constexpr(AtomicsImplementationExists)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                   TailNumber::Odd)
                                {
                                    const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                        GridwiseGemm,
                                        false,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Odd>;
                                    Run(kernel);
                                }
                                else
                                {
                                    const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                        GridwiseGemm,
                                        false,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Even>;
                                    Run(kernel);
                                }
                            }
                        }
                        else
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                            {
                                const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    false,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    false,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                                Run(kernel);
                            }
                        }
                    }
                }
                else
                {
                    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                    {
                        if(arg.KBatch > 1)
                        {
                            if constexpr(AtomicsImplementationExists)
                            {
                                const auto kernel = kernel_gemm_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    false,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy>;
                                Run(kernel);
                            }
                        }
                        else
                        {
                            const auto kernel =
                                kernel_gemm_wmma_cshuffle_v3<GridwiseGemm,
                                                             false,
                                                             InMemoryDataOperationEnum::Set,
                                                             minimum_occupancy>;
                            Run(kernel);
                        }
                    }
                }
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            return false;
        }

        if constexpr(std::is_same_v<EDataType, ck::half_t> ||
                     std::is_same_v<EDataType, ck::bhalf_t>)
        {
            if(arg.KBatch > 1 && ck::is_gfx11_supported())
            {
                // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
                return false;
            }
        }

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                return false;
            }
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }

        if constexpr(IsBPreShuffled)
        {
            if(arg.N % NPerBlock != 0 || arg.K % KPerBlock != 0)
            {
                return false;
            }
        }

        return GridwiseGemm::CheckValidity(arg);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
