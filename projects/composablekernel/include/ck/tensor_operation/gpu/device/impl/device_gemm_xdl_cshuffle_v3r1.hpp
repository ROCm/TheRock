// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <typeinfo>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_threadwise_multi_d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
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
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ReduceDataType                     = CDataType,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA>
struct DeviceGemm_Xdl_CShuffleV3R1 : public DeviceGemmV2R1<ALayout,
                                                           BLayout,
                                                           DsLayout,
                                                           CLayout,
                                                           ADataType,
                                                           BDataType,
                                                           DsDataType,
                                                           CDataType,
                                                           AElementwiseOperation,
                                                           BElementwiseOperation,
                                                           CElementwiseOperation>
{
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    static constexpr index_t NumDTensor = DsDataType::Size();

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    // GridwiseGemm
    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        CLayout,
        ADataType,
        BDataType,
        GemmAccDataType,
        CShuffleDataType,
        ReduceDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        PassThrough,
        GemmSpec,
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
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    struct Argument : public GridwiseGemm64::Argument
    {
        Argument(const ADataType* p_a_grid_,
                 const BDataType* p_b_grid_,
                 const std::array<const void*, NumDTensor> p_ds_,
                 CDataType* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 std::array<ck::index_t, NumDTensor> StrideDs_,
                 index_t StrideC_,
                 index_t k_batch_)
            : GridwiseGemm64::Argument(p_a_grid_,
                                       p_b_grid_,
                                       reinterpret_cast<ReduceDataType*>(p_c_grid_),
                                       M_,
                                       N_,
                                       K_,
                                       StrideA_,
                                       StrideB_,
                                       StrideC_,
                                       k_batch_,
                                       true),
              p_ds(p_ds_),
              StrideDs(StrideDs_)
        {
        }

        const std::array<const void*, NumDTensor> p_ds;
        std::array<ck::index_t, NumDTensor> StrideDs;
    };

    using ReduceAdd               = ck::reduce::Add;
    using OutElementwiseOperation = CElementwiseOperation;

    static constexpr auto DsVectorLengthSequence = generate_sequence_v2(
        [](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
            if constexpr(std::is_same<CLayout, DLayout>::value)
                return Number<CShuffleBlockTransferScalarPerVector_NPerBlock>{};
            else
                return Number<1>{};
        },
        Number<NumDTensor>{});

    using DeviceReduceInstance = DeviceReduceThreadWiseMultiD<
        ReduceDataType,  // InDataType,
        DsDataType,      // DsDatatype
        GemmAccDataType, // AccDataType,
        CDataType,       // OutDataType,
        3,               // Rank
        1,               // NumReduceDim
        ReduceAdd,
        PassThrough,
        OutElementwiseOperation,
        256,                                            // BlockSize_,
        CShuffleBlockTransferScalarPerVector_NPerBlock, // MThreadSliceSize_,
        1,                                              // KThreadSliceSize_,
        0,                                              // InSrcVectorDim_,
        CShuffleBlockTransferScalarPerVector_NPerBlock, // InSrcVectorSize_,
        CShuffleBlockTransferScalarPerVector_NPerBlock, // OutDstVectorSize_
        decltype(DsVectorLengthSequence)>;

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float RunReduce(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            static constexpr index_t NumInDim  = 3;
            static constexpr index_t NumOutDim = 2;

            std::array<ck::index_t, NumInDim> in_lengths   = {arg.KBatch, arg.M, arg.N};
            std::array<ck::index_t, NumOutDim> out_lengths = {arg.M, arg.N};

            std::array<ck::index_t, NumInDim> in_strides;
            std::array<ck::index_t, NumOutDim> out_strides;
            if constexpr(std::is_same<CLayout, ck::tensor_layout::gemm::RowMajor>::value)
            {
                in_strides  = {arg.M * arg.N, arg.N, 1};
                out_strides = {arg.N, 1};
            }
            else
            {
                in_strides  = {arg.M * arg.N, 1, arg.M};
                out_strides = {1, arg.M};
            }

            std::array<int, 1> reduce_dims{0};

            std::array<std::array<index_t, NumOutDim>, NumDTensor> DsLengths;
            std::array<std::array<index_t, NumOutDim>, NumDTensor> DsStrides;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                DsLengths[i] = out_lengths;

                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same<DLayout, ck::tensor_layout::gemm::RowMajor>::value)
                {
                    DsStrides[i] = {arg.StrideDs[i], 1};
                }
                else
                {
                    DsStrides[i] = {1, arg.StrideDs[i]};
                }
            });

            auto reduce = DeviceReduceInstance{};

            auto argument_ptr = reduce.MakeArgumentPointer(in_lengths,
                                                           in_strides,
                                                           DsLengths,
                                                           DsStrides,
                                                           out_lengths,
                                                           out_strides,
                                                           reduce_dims,
                                                           arg.p_workspace_,
                                                           arg.p_ds,
                                                           arg.p_c_grid,
                                                           PassThrough{},
                                                           OutElementwiseOperation{});

            auto invoker_ptr = reduce.MakeInvokerPointer();

            float ave_time = 0;

            if(reduce.IsSupportedArgument(argument_ptr.get()))
            {
                ave_time = invoker_ptr->Run(argument_ptr.get(), stream_config);
            }
            else
            {
                throw std::runtime_error(
                    "The runtime parameters seems not supported by the device instance, exiting!");
            }

            return ave_time;
        }

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg_, const StreamConfig& stream_config = StreamConfig{})
        {
            auto arg = *reinterpret_cast<const typename GridwiseGemm::Argument*>(&arg_);

            if(!(!(arg.IsReduceAdd() || NumDTensor > 0) &&
                 std::is_same<CDataType, ReduceDataType>::value))
            {
                if(arg.p_workspace_ == nullptr)
                {
                    throw std::runtime_error("using reduce , but empty workspace!");
                }

                arg.p_c_grid = reinterpret_cast<ReduceDataType*>(arg.p_workspace_);
            }

            if(stream_config.log_level_ > 0)
            {
                arg.Print();
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
                    ck::utility::RotatingMemWrapper<typename GridwiseGemm::Argument> rotating_mem(
                        arg,
                        stream_config.rotating_count,
                        arg.M * arg.K * sizeof(ADataType),
                        arg.K * arg.N * sizeof(BDataType));
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg);
                }
                else
                {
                    ave_time = launch_and_time_kernel(
                        stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, arg);
                }
            };

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {

                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy>;
                    Run(kernel);
                }
                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                        true,
                                                        InMemoryDataOperationEnum::Set,
                                                        minimum_occupancy,
                                                        TailNumber::One>;
                        Run(kernel);
                    }
                    else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Full)
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                        true,
                                                        InMemoryDataOperationEnum::Set,
                                                        minimum_occupancy,
                                                        TailNumber::Full>;
                        Run(kernel);
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            true,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy,
                                                            TailNumber::Two>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Three)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            true,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy,
                                                            TailNumber::Three>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Four)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            true,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy,
                                                            TailNumber::Four>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Five)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            true,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy,
                                                            TailNumber::Five>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            true,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy,
                                                            TailNumber::Six>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Seven)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            true,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy,
                                                            TailNumber::Seven>;
                            Run(kernel);
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {

                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3_2lds<GridwiseGemm,
                                                             true,
                                                             InMemoryDataOperationEnum::Set,
                                                             minimum_occupancy,
                                                             TailNumber::Odd>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3_2lds<GridwiseGemm,
                                                             true,
                                                             InMemoryDataOperationEnum::Set,
                                                             minimum_occupancy,
                                                             TailNumber::Even>;
                        Run(kernel);
                    }
                }
                else
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                        true,
                                                        InMemoryDataOperationEnum::Set,
                                                        minimum_occupancy,
                                                        TailNumber::Odd>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                        true,
                                                        InMemoryDataOperationEnum::Set,
                                                        minimum_occupancy,
                                                        TailNumber::Even>;
                        Run(kernel);
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {

                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                    false,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy>;
                    Run(kernel);
                }
            }

            if(!(!(arg.IsReduceAdd() || NumDTensor > 0) &&
                 std::is_same<CDataType, ReduceDataType>::value))
            {
                // reduce c data
                ave_time += RunReduce(arg_, stream_config);
            }
            return ave_time;
        }

        INVOKER_RUN_IMPL

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
        if(!ck::is_xdl_wmma_supported<ComputeTypeA, ComputeTypeB, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }

        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                return GridwiseGemm64::CheckValidity(arg);
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                return GridwiseGemm32::CheckValidity(
                    reinterpret_cast<const typename GridwiseGemm32::Argument&>(arg));
            }
        }
        return false;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             const std::array<const void*, NumDTensor> p_ds,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<ck::index_t, NumDTensor> StrideDs,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CElementwiseOperation)
    {
        return Argument{p_a, p_b, p_ds, p_c, M, N, K, StrideA, StrideB, StrideDs, StrideC, KBatch};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      std::array<const void*, NumDTensor> p_ds,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      std::array<ck::index_t, NumDTensor> StrideDs,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          p_ds,
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideC,
                                          KBatch);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceGemmXdlUniversalReduce"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock<<"x"<<NPerBlock<<"x"<<KPerBlock << ", "
            << "WaveTile: "
            << MPerXDL<<"x"<<NPerXDL << ", "
            << "WaveMap: "
            << MXdlPerWave<<"x" << NXdlPerWave<<", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector<<"x"<<BBlockTransferSrcScalarPerVector<<", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm64::BlockwiseGemmPipe::PrefetchStages;
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = *dynamic_cast<const Argument*>(p_arg);

        if(!(!(arg.IsReduceAdd() || NumDTensor > 0) &&
             std::is_same<CDataType, ReduceDataType>::value))
        {
            std::cout << "using workspace" << std::endl;
            return arg.M * arg.N * arg.KBatch * sizeof(ReduceDataType);
        }

        return 0;
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
