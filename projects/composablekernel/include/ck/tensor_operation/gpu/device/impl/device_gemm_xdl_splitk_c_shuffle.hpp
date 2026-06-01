// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL,
          typename ComputeType        = CDataType,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          typename LDSTypeA           = ComputeType,
          typename LDSTypeB           = ComputeType>

struct DeviceGemmXdlSplitKCShuffle : public DeviceGemmSplitK<ALayout,
                                                             BLayout,
                                                             CLayout,
                                                             ADataType,
                                                             BDataType,
                                                             CDataType,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation,
                                                             ComputeType>
{
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    // TODO: should be exposed as Tparams.
    static constexpr index_t NumGemmKPrefetchStage = 1;

    using ComputeTypeA = ComputeType;
    using ComputeTypeB = ComputeType;

    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2<
        BlockSize,
        ADataType,
        BDataType,
        AccDataType,
        CDataType,
        ALayout,
        BLayout,
        CLayout,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        NumGemmKPrefetchStage,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave_,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CBlockTransferScalarPerVector_NWaveNPerXDL,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        LoopSched,
        PipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        LDSTypeA,
        LDSTypeB>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    struct Argument : public GridwiseGemm64::Argument
    {
        Argument(const ADataType* p_a_grid_,
                 const BDataType* p_b_grid_,
                 CDataType* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_,
                 index_t MPadded_,
                 index_t NPadded_,
                 index_t KPadded_,
                 index_t K0Padded_,
                 index_t k_batch_,
                 AElementwiseOperation a_element_op_,
                 BElementwiseOperation b_element_op_,
                 CElementwiseOperation c_element_op_)
            : GridwiseGemm64::Argument(p_a_grid_,
                                       p_b_grid_,
                                       p_c_grid_,
                                       M_,
                                       N_,
                                       K_,
                                       StrideA_,
                                       StrideB_,
                                       StrideC_,
                                       MPadded_,
                                       NPadded_,
                                       KPadded_,
                                       K0Padded_,
                                       k_batch_),
              a_element_op(a_element_op_),
              b_element_op(b_element_op_),
              c_element_op(c_element_op_)
        {
        }

        AElementwiseOperation a_element_op;
        BElementwiseOperation b_element_op;
        CElementwiseOperation c_element_op;
    };

    using DefaultBlock2CTileMap = typename GridwiseGemm64::DefaultBlock2CTileMap;

    // Invoker
    struct Invoker : public BaseInvoker
    {

        void Print(const Argument& karg) { karg.Print(); }

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                Print(arg);
            }

            typename GridwiseGemm::Argument karg(arg.p_a_grid,
                                                 arg.p_b_grid,
                                                 arg.p_c_grid,
                                                 arg.M,
                                                 arg.N,
                                                 arg.K,
                                                 arg.StrideA,
                                                 arg.StrideB,
                                                 arg.StrideC,
                                                 arg.MPadded,
                                                 arg.NPadded,
                                                 arg.KPadded,
                                                 arg.K0Padded,
                                                 arg.k_batch);
            const auto kbatch = karg.k_batch;
            if(!GridwiseGemm::CheckValidity(karg))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2 has invalid "
                    "setting");
            }

            const auto b2c_map = DefaultBlock2CTileMap{};
            index_t gdx, gdy, gdz;
            ck::tie(gdx, gdy, gdz) = b2c_map.CalculateGridSize(karg.M, karg.N, karg.k_batch);
            const auto K0Padded    = karg.K0Padded;

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0Padded);

            float ave_time = 0;

            const auto Run = [&](const auto& kernel) {
                if(kbatch > 1)
                    hipGetErrorString(hipMemsetAsync(karg.p_c_grid,
                                                     0,
                                                     karg.M * karg.N * sizeof(CDataType),
                                                     stream_config.stream_id_));

                ave_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(gdx, gdy, gdz),
                                           dim3(BlockSize),
                                           0,
                                           static_cast<typename GridwiseGemm::Argument>(karg),
                                           b2c_map,
                                           arg.a_element_op,
                                           arg.b_element_op,
                                           arg.c_element_op);
            };

            if(has_main_k0_block_loop)
            {
                if(kbatch == 1)
                {
                    const auto kernel =
                        kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                             true,
                                                             InMemoryDataOperationEnum::Set,
                                                             DefaultBlock2CTileMap,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel =
                        kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                             true,
                                                             InMemoryDataOperationEnum::AtomicAdd,
                                                             DefaultBlock2CTileMap,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation>;

                    Run(kernel);
                }
            }
            else
            {
                if(kbatch == 1)
                {
                    const auto kernel =
                        kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                             false,
                                                             InMemoryDataOperationEnum::Set,
                                                             DefaultBlock2CTileMap,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel =
                        kernel_gemm_xdlops_v2r4r2_simplified<GridwiseGemm,
                                                             false,
                                                             InMemoryDataOperationEnum::AtomicAdd,
                                                             DefaultBlock2CTileMap,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation>;

                    Run(kernel);
                }
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

    static bool IsSupportedArgument(const Argument& karg)
    {
        // gfx11 doesn't support float atomic
        if(ck::is_gfx11_supported())
        {
            return false;
        }
        if(!ck::is_xdl_wmma_supported<ComputeTypeA, ComputeTypeB, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                return GridwiseGemm64::CheckValidity(karg);
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                return GridwiseGemm32::CheckValidity(
                    reinterpret_cast<const typename GridwiseGemm32::Argument&>(karg));
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
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             index_t KBatch)
    {
        return Argument(p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        GridwiseGemm64::CalculateMPadded(M),
                        GridwiseGemm64::CalculateNPadded(N),
                        GridwiseGemm64::CalculateKPadded(K, KBatch),
                        GridwiseGemm64::CalculateK0Padded(K, KBatch),
                        KBatch,
                        a_element_op,
                        b_element_op,
                        c_element_op);
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      ck::index_t KBatch = 1) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          GridwiseGemm64::CalculateMPadded(M),
                                          GridwiseGemm64::CalculateNPadded(N),
                                          GridwiseGemm64::CalculateKPadded(K, KBatch),
                                          GridwiseGemm64::CalculateK0Padded(K, KBatch),
                                          KBatch,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
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

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{{PipelineVersion::v1, "v1"},
                                                                       {PipelineVersion::v2, "v2"}};

        str << GridwiseGemm64::GetTypeString() << " LoopScheduler: " << LoopSchedToString[LoopSched]
            << ", PipelineVersion: " << PipelineVersionToString[PipelineVer];

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
