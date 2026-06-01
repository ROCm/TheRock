// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
 * limitations.
 *
 * \tparam Block2CTileMap Block2CTileMap::CalculateBottomIndex() takes in id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 *
 * \note Using \p ComputePtrOffsetOfBatch gives us the flexibility that 2 workgroups can compute 2
 * tiles from different matrices. Keep in mind that these 2 matrices can share the same grid
 * descriptor (like in BatchedGEMM), or use their own grid descriptors (in GroupedGemm). \link
 * device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for \link
 * DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the computing of
 * pointer offset into \p ComputePtrOffsetOfStridedBatch.
 *
 * \note \p Block2CTileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemm and GroupedGemm (and the corresponding GEMM fusion).
 *
 */
template <typename DeviceOp, typename GridwiseGemm, bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_batched_gemm_xdlops_v2r3(const typename DeviceOp::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        const index_t num_blocks_per_batch =
            __builtin_amdgcn_readfirstlane(get_grid_size() / karg.Batch);
        const index_t g_idx =
            __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

        const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(karg.compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
        const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(karg.compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
        const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(karg.compute_ptr_offset_of_batch.GetCPtrOffset(g_idx)));

        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        const auto a_grid_desc_k0_m_k1 =
            amd_wave_read_first_lane(GridwiseGemm::MakeAGridDescriptor_K0_M_K1(
                karg.M, karg.MPadded, karg.K, karg.K0, karg.StrideA));
        const auto b_grid_desc_k0_n_k1 =
            amd_wave_read_first_lane(GridwiseGemm::MakeBGridDescriptor_K0_N_K1(
                karg.K, karg.N, karg.NPadded, karg.K0, karg.StrideB));
        const auto c_grid_desc_m_n = amd_wave_read_first_lane(GridwiseGemm::MakeCGridDescriptor_M_N(
            karg.M, karg.MPadded, karg.N, karg.NPadded, karg.StrideC));

        GridwiseGemm::template Run<HasMainKBlockLoop>(karg.p_a_grid + a_batch_offset,
                                                      karg.p_b_grid + b_batch_offset,
                                                      karg.p_c_grid + c_batch_offset,
                                                      p_shared,
                                                      a_grid_desc_k0_m_k1,
                                                      b_grid_desc_k0_n_k1,
                                                      c_grid_desc_m_n);
    }
#else
    ignore = karg;
#endif
}

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
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector,
          ck::index_t NumGemmKPrefetchStage = 1,
          ck::LoopScheduler LoopSched       = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer   = ck::PipelineVersion::v1>
struct DeviceBatchedGemmXdl : public DeviceBatchedGemm<ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       ADataType,
                                                       BDataType,
                                                       CDataType,
                                                       AElementwiseOperation,
                                                       BElementwiseOperation,
                                                       CElementwiseOperation>
{
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto K1Number = Number<K1>{};

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                       index_t BatchStrideB,
                                       index_t BatchStrideC)
            : BatchStrideA_(BatchStrideA), BatchStrideB_(BatchStrideB), BatchStrideC_(BatchStrideC)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB_);
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC_);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        index_t BatchStrideC_;
    };

    // GridwiseGemm
    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3_ext<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::Set,
        ALayout,
        BLayout,
        CLayout,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpecialization::MNKPadding,
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
        Sequence<2, 3, 0, 1, 7, 5, 4, 6>,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector,
        NumGemmKPrefetchStage,
        LoopSched,
        PipelineVer>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    using Problem = typename GridwiseGemm64::Problem;

    // Argument
    struct Argument : public Problem, public BaseArgument
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
                 index_t BatchStrideA,
                 index_t BatchStrideB,
                 index_t BatchStrideC,
                 index_t Batch_)
            : Problem{M_, N_, K_, StrideA_, StrideB_, StrideC_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_},
              Batch(Batch_),
              compute_ptr_offset_of_batch{BatchStrideA, BatchStrideB, BatchStrideC}
        {
        }

        const ADataType* p_a_grid;
        const BDataType* p_b_grid;
        CDataType* p_c_grid;
        index_t Batch;
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceBatchedGemmXdl::Argument;

        template <typename GridwiseGemm>
        float RunImp(const Argument& karg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                karg.Print();
            }

            typename GridwiseGemm::Problem arg(
                karg.M, karg.N, karg.K, karg.StrideA, karg.StrideB, karg.StrideC);
            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3_ext has invalid setting");
            }

            auto [gdx, gdy, gdz] = GridwiseGemm::CalculateGridSize(karg.M, karg.N);
            gdx *= karg.Batch;

            float ave_time = 0;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(karg.K))
            {
                const auto kernel =
                    kernel_batched_gemm_xdlops_v2r3<DeviceBatchedGemmXdl, GridwiseGemm, true>;

                ave_time = launch_and_time_kernel(
                    stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, karg);
            }
            else
            {
                const auto kernel =
                    kernel_batched_gemm_xdlops_v2r3<DeviceBatchedGemmXdl, GridwiseGemm, false>;

                ave_time = launch_and_time_kernel(
                    stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, karg);
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

    static bool IsSupportedArgument(const Problem& problem)
    {
        if(!ck::is_xdl_wmma_supported<ADataType, BDataType, MPerXDL, NPerXDL>())
        {
            return false;
        }
        // temp disable on gfx11
        if(ck::is_gfx11_supported())
        {
            return false;
        }
        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                return GridwiseGemm64::CheckValidity(problem);
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                return GridwiseGemm32::CheckValidity(
                    reinterpret_cast<const typename GridwiseGemm32::Problem&>(problem));
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
                             index_t BatchStrideA,
                             index_t BatchStrideB,
                             index_t BatchStrideC,
                             index_t Batch)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        BatchStrideA,
                        BatchStrideB,
                        BatchStrideC,
                        Batch};
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
                                                      index_t BatchStrideA,
                                                      index_t BatchStrideB,
                                                      index_t BatchStrideC,
                                                      index_t Batch,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation) override
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
                                          BatchStrideA,
                                          BatchStrideB,
                                          BatchStrideC,
                                          Batch);
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

        // clang-format off
        str << "DeviceBatchedGemmXdl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ">"
            << " NumGemmKPrefetchStage: "
            << NumGemmKPrefetchStage << ", "
            << "LoopScheduler: "
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
