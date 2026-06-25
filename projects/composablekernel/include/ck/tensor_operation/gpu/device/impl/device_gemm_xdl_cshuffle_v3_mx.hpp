// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"

#include "ck/host_utility/flush_cache.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_mx.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_mx.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_mx_bpreshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// clang-format off
/**
 * \brief WIP: Implements XDL CShuffle V3 GEMM for microscale-compliant data types
 *
 * This class is a work-in-progress implementation of the XDL CShuffle V3 GEMM for
 * microscale-compliant data types.
 *
 * Assumptions:
 * - A and B data types are compliant with the OCP Microscaling Formats (MX) Specification
 * - Each scale applies to ScaleBlockSize elements in K direction
 * - A scale matrix is a row-major
 * - B scale matrix is a column-major
 * - Scale data types must have get_exponent_value() specialization, whereas lowest 8 bits of the
 * exponent will be interpreted as conventional biased Float32 exponent (E8M0)
 *
 * Tunable parameters.
 * The CK instance includes a series of tunable template parameters to control the parallel
 * granularity of the workload to achieve load balancing on different hardware platforms. These
 * parameters include Block Size, M/N/K Per Block, M/N per XDL, AK1, BK1, etc.
 *  - Block Size determines the number of threads in the thread block.
 *  - M/N/K Per Block determines the size of tile that each thread block is responsible for
 * calculating.
 *  - M/N Per XDL refers to M/N size for Instinct accelerator Matrix Fused Multiply Add (MFMA)
 * instructions operating on a per-wavefront basis.
 *  - A/B K1 is related to the data type. It can be any value ranging from 1 to K Per Block. To
 * achieve the optimal load/store performance, 128bit per load is suggested. In addition, the A/B
 * loading parameters must be changed accordingly to match the A/B K1 value; otherwise, it will
 * result in compilation errors.
 *
 * Conditions for achieving computational load balancing on different hardware platforms can vary.
 *
 * \tparam KPerBlock is the number of elements in K dimension that each block processes (multiply with packed_size_v to get the actual KPerBlock)
 *
 * Serialized version of the algorithm:
 * \code
 * // E = A * B + C
 * // Loop over E[MPerBlock,NPerBlock] tiles
 * for(int mb = 0; mb < M; mb += MPerBlock){
 *    for(int nb = 0; nb < N; nb += NPerBlock){
 *       // initialize E[MPerBlock,NPerBlock] tile
 *       for(int mt = mb; mt < mb + MPerBlock; mt++){
 *          for(int nt = nb; nt < nb + NPerBlock; nt++){
 *             E[mt,nt] = C[mt,nt];
 *          }
 *       }
 *
 *       // multiply-accumulate per tile
 *       for(int kb = 0; kb < K; kb += KPerBlock){
 *         for(int m0 = mb; m0 < mb + MPerBlock; m0 += MWaves * MPerXDL){
 *           for(int n0 = nb; n0 < nb + NPerBlock; n0 += NWaves * NPerXDL){
 *             for(int mw = m0; mw < m0 + MWaves * MPerXDL; mw += MPerXDL){
 *               for(int nw = n0; nw < n0 + NWaves * NPerXDL; nw += NPerXDL){
 *                 for(int k0 = kb; k0 < kb + KPerBlock; k0 += mfma.num_input_blks*KPack){
 *                   // MFMA accumulation
 *                   for(int k_pack = k0; k_pack < k0 + mfma.num_input_blks*KPack; k_pack += KPerXdlops){
 *                     // MFMA instruction
 *                     for(int k_mfma = k_pack; k_mfma < k_pack + KPerXdlops; k_mfma += mfma.k_per_blk){
 *                       for(int m = mw; m < mw + MPerXDL; m++){
 *                         for(int n = nw; n < nw + NPerXDL; n++){
 *                           for(int k = k_mfma; k < k_mfma + mfma.k_per_blk; k++){
 *                            E[m,n] += A[m,k] * B[k,n];
 *                           }
 *                         }
 *                       }
 *                     }
 *                   }
 *                 }
 *               }
 *             }
 *           }
 *         }
 *       }
 *    }
 * }
 * \endcode
 *
 */
// clang-format on
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename CDataType,
          typename GemmAccDataType, // TODO: always float
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t ScaleBlockSize, // Scaling block size
          index_t BlockSize,      // Thread block size
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock, // multiply with packed_size_v to get the actual KPerBlock
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
          typename ComputeTypeA =
              ADataType, // XXX: These should always be the same as ADataType and BDataType
          typename ComputeTypeB =
              BDataType // TODO: Hardcode them and remove from the list of template parameters
          >
struct DeviceGemmMX_Xdl_CShuffleV3 : public DeviceGemmMX<ALayout,
                                                         BLayout,
                                                         CLayout,
                                                         ADataType,
                                                         AScaleDataType,
                                                         BDataType,
                                                         BScaleDataType,
                                                         CDataType,
                                                         ScaleBlockSize,
                                                         AElementwiseOperation,
                                                         BElementwiseOperation,
                                                         CElementwiseOperation>
{
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    // GridwiseGemm
    template <index_t NXdlPerWave_>
    using GridwiseGemmMXBase = GridwiseGemmMX_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        CLayout,
        ADataType,
        AScaleDataType,
        BDataType,
        BScaleDataType,
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        ScaleBlockSize,
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
    template <index_t NXdlPerWave_>
    using GridwiseGemmMXBPreshuffleBase = GridwiseGemmMX_xdl_cshuffle_v3_bpreshuffle<
        ALayout,
        BLayout,
        CLayout,
        ADataType,
        AScaleDataType,
        BDataType,
        BScaleDataType,
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        ScaleBlockSize,
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

    using GridwiseGemm64 = conditional_t< //
        !is_same_v<BLayout, tensor_layout::gemm::MFMA>,
        GridwiseGemmMXBase<math::max(NXdlPerWave64, 1)>,
        GridwiseGemmMXBPreshuffleBase<math::max(NXdlPerWave64, 1)>>;
    using GridwiseGemm32 = conditional_t< //
        !is_same_v<BLayout, tensor_layout::gemm::MFMA>,
        GridwiseGemmMXBase<NXdlPerWave32>,
        GridwiseGemmMXBPreshuffleBase<NXdlPerWave32>>;

    using Argument = typename GridwiseGemm64::Argument;

    // Invoker
    struct Invoker : public BaseInvoker
    {
        template <typename GridwiseGemm>
        float RunImp(const typename GridwiseGemm::Argument& arg,
                     const StreamConfig& stream_config = StreamConfig{})
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
                    auto arg_ = arg;

                    const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(
                        arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideA, arg_.AK0);
                    const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(
                        arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideB, arg_.BK0);

                    auto size_a_buffer =
                        a_grid_desc_ak0_m_ak1.GetElementSpaceSize() * sizeof(ADataType);
                    auto size_b_buffer =
                        b_grid_desc_bk0_n_bk1.GetElementSpaceSize() * sizeof(BDataType);

                    ck::utility::RotatingMemWrapper<typename GridwiseGemm::Argument> rotating_mem(
                        arg_, stream_config.rotating_count, size_a_buffer, size_b_buffer);
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        // clear c mem
                        if(arg_.KBatch > 1)
                            hipGetErrorString(hipMemsetAsync(arg_.p_c_grid,
                                                             0,
                                                             arg_.M * arg_.N * sizeof(CDataType),
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
                        hipGetErrorString(hipMemsetAsync(arg.p_c_grid,
                                                         0,
                                                         arg.M * arg.N * sizeof(CDataType),
                                                         stream_config.stream_id_));

                    ave_time = launch_and_time_kernel(
                        stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, arg);
                }
            };

            // TODO: Check if this is the right algorithm for minimum_occupancy
            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave
                    ? (BlkGemmPipelineVer == BlockGemmPipelineVersion::v3 &&
                       MPerBlock * NPerBlock * KPerBlock * sizeof(ADataType) <= 128 * 128 * 64 * 2)
                          ? 2
                          : 1
                    : 2;

            constexpr auto TailNumChoices = []() {
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                    return Tuple<constant<TailNumber::Full>>{};
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                    return Tuple<constant<TailNumber::Even>, constant<TailNumber::Odd>>{};
                else
                    static_assert(false, "Unexpected BlkGemmPipelineVer!");
            }();
            constexpr bool Use2LDS = []() {
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                    return false;
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                    return true;
                else
                    static_assert(false, "Unexpected BlkGemmPipelineVer!");
            }();
            const TailNumber tail_num = GridwiseGemm::CalculateKBlockLoopTailNum(K_split);
            using BoolChoices         = Tuple<ck::true_type, ck::false_type>;
            static_for_product<BoolChoices,
                               BoolChoices,
                               remove_cvref_t<decltype(TailNumChoices)>>{}(
                [&](auto mainloop_choice, auto KBatch_cond_choice, auto tail_num_choice) {
                    constexpr auto CGlobalMemoryDataOperation =
                        KBatch_cond_choice.value ? InMemoryDataOperationEnum::AtomicAdd
                                                 : InMemoryDataOperationEnum::Set;
                    if(mainloop_choice.value == has_main_k_block_loop &&
                       KBatch_cond_choice.value == (arg.KBatch > 1) &&
                       tail_num_choice.value == tail_num)
                    {
                        const auto kernel = kernel_gemm_xdl_cshuffle_v3_mx< //
                            Use2LDS,
                            GridwiseGemm,
                            mainloop_choice.value,
                            CGlobalMemoryDataOperation,
                            minimum_occupancy,
                            tail_num_choice.value>;
                        Run(kernel);
                    }
                });
            return ave_time;
        }

        INVOKER_RUN3_IMPL
        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        static_assert(is_scale_mfma_data_type<ADataType>() && is_scale_mfma_data_type<BDataType>(),
                      "Only microscaling formats are supported for ADataType and BDataType");

        static_assert(ScaleBlockSize == 32, "Only ScaleBlockSize 32 is supported");

        static_assert(is_same_v<ComputeTypeA, ADataType> && is_same_v<ComputeTypeB, BDataType>,
                      "ComputeTypeA and ComputeTypeB must be the same as ADataType and BDataType");

        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if constexpr(!IsValidCompilationParameter())
        {
            return false;
        }

        if(ck::get_device_name() != "gfx950")
        {
            return false;
        }

        if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> && arg.KBatch > 1)
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
                             const AScaleDataType* p_a_scale,
                             const BDataType* p_b,
                             const BScaleDataType* p_b_scale,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideScaleA,
                             index_t StrideB,
                             index_t StrideScaleB,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a,
                        p_a_scale,
                        p_b,
                        p_b_scale,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideScaleA,
                        StrideB,
                        StrideScaleB,
                        StrideC,
                        KBatch,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_a_scale,
                                                      const void* p_b,
                                                      const void* p_b_scale,
                                                      void* p_c,
                                                      ck::index_t M,
                                                      ck::index_t N,
                                                      ck::index_t K,
                                                      ck::index_t StrideA,
                                                      ck::index_t StrideScaleA,
                                                      ck::index_t StrideB,
                                                      ck::index_t StrideScaleB,
                                                      ck::index_t StrideC,
                                                      ck::index_t KBatch,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const AScaleDataType*>(p_a_scale),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<const BScaleDataType*>(p_b_scale),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideScaleA,
                                          StrideB,
                                          StrideScaleB,
                                          StrideC,
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
        str << "DeviceGemmMX_Xdl_CShuffleV3"
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
            << GridwiseGemm64::BlockwiseGemmPipe::PrefetchStages << ", "
            << "Kpack: "
            << GridwiseGemm64::BlockwiseGemmPipe::AMmaKStride << ", "
            << "ScaleBlockSize: "
            << ScaleBlockSize;
        // clang-format on

        return str.str();
    }
    REGISTER_EXTRA_PRINTING_METHODS
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
