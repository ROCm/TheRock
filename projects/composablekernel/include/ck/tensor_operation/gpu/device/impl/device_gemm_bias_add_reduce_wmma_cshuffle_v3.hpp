// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_reduce.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
template <typename GridwiseGemm,
          typename ReduceTrait,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_gemm_bias_add_reduce_wmma_cshuffle_v3(
        typename GridwiseGemm::Argument karg,
        typename ReduceTrait::ReducePtrsGlobal_ p_reduces_grid,
        const typename ReduceTrait::ReduceInElementwiseOperations_ reduce_in_element_ops,
        const typename ReduceTrait::ReduceAccElementwiseOperations_ reduce_out_element_ops,
        const typename ReduceTrait::D0ElementwiseOperation_ d0_element_op)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using e_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(EGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<e_data_type, ck::half_t> ||
                    std::is_same_v<e_data_type, ck::bhalf_t>)))
    {
#endif
        using EpilogueType = typename GridwiseGemm::template EpilogueReduceCShuffle<ReduceTrait>;
        constexpr index_t LDS_size =
            GridwiseGemm::template GetSharedMemoryNumberOfByte<EpilogueType>();
        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        auto epilogue_args = EpilogueType(
            p_reduces_grid, reduce_in_element_ops, reduce_out_element_ops, karg.M, d0_element_op);

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_shared, splitk_batch_offset, karg, epilogue_args);
#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
    ignore = p_reduces_grid;
    ignore = reduce_in_element_ops;
    ignore = reduce_out_element_ops;
    ignore = d0_element_op;
#endif
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename BiasDataType,
          typename D0DataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename ReduceAccDataType, // Reduce
          typename ReducePtrsGlobal,  // Reduce
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename D0ElementwiseOperation,
          typename ReduceOperations,                // Reduce
          typename ReduceInElementwiseOperations,   // Reduce
          typename ReduceAccElementwiseOperations,  // Reduce
          typename ReduceGlobalMemoryDataOperation, // Reduce
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector,
          typename CReduceThreadClusterLengths_MPerBlock_NPerBlock,            // Reduce
          index_t CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,    // Reduce
          index_t CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock, // Reduce
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceGemmBiasAddReduce_Wmma_CShuffleV3
    : public DeviceGemmReduce<1, ReduceOperations::Size()>
{
    using CDEShuffleBlockTransferScalarPerVectors = Sequence<CShuffleBlockTransferScalarPerVector,
                                                             CShuffleBlockTransferScalarPerVector,
                                                             CShuffleBlockTransferScalarPerVector>;

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        Tuple<ELayout, ELayout>,
        ELayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        CShuffleDataType,
        Tuple<BiasDataType, D0DataType>,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
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
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        PermuteA,
        PermuteB>;

    using ReduceTrait = ReduceTrait_<ReduceAccDataType,
                                     ReducePtrsGlobal,
                                     D0ElementwiseOperation,
                                     ReduceOperations,
                                     ReduceInElementwiseOperations,
                                     ReduceAccElementwiseOperations,
                                     ReduceGlobalMemoryDataOperation,
                                     CReduceThreadClusterLengths_MPerBlock_NPerBlock,
                                     CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
                                     CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 EDataType* p_e_grid,
                 const BiasDataType* p_bias_grid,
                 const D0DataType* p_d0_grid,
                 ReducePtrsGlobal p_reduces_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t StrideC1,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 D0ElementwiseOperation d0_element_op,
                 ReduceInElementwiseOperations reduce_in_element_ops,
                 ReduceAccElementwiseOperations reduce_out_element_ops)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_e_grid_{p_e_grid},
              p_bias_grid_{p_bias_grid},
              p_d0_grid_{p_d0_grid},
              p_reduces_grid_{p_reduces_grid},
              MRaw_{MRaw},
              NRaw_{NRaw},
              KRaw_{KRaw},
              StrideA_{StrideA},
              StrideB_{StrideB},
              StrideC_{StrideC},
              StrideC1_{StrideC1},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              d0_element_op_{d0_element_op},
              reduce_in_element_ops_{reduce_in_element_ops},
              reduce_out_element_ops_{reduce_out_element_ops}
        {
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        EDataType* p_e_grid_;
        const BiasDataType* p_bias_grid_;
        const D0DataType* p_d0_grid_;
        ReducePtrsGlobal p_reduces_grid_;
        index_t MRaw_;
        index_t NRaw_;
        index_t KRaw_;
        index_t StrideA_;
        index_t StrideB_;
        index_t StrideC_;
        index_t StrideC1_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        D0ElementwiseOperation d0_element_op_;
        ReduceInElementwiseOperations reduce_in_element_ops_;
        ReduceAccElementwiseOperations reduce_out_element_ops_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            typename GridwiseGemm::Argument gemm_arg{
                std::array<const void*, 1>{arg.p_a_grid_},
                std::array<const void*, 1>{arg.p_b_grid_},
                std::array<const void*, 2>{arg.p_bias_grid_, arg.p_d0_grid_},
                static_cast<EDataType*>(arg.p_e_grid_),
                arg.MRaw_,
                arg.NRaw_,
                arg.KRaw_,
                std::array<index_t, 1>{arg.StrideA_},     // StrideAs
                std::array<index_t, 1>{arg.StrideB_},     // StrideBs
                std::array<index_t, 2>{0, arg.StrideC1_}, // StrideDs
                arg.StrideC_,                             // StrideE
                1,                                        // kbatch
                arg.a_element_op_,
                arg.b_element_op_,
                arg.c_element_op_};

            if(stream_config.log_level_ > 0)
            {
                gemm_arg.Print();
                GridwiseGemm::BlockwiseGemmPipe::HotLoopInstList::Print();
            }

            if(!GridwiseGemm::CheckValidity(gemm_arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.MRaw_, arg.NRaw_, 1);

            float ave_time = 0;

            index_t K_split = (arg.KRaw_ + KPerBlock - 1) / KPerBlock * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);
            TailNumber TailNum               = GridwiseGemm::CalculateKBlockLoopTailNum(arg.KRaw_);

            const auto Run = [&](const auto& kernel) {
                // Note: cache flushing not supported

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   gemm_arg,
                                                   arg.p_reduces_grid_,
                                                   arg.reduce_in_element_ops_,
                                                   arg.reduce_out_element_ops_,
                                                   arg.d0_element_op_);
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

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(TailNum == TailNumber::Full)
                    {
                        const auto kernel = kernel_gemm_bias_add_reduce_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ReduceTrait,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        throw std::runtime_error("wrong! Invalid pipeline setting");
                    }
                }
            }
            else
            {
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(TailNum == TailNumber::Full)
                    {
                        const auto kernel = kernel_gemm_bias_add_reduce_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ReduceTrait,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        throw std::runtime_error("wrong! Invalid pipeline v1 setting");
                    }
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(TailNum == TailNumber::Even)
                    {
                        const auto kernel = kernel_gemm_bias_add_reduce_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ReduceTrait,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            TailNumber::Even>;
                        Run(kernel);
                    }
                    else if(TailNum == TailNumber::Odd)
                    {
                        const auto kernel = kernel_gemm_bias_add_reduce_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ReduceTrait,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            TailNumber::Odd>;
                        Run(kernel);
                    }
                    else
                    {
                        throw std::runtime_error("wrong! Invalid pipeline v3 setting");
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
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Device implementation supports only gfx11 and gfx12! " << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "FP8 and BF8 not supported on gfx11! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if((arg.KRaw_ % AK1 != 0 || arg.KRaw_ % BK1 != 0) &&
           !(GemmSpec == GemmSpecialization::MKPadding ||
             GemmSpec == GemmSpecialization::NKPadding ||
             GemmSpec == GemmSpecialization::MNKPadding ||
             GemmSpec == GemmSpecialization::KPadding))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Without padding, K must be divisible by AK1 and BK1! " << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        typename GridwiseGemm::Argument gemm_arg{
            std::array<const void*, 1>{arg.p_a_grid_},
            std::array<const void*, 1>{arg.p_b_grid_},
            std::array<const void*, 2>{arg.p_bias_grid_, arg.p_d0_grid_},
            static_cast<EDataType*>(arg.p_e_grid_),
            arg.MRaw_,
            arg.NRaw_,
            arg.KRaw_,
            std::array<index_t, 1>{arg.StrideA_},     // StrideAs
            std::array<index_t, 1>{arg.StrideB_},     // StrideBs
            std::array<index_t, 2>{0, arg.StrideC1_}, // StrideDs
            arg.StrideC_,                             // StrideE
            1,                                        // kbatch
            arg.a_element_op_,
            arg.b_element_op_,
            arg.c_element_op_};

        return GridwiseGemm::CheckValidity(gemm_arg);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static constexpr int NumReduce = ReduceOperations::Size();
    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             const void* p_bias,
                             std::array<const void*, 1> p_ds,
                             void* p_c,
                             std::array<void*, NumReduce> p_reduces,
                             ck::index_t M,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t StrideA,
                             ck::index_t StrideB,
                             ck::index_t StrideC,
                             std::array<ck::index_t, 1> StrideDs,
                             std::array<void*, 3> gemm_element_ops,
                             std::array<void*, 1> d_element_ops,
                             std::array<void*, NumReduce> reduce_in_element_op,
                             std::array<void*, NumReduce> reduce_out_element_op)
    {
        ReducePtrsGlobal reduce_tuple = generate_tuple(
            [&](auto I) {
                auto tmp = ReducePtrsGlobal{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return static_cast<T*>(p_reduces[I]);
            },
            Number<NumReduce>{});

        ReduceInElementwiseOperations reduce_in_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceInElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_in_element_op[I]));
            },
            Number<NumReduce>{});
        ReduceAccElementwiseOperations reduce_out_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceAccElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_out_element_op[I]));
            },
            Number<NumReduce>{});

        AElementwiseOperation a_element_op =
            *(static_cast<AElementwiseOperation*>(gemm_element_ops[0]));
        BElementwiseOperation b_element_op =
            *(static_cast<BElementwiseOperation*>(gemm_element_ops[1]));
        CElementwiseOperation c_element_op =
            *(static_cast<CElementwiseOperation*>(gemm_element_ops[2]));
        D0ElementwiseOperation d_element_op =
            *(static_cast<D0ElementwiseOperation*>(d_element_ops[0]));

        return Argument{static_cast<const ADataType*>(p_a),
                        static_cast<const BDataType*>(p_b),
                        static_cast<EDataType*>(p_c),
                        static_cast<const BiasDataType*>(p_bias),
                        static_cast<const D0DataType*>(p_ds[0]),
                        reduce_tuple,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        StrideDs[0],
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        d_element_op,
                        reduce_in_element_ops,
                        reduce_out_element_ops};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const void* p_bias,
                        std::array<const void*, 1> p_ds,
                        void* p_c,
                        std::array<void*, NumReduce> p_reduces,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        std::array<ck::index_t, 1> StrideDs,
                        std::array<void*, 3> gemm_element_ops,
                        std::array<void*, 1> d_element_ops,
                        std::array<void*, NumReduce> reduce_in_element_op,
                        std::array<void*, NumReduce> reduce_out_element_op,
                        index_t /* KBatch */ = 1) override
    {
        ReducePtrsGlobal reduce_tuple = generate_tuple(
            [&](auto I) {
                auto tmp = ReducePtrsGlobal{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return static_cast<T*>(p_reduces[I]);
            },
            Number<NumReduce>{});

        ReduceInElementwiseOperations reduce_in_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceInElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_in_element_op[I]));
            },
            Number<NumReduce>{});
        ReduceAccElementwiseOperations reduce_out_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceAccElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_out_element_op[I]));
            },
            Number<NumReduce>{});

        AElementwiseOperation a_element_op =
            *(static_cast<AElementwiseOperation*>(gemm_element_ops[0]));
        BElementwiseOperation b_element_op =
            *(static_cast<BElementwiseOperation*>(gemm_element_ops[1]));
        CElementwiseOperation c_element_op =
            *(static_cast<CElementwiseOperation*>(gemm_element_ops[2]));
        D0ElementwiseOperation d_element_op =
            *(static_cast<D0ElementwiseOperation*>(d_element_ops[0]));

        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<EDataType*>(p_c),
                                          static_cast<const BiasDataType*>(p_bias),
                                          static_cast<const D0DataType*>(p_ds[0]),
                                          reduce_tuple,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          StrideDs[0],
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          d_element_op,
                                          reduce_in_element_ops,
                                          reduce_out_element_ops);
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

        // clang-format off
        str << "DeviceGemmBiasAddReduce_Wmma_CShuffleV3"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerWmma << ", "
            << NPerWmma << ", "
            << MRepeat << ", "
            << NRepeat << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMRepeatPerShuffle << ", "
            << CShuffleNRepeatPerShuffle
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
