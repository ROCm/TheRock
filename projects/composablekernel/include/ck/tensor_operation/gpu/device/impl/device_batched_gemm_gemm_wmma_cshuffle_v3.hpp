// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/operator_transform/transform_contraction_to_gemm_arraybase.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DeviceOp, typename GridwiseOp, bool HasMainKBlockLoop, TailNumber TailNum>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_batched_gemm_gemm_wmma_cshuffle_v3(typename DeviceOp::RawArg arg)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__))

    __shared__ char p_shared[GridwiseOp::GetSharedMemoryNumberOfByte()];
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / arg.batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset =
        __builtin_amdgcn_readfirstlane((arg.compute_base_ptr_of_batch.GetABasePtr(g_idx)));
    const long_index_t b0_batch_offset =
        __builtin_amdgcn_readfirstlane((arg.compute_base_ptr_of_batch.GetB0BasePtr(g_idx)));
    const long_index_t b1_batch_offset =
        __builtin_amdgcn_readfirstlane((arg.compute_base_ptr_of_batch.GetB1BasePtr(g_idx)));
    const long_index_t c_batch_offset =
        __builtin_amdgcn_readfirstlane((arg.compute_base_ptr_of_batch.GetCBasePtr(g_idx)));

    GridwiseOp::template Run<HasMainKBlockLoop, TailNum>(
        arg.p_a_grid + a_batch_offset,
        arg.p_b0_grid + b0_batch_offset,
        arg.p_b1_grid + b1_batch_offset,
        arg.p_c_grid + c_batch_offset,
        p_shared,
        arg.a_grid_desc,
        arg.b0_grid_desc,
        arg.b1_grid_desc,
        arg.c_grid_desc_mblock_mperblock_nblock_nperblock,
        arg.a_element_op,
        arg.b0_element_op,
        arg.acc_element_op,
        arg.b1_element_op,
        arg.c_element_op,
        arg.block_2_ctile_map);
#else
    ignore = arg;
#endif // (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__)
}

// Computes C = A  * B0 * B1
//         MN = MK * KL * LN
//              ^^^^^^ (Acc0)
//              ^^^^^^^^^^^ (Acc1)
template <typename ALayout,
          typename B0layout,
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t LPerBlock,     // Gemm0NPerBlock
          ck::index_t KPerBlock,     // Gemm0KPerBlock
          ck::index_t NPerBlock,     // Gemm1NPerBlock
          ck::index_t LTilePerBlock, // Gemm1KPerBlock
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t L1,       // B1K1
          ck::index_t MPerWmma, // Gemm0/1 MPerWmma
          ck::index_t LPerWmma, // Gemm0/1 NPerWmma
          ck::index_t MRepeat,  // Gemm0/1 MWmmaPerWave or Mrepeat
          ck::index_t LRepeat,  // Gemm0 NWmmaPerWave or Nrepeat
          ck::index_t NRepeat,  // Gemm1 NWmmaPerWave or Nrepeat
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename B0BlockTransferThreadClusterLengths_K0_L_K1,
          typename B0BlockTransferThreadClusterArrangeOrder,
          typename B0BlockTransferSrcAccessOrder,
          ck::index_t B0BlockTransferSrcVectorDim,
          ck::index_t B0BlockTransferSrcScalarPerVector,
          ck::index_t B0BlockTransferDstScalarPerVector_K1,
          bool B0BlockLdsAddExtraL,
          typename B1BlockTransferThreadClusterLengths_L0_N_L1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          ck::index_t B1BlockTransferSrcVectorDim,
          ck::index_t B1BlockTransferSrcScalarPerVector,
          ck::index_t B1BlockTransferDstScalarPerVector_L1,
          bool B1BlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1>
struct DeviceBatchedGemmGemm_Wmma_CShuffleV3 : public DeviceBatchedGemmGemm<ALayout,
                                                                            B0layout,
                                                                            B1Layout,
                                                                            CLayout,
                                                                            ADataType,
                                                                            B0DataType,
                                                                            B1DataType,
                                                                            CDataType,
                                                                            AElementwiseOperation,
                                                                            B0ElementwiseOperation,
                                                                            AccElementwiseOperation,
                                                                            B1ElementwiseOperation,
                                                                            CElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmGemm_Wmma_CShuffleV3;

    static constexpr auto I0 = Number<0>{};

    // To match XDL implementation NPerWmma (A.k.a Gemm1 NPerWmma) is set equal
    // to LPerWmma (A.k.a Gemm0 NPerWmma).
    static constexpr index_t NPerWmma = LPerWmma;

    // TODO: Now that we are no longer using NumDim or TensorSpec, we can probably use a simpler
    // Transform operator or just not use one at all.
    using Transform = TransformBatchedContractionContractionToBatchedGemmGemm_Wmma<
        Sequence<1, 1, 1, 1, 1>,
        Sequence<MPerBlock, LPerBlock, KPerBlock, NPerBlock>,
        GemmSpec,
        TensorSpecialization::Default,  // ASpec
        TensorSpecialization::Default,  // B0Spec
        TensorSpecialization::Default,  // B1Spec
        TensorSpecialization::Default>; // CSpec

    __host__ __device__ static auto
    MakeAGridDescriptor(const std::array<index_t, 3>& a_g_m_k_lengths_vec,
                        const std::array<index_t, 3>& a_g_m_k_strides_vec)
    {
        return Transform::MakeAGridDescriptor_AK0_M_AK1(
            Transform::MakeAGridDescriptor_M_K(a_g_m_k_lengths_vec, a_g_m_k_strides_vec),
            Number<AK1>{});
    }

    __host__ __device__ static auto
    MakeB0GridDescriptor(const std::array<index_t, 3>& b0_g_l_k_lengths_vec,
                         const std::array<index_t, 3>& b0_g_l_k_strides_vec)
    {
        return Transform::MakeB0GridDescriptor_BK0_N_BK1(
            Transform::MakeB0GridDescriptor_N_K(b0_g_l_k_lengths_vec, b0_g_l_k_strides_vec),
            Number<BK1>{});
    }

    __host__ __device__ static auto
    MakeB1GridDescriptor(const std::array<index_t, 3>& b1_g_n_l_lengths_vec,
                         const std::array<index_t, 3>& b1_g_n_l_strides_vec)
    {
        return Transform::MakeB1GridDescriptor_BK0_N_BK1(
            Transform::MakeB1GridDescriptor_N_K(b1_g_n_l_lengths_vec, b1_g_n_l_strides_vec),
            Number<L1>{});
    }

    using AGridDesc     = decltype(MakeAGridDescriptor({}, {}));
    using B0GridDesc    = decltype(MakeB0GridDescriptor({}, {}));
    using B1GridDesc    = decltype(MakeB1GridDescriptor({}, {}));
    using CGridDesc_M_N = decltype(Transform::MakeCGridDescriptor_M_N({}, {}));

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(index_t BatchStrideA,
                                     index_t BatchStrideB0,
                                     index_t BatchStrideB1,
                                     index_t BatchStrideC)
            : BatchStrideA_(BatchStrideA),
              BatchStrideB0_(BatchStrideB0),
              BatchStrideB1_(BatchStrideB1),
              BatchStrideC_(BatchStrideC)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetB0BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB0_);
        }

        __host__ __device__ constexpr long_index_t GetB1BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB1_);
        }

        __host__ __device__ constexpr long_index_t GetCBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC_);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB0_;
        index_t BatchStrideB1_;
        index_t BatchStrideC_;
    };

    // GridwiseOp
    using GridwiseOp = GridwiseBatchedGemmGemm_wmma_cshuffle_v3<
        // DataType Family
        ADataType,
        B0DataType,
        AccDataType, // Acc0DataType
        B1DataType,
        AccDataType, // Acc1DataType
        CShuffleDataType,
        CDataType,
        // ElementwiseOp Family
        AElementwiseOperation,
        B0ElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        // InMemory Data Descriptor
        AGridDesc,
        B0GridDesc,
        B1GridDesc,
        CGridDesc_M_N,
        // Tiling Family
        MPerBlock,
        LPerBlock,
        KPerBlock,
        AK1,
        BK1,
        NPerBlock,
        LTilePerBlock,
        L1,
        MPerWmma,
        LPerWmma,
        NPerWmma,
        MRepeat,
        LRepeat,
        NRepeat,
        // ThreadCluster Family
        BlockSize,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        true,
        ABlockLdsAddExtraM,
        B0BlockTransferThreadClusterLengths_K0_L_K1,
        B0BlockTransferThreadClusterArrangeOrder,
        B0BlockTransferSrcAccessOrder,
        B0BlockTransferSrcVectorDim,
        B0BlockTransferSrcScalarPerVector,
        B0BlockTransferDstScalarPerVector_K1,
        true,
        B0BlockLdsAddExtraL,
        B1BlockTransferThreadClusterLengths_L0_N_L1,
        B1BlockTransferThreadClusterArrangeOrder,
        B1BlockTransferSrcAccessOrder,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        B1BlockTransferDstScalarPerVector_L1,
        false,
        B1BlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        Transform::matrix_padder.PadN,
        BlkGemmPipeSched,
        BlkGemmPipelineVer>;

    struct RawArg : public BaseArgument
    {
        using arr3 = std::array<ck::index_t, 3>;

        RawArg(const ADataType* p_a_grid_,
               const B0DataType* p_b0_grid_,
               const B1DataType* p_b1_grid_,
               CDataType* p_c_grid_,
               index_t M_,
               index_t N_,
               index_t K_,
               index_t O_,
               index_t Batch,
               index_t StrideA,
               index_t StrideB0,
               index_t StrideB1,
               index_t StrideC,
               index_t BatchStrideA,
               index_t BatchStrideB0,
               index_t BatchStrideB1,
               index_t BatchStrideC,
               AElementwiseOperation a_element_op_,
               B0ElementwiseOperation b0_element_op_,
               AccElementwiseOperation acc_element_op_,
               B1ElementwiseOperation b1_element_op_,
               CElementwiseOperation c_element_op_)
            : p_a_grid{p_a_grid_},
              p_b0_grid{p_b0_grid_},
              p_b1_grid{p_b1_grid_},
              p_c_grid{p_c_grid_},
              M{M_},
              N{N_},
              K{K_},
              O{O_},
              batch_count{Batch},
              a_element_op{a_element_op_},
              b0_element_op{b0_element_op_},
              acc_element_op{acc_element_op_},
              b1_element_op{b1_element_op_},
              c_element_op{c_element_op_},
              compute_base_ptr_of_batch{BatchStrideA, BatchStrideB0, BatchStrideB1, BatchStrideC}
        {

            a_g_m_k_lengths = arr3{batch_count, M, K};
            a_g_m_k_strides = arr3{BatchStrideA, StrideA, 1}; // A layout [batch_count, M, K]

            b0_g_n_k_lengths = arr3{batch_count, N, K};
            b0_g_n_k_strides = arr3{BatchStrideB0, StrideB0, 1}; // B0 layout [batch_count, N, K]

            b1_g_o_n_lengths = arr3{batch_count, O, N};
            b1_g_o_n_strides =
                is_same_v<B1Layout, tensor_layout::gemm::RowMajor>
                    ? arr3{BatchStrideB1, 1, StrideB1}  // B1 layout [batch_count, N, O]
                    : arr3{BatchStrideB1, StrideB1, 1}; // B1 layout [batch_count, O, N]

            c_g_m_o_lengths = arr3{batch_count, M, O};
            c_g_m_o_strides = arr3{BatchStrideC, StrideC, 1}; // C layout [batch_count, M, O]

            a_grid_desc     = MakeAGridDescriptor(a_g_m_k_lengths, a_g_m_k_strides);
            b0_grid_desc    = MakeB0GridDescriptor(b0_g_n_k_lengths, b0_g_n_k_strides);
            b1_grid_desc    = MakeB1GridDescriptor(b1_g_o_n_lengths, b1_g_o_n_strides);
            c_grid_desc_m_n = Transform::MakeCGridDescriptor_M_N(c_g_m_o_lengths, c_g_m_o_strides);
            c_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseOp::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n);
            block_2_ctile_map = GridwiseOp::MakeDefaultBlock2CTileMap(c_grid_desc_m_n, 1, 1);
        }
        // Pointers
        const ADataType* p_a_grid;
        const B0DataType* p_b0_grid;
        const B1DataType* p_b1_grid;
        CDataType* p_c_grid;

        // Raw Problem Size
        index_t M;
        index_t N;
        index_t K;
        index_t O;
        index_t batch_count;

        arr3 a_g_m_k_lengths;
        arr3 a_g_m_k_strides;
        arr3 b0_g_n_k_lengths;
        arr3 b0_g_n_k_strides;
        arr3 b1_g_o_n_lengths;
        arr3 b1_g_o_n_strides;
        arr3 c_g_m_o_lengths;
        arr3 c_g_m_o_strides;

        AElementwiseOperation a_element_op;
        B0ElementwiseOperation b0_element_op;
        AccElementwiseOperation acc_element_op;
        B1ElementwiseOperation b1_element_op;
        CElementwiseOperation c_element_op;

        // Grid descriptors and other mem calculators
        AGridDesc a_grid_desc;
        B0GridDesc b0_grid_desc;
        B1GridDesc b1_grid_desc;
        CGridDesc_M_N c_grid_desc_m_n;
        typename GridwiseOp::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock;

        typename GridwiseOp::DefaultBlock2CTileMap block_2_ctile_map;

        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch;
    };

    static bool IsSupportedArgument([[maybe_unused]] const RawArg& arg)
    {
        // Print lambda with env check and printf() style formmating.
        const char* curFunc = __func__;
        auto print          = [&curFunc](const char* format, ...) -> void {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#endif
                va_list args;
                va_start(args, format);
                std::vfprintf(stdout, format, args);
                va_end(args);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
                std::cout << "In file: " << __FILE__ << ", function: " << curFunc << "\n";
            }
        };

        if(!(ck::is_gfx11_supported() || ck::is_gfx12_supported()))
        {
            print("DeviceOp: Arch err\n");
            return false;
        }

        if constexpr(std::is_same_v<ADataType, f8_t> || std::is_same_v<ADataType, bf8_t> ||
                     std::is_same_v<B0DataType, f8_t> || std::is_same_v<B0DataType, bf8_t> ||
                     std::is_same_v<B1DataType, f8_t> || std::is_same_v<B1DataType, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                print("DeviceOp: gfx 11 does not support fp8\n");
                return false;
            }
        }

        if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
        {
            print("DeviceOp: Acc0 Type err\n");
            return false;
        }

        if constexpr(!(is_same_v<ALayout, tensor_layout::gemm::RowMajor>))
        {
            print("DeviceOp: A layout must be Row\n");
            return false;
        }

        if constexpr(!(is_same_v<B0layout, tensor_layout::gemm::ColumnMajor>))
        {
            print("DeviceOp: B layout must be Column\n");
            return false;
        }

        if constexpr(!(is_same_v<B1Layout, tensor_layout::gemm::RowMajor> ||
                       is_same_v<B1Layout, tensor_layout::gemm::ColumnMajor>))
        {
            print("DeviceOp: B1 layout must be Column or Row\n");
            return false;
        }

        if constexpr(!(is_same_v<CLayout, tensor_layout::gemm::RowMajor>))
        {
            print("DeviceOp: C layout must be Row\n");
            return false;
        }

        // Other padding modes have not been tested and do not get checked individually.
        if constexpr(GemmSpec != GemmSpecialization::Default &&
                     GemmSpec != GemmSpecialization::MNKOPadding)
        {
            print("Padding mode must be default or MNKO\n");
            return false;
        }

        // Per wmma dimensions not equal to 16 are very untested.
        if constexpr(MPerWmma != 16 || LPerWmma != 16 || NPerWmma != 16)
        {
            print("M, L, N per Wmma must be 16\n");
            return false;
        }

        if(!GridwiseOp::CheckValidity(arg.a_grid_desc,
                                      arg.b0_grid_desc,
                                      arg.b1_grid_desc,
                                      arg.c_grid_desc_m_n,
                                      arg.block_2_ctile_map))
        {
            return false;
        }

        // Check scalar per vector requirement
        const auto a_extent_lowest  = ABlockTransferSrcVectorDim == 2 ? arg.K : arg.M;
        const auto b0_extent_lowest = B0BlockTransferSrcVectorDim == 2 ? arg.K : arg.N;
        const auto b1_extent_lowest = B1BlockTransferSrcVectorDim == 2 ? arg.N : arg.O;
        const auto c_extent_lowest  = arg.O;

        if(!(a_extent_lowest % ABlockTransferSrcScalarPerVector == 0 &&
             b0_extent_lowest % B0BlockTransferSrcScalarPerVector == 0 &&
             b1_extent_lowest % B1BlockTransferSrcScalarPerVector == 0 &&
             c_extent_lowest % CShuffleBlockTransferScalarPerVector_NPerBlock == 0))
        {
            print("DeviceOp: Data Transfer Vector scalar err\n");
            return false;
        }

        // Check vector load/store requirement
        const auto a_stride_lowest =
            ABlockTransferSrcVectorDim == 2 ? arg.a_g_m_k_strides[2] : arg.a_g_m_k_strides[1];
        const auto b0_stride_lowest =
            B0BlockTransferSrcVectorDim == 2 ? arg.b0_g_n_k_strides[2] : arg.b0_g_n_k_strides[1];
        const auto b1_stride_lowest =
            B1BlockTransferSrcVectorDim == 2 ? arg.b1_g_o_n_strides[2] : arg.b1_g_o_n_strides[1];
        const auto c_stride_lowest = arg.c_g_m_o_strides[2];

        if(!(a_stride_lowest == 1 || b0_stride_lowest == 1 || b1_stride_lowest == 1 ||
             c_stride_lowest == 1))
        {
            print("DeviceOp: Data Vectorize transfer err\n");
            return false;
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MNKOPadding))
        {
            return false;
        }

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const RawArg*>(p_arg));
    }

    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::RawArg;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto M0 = math::integer_divide_ceil(arg.M, MPerBlock);
            const auto N0 = math::integer_divide_ceil(arg.O, NPerBlock);

            const index_t grid_size = arg.batch_count * M0 * N0;

            auto launch_kernel = [&](auto has_main_k_block_loop, auto tail_number) {
                constexpr bool has_loop = decltype(has_main_k_block_loop)::value;
                constexpr TailNumber tn = tail_number;

                const auto kernel =
                    kernel_batched_gemm_gemm_wmma_cshuffle_v3<DeviceOp, GridwiseOp, has_loop, tn>;

                return launch_and_time_kernel(
                    stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, arg);
            };

            bool HasMainKBlockLoop = GridwiseOp::CalculateHasMainKBlockLoop(arg.K);
            TailNumber TailNum     = GridwiseOp::CalculateKBlockLoopTailNum(arg.K);

            if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
            {
                if(HasMainKBlockLoop && TailNum == TailNumber::Full)
                {
                    return launch_kernel(std::integral_constant<bool, true>{},
                                         std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else if(!HasMainKBlockLoop && TailNum == TailNumber::Full)
                {
                    return launch_kernel(std::integral_constant<bool, false>{},
                                         std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else
                {
                    printf("Invalid HasMainKBlockLoop and TailNum combination for V1!\n");
                    return 0.0f;
                }
            }
            else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
            {
                if(HasMainKBlockLoop && TailNum == TailNumber::Full)
                {
                    return launch_kernel(std::integral_constant<bool, true>{},
                                         std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else if(!HasMainKBlockLoop && TailNum == TailNumber::Even)
                {
                    return launch_kernel(std::integral_constant<bool, false>{},
                                         std::integral_constant<TailNumber, TailNumber::Even>{});
                }
                else if(!HasMainKBlockLoop && TailNum == TailNumber::Odd)
                {
                    return launch_kernel(std::integral_constant<bool, false>{},
                                         std::integral_constant<TailNumber, TailNumber::Odd>{});
                }
                else
                {
                    printf("Invalid HasMainKBlockLoop and TailNum combination for V3!\n");
                    return 0.0f;
                }
            }
            else
            {
                printf("Invalid pipeline version!\n");
                return 0.0f;
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b0,
                                                      const void* p_b1,
                                                      void* p_c,
                                                      ck::index_t M,
                                                      ck::index_t N,
                                                      ck::index_t K,
                                                      ck::index_t O,
                                                      ck::index_t Batch,
                                                      ck::index_t StrideA,
                                                      ck::index_t StrideB0,
                                                      ck::index_t StrideB1,
                                                      ck::index_t StrideC,
                                                      ck::index_t BatchStrideA,
                                                      ck::index_t BatchStrideB0,
                                                      ck::index_t BatchStrideB1,
                                                      ck::index_t BatchStrideC,
                                                      AElementwiseOperation a_element_op,
                                                      B0ElementwiseOperation b0_element_op,
                                                      AccElementwiseOperation acc_element_op,
                                                      B1ElementwiseOperation b1_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<RawArg>(static_cast<const ADataType*>(p_a),
                                        static_cast<const B0DataType*>(p_b0),
                                        static_cast<const B1DataType*>(p_b1),
                                        static_cast<CDataType*>(p_c),
                                        M,
                                        N,
                                        K,
                                        O,
                                        Batch,
                                        StrideA,
                                        StrideB0,
                                        StrideB1,
                                        StrideC,
                                        BatchStrideA,
                                        BatchStrideB0,
                                        BatchStrideB1,
                                        BatchStrideC,
                                        a_element_op,
                                        b0_element_op,
                                        acc_element_op,
                                        b1_element_op,
                                        c_element_op);
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    template <typename T>
    static constexpr const char* DataTypeToString()
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return "fp32";
        }
        else if constexpr(std::is_same_v<T, ck::half_t>)
        {
            return "fp16";
        }
        else if constexpr(std::is_same_v<T, ck::bhalf_t>)
        {
            return "bf16";
        }
        else if constexpr(std::is_same_v<T, ck::f8_t>)
        {
            return "fp8";
        }
        else if constexpr(std::is_same_v<T, ck::bf8_t>)
        {
            return "bf8";
        }
        else if constexpr(std::is_same_v<T, int32_t>)
        {
            return "int32";
        }
        else if constexpr(std::is_same_v<T, int8_t>)
        {
            return "int8";
        }
        else if constexpr(std::is_same_v<T, ck::int4_t>)
        {
            return "int4";
        }
        else
        {
            return "unknown";
        }
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
        str << "DeviceBatchedGemmGemm_Wmma_CShuffleV3"
            << "<"
            << ALayout::name[0]
            << B0layout::name[0]
            << B1Layout::name[0]
            << CLayout::name[0] << ", "
            << "A " << DataTypeToString<ADataType>() << ", "
            << "B0 " << DataTypeToString<B0DataType>() << ", "
            << "B1 " << DataTypeToString<B1DataType>() << ", "
            << "C " << DataTypeToString<CDataType>() << ", "
            << "Acc " << DataTypeToString<AccDataType>() << ", "
            << "Cshuf " << DataTypeToString<CShuffleDataType>() << ", "
            << BlockSize << ", "
            << MPerBlock << ", "
            << LPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << LTilePerBlock << ", "
            << L1 << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">"
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseOp::BlockwiseGemmPipe::PrefetchStages;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
