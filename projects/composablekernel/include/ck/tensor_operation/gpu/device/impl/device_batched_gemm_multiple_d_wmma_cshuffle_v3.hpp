// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multi_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename ComputePtrOffsetOfStridedBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_batched_gemm_multi_d_wmma_cshuffle_v3(
        typename GridwiseGemm::Argument karg, // This works for now but it actually receives a
                                              // DeviceBatchedGemm_Wmma_CShuffleV3::Argument
                                              // argument through implicit conversion to base class!
        const ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using EDataType = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(EGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<EDataType, ck::half_t> ||
                    std::is_same_v<EDataType, ck::bhalf_t>)))
    {
#endif
        // The normal approach to batching would be to increase the grid size by just stretching out
        // the grid Z dimension (which is the outermost dimension), but this depends on lower level
        // functions not directly using the Z dimension for other calculations. As it turns out, k
        // batching does rely directly on blockIdx.Z through SplitKBatchOffset. Therefore, for now
        // we will use the grid Y dimension for batching. This may be a bit fragile.
        const index_t g_idx = amd_wave_read_first_lane(blockIdx.y);

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);
        const long_index_t c_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx));

        constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
            typename GridwiseGemm::EpilogueCShuffle>();
        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        static_for<0, GridwiseGemm::NumATensor, 1>{}(
            [&](auto i) { splitk_batch_offset.a_k_split_offset[i] += a_batch_offset; });

        static_for<0, GridwiseGemm::NumBTensor, 1>{}(
            [&](auto i) { splitk_batch_offset.b_k_split_offset[i] += b_batch_offset; });

        splitk_batch_offset.c_reduce_offset += c_batch_offset;

        // populate pointer, desc for Ds
        static_for<0, GridwiseGemm::NumDTensor, 1>{}([&](auto i) {
            // D pointer
            karg.p_ds_grid(i) = karg.p_ds_grid(i) + ds_batch_offset[i];
        });

        auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_shared, splitk_batch_offset, karg, epilogue_args);
#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
    ignore = compute_ptr_offset_of_batch;
#endif
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
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
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = ADataType,
          typename ComputeTypeB                       = BDataType>
struct DeviceBatchedGemmMultiD_Wmma_CShuffleV3
    : public DeviceBatchedGemmV2MultiD<ALayout,
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
    using CDEShuffleBlockTransferScalarPerVectors_ = CDEShuffleBlockTransferScalarPerVectors;
    using CDataType_                               = EDataType;
    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
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
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,
        false>;

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch() = default;
        ComputePtrOffsetOfStridedBatch(
            index_t BatchStrideA,
            index_t BatchStrideB,
            std::array<ck::index_t, GridwiseGemm::NumDTensor> BatchStrideDs,
            index_t BatchStrideC)
            : BatchStrideA_(BatchStrideA),
              BatchStrideB_(BatchStrideB),
              BatchStrideDs_(BatchStrideDs),
              BatchStrideC_(BatchStrideC)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(BatchStrideA_) * g_idx;
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(BatchStrideB_) * g_idx;
        }

        __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
        {
            std::array<long_index_t, GridwiseGemm::NumDTensor> ds_offset_;

            static_for<0, GridwiseGemm::NumDTensor, 1>{}([&](auto i) {
                ds_offset_[i] = static_cast<long_index_t>(BatchStrideDs_[i]) * g_idx;
            });

            return ds_offset_;
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(BatchStrideC_) * g_idx;
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        std::array<ck::index_t, GridwiseGemm::NumDTensor> BatchStrideDs_;
        index_t BatchStrideC_;
    };

    struct Argument : public GridwiseGemm::Argument
    {
        index_t Batch;
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch;

        Argument() = default;
        Argument(const ADataType* p_a_grid_,
                 const BDataType* p_b_grid_,
                 std::array<const void*, GridwiseGemm::NumDTensor> p_ds_grid_,
                 EDataType* p_e_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 std::array<index_t, GridwiseGemm::NumDTensor> StrideDs_,
                 index_t StrideE_,
                 index_t BatchStrideA_,
                 index_t BatchStrideB_,
                 const std::array<ck::index_t, GridwiseGemm::NumDTensor>& BatchStrideDs_,
                 index_t BatchStrideE_,
                 index_t Batch_,
                 AElementwiseOperation a_element_op_,
                 BElementwiseOperation b_element_op_,
                 CDEElementwiseOperation cde_element_op_,
                 index_t KBatch_)
            : GridwiseGemm::Argument{std::array<const void*, 1>{p_a_grid_},
                                     std::array<const void*, 1>{p_b_grid_},
                                     p_ds_grid_,
                                     p_e_grid_,
                                     M_,
                                     N_,
                                     K_,
                                     std::array<index_t, 1>{StrideA_},
                                     std::array<index_t, 1>{StrideB_},
                                     StrideDs_,
                                     StrideE_,
                                     KBatch_,
                                     a_element_op_,
                                     b_element_op_,
                                     cde_element_op_,
                                     false},
              Batch{Batch_},
              compute_ptr_offset_of_batch{
                  BatchStrideA_, BatchStrideB_, BatchStrideDs_, BatchStrideE_}
        {
        }
        template <typename EType>
        void SetEPointer(void* ptr)
        {
            this->p_e_grid = static_cast<EType*>(ptr);
        }
    };

    struct ActiveWorkgroupsPerCU
    {
        ActiveWorkgroupsPerCU()
        {
            constexpr int dynamic_smem_size = 0;
            int max_occupancy               = 0;

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

            hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_occupancy,
                kernel_batched_gemm_multi_d_wmma_cshuffle_v3<GridwiseGemm,
                                                             ComputePtrOffsetOfStridedBatch,
                                                             true,
                                                             InMemoryDataOperationEnum::AtomicAdd,
                                                             minimum_occupancy>,
                BlockSize,
                dynamic_smem_size));

            max_occupancy_ = std::max(1, max_occupancy);
        }
        int max_occupancy_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
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

            gdy *= arg.Batch;

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

                    // Packed sizes are 1 for all implemented data types but we include it anyway
                    // for future compatibility.
                    std::array<std::size_t, 1> size_as_buffers;
                    size_as_buffers[0] = arg_.Batch *
                                         a_grid_desc_ak0_m_ak1[Number<0>{}].GetElementSpaceSize() *
                                         sizeof(ADataType) / GridwiseGemm::APackedSize;

                    std::array<std::size_t, 1> size_bs_buffers;
                    size_bs_buffers[0] = arg_.Batch *
                                         b_grid_desc_bk0_n_bk1[Number<0>{}].GetElementSpaceSize() *
                                         sizeof(BDataType) / GridwiseGemm::BPackedSize;

                    const auto ds_grid_desc_m_n = GridwiseGemm::MakeDsGridDescriptor_M_N(
                        arg_.M, arg_.MPadded, arg_.N, arg_.NPadded, arg_.StrideDs);

                    std::array<std::size_t, GridwiseGemm::NumDTensor> size_ds_buffers;
                    static_for<0, GridwiseGemm::NumDTensor, 1>{}([&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                        size_ds_buffers[i] =
                            ds_grid_desc_m_n[i].GetElementSpaceSize() * sizeof(DDataType);
                    });
                    ck::utility::RotatingMemWrapperMultiABD<Argument,
                                                            Tuple<ADataType>,
                                                            Tuple<BDataType>,
                                                            DsDataType>
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
                            HIP_CHECK_ERROR(
                                hipMemsetAsync(arg_.p_e_grid,
                                               0,
                                               arg.Batch * arg_.M * arg_.N * sizeof(EDataType),
                                               stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg_,
                        arg_.compute_ptr_offset_of_batch);
                }
                else
                {
                    const auto clear_workspace = [&]() {
                        if(arg.KBatch > 1)
                            HIP_CHECK_ERROR(
                                hipMemsetAsync(arg.p_e_grid,
                                               0,
                                               arg.Batch * arg.M * arg.N * sizeof(EDataType),
                                               stream_config.stream_id_));
                    };

                    ave_time =
                        launch_and_time_kernel_with_preprocess(stream_config,
                                                               clear_workspace,
                                                               kernel,
                                                               dim3(gdx, gdy, gdz),
                                                               dim3(BlockSize),
                                                               0,
                                                               arg,
                                                               arg.compute_ptr_offset_of_batch);
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

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_batched_gemm_multi_d_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ComputePtrOffsetOfStridedBatch,
                            true,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_batched_gemm_multi_d_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ComputePtrOffsetOfStridedBatch,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_batched_gemm_multi_d_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ComputePtrOffsetOfStridedBatch,
                            false,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_batched_gemm_multi_d_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ComputePtrOffsetOfStridedBatch,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
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
                std::cout << "Unsupported: Architecture must be gfx11/gfx12." << std::endl;
            }
            return false;
        }

        if constexpr(std::is_same_v<EDataType, ck::half_t> ||
                     std::is_same_v<EDataType, ck::bhalf_t>)
        {
            if(arg.KBatch > 1 && ck::is_gfx11_supported())
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Unsupported splitK on gfx11." << std::endl;
                }
                // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
                return false;
            }
        }

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Unsupported f8 / bf8 on gfx11." << std::endl;
                }
                return false;
            }
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported K dimension without padding." << std::endl;
            }
            return false;
        }

        return GridwiseGemm::CheckValidity(arg);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, GridwiseGemm::NumDTensor> p_ds,
                             void* p_e,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t Batch,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, GridwiseGemm::NumDTensor> StrideDs,
                             index_t StrideE,
                             index_t BatchStrideA,
                             index_t BatchStrideB,
                             const std::array<ck::index_t, GridwiseGemm::NumDTensor>& BatchStrideDs,
                             index_t BatchStrideE,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op,
                             index_t KBatch = 1)
    {
        return Argument{static_cast<const ADataType*>(p_a),
                        static_cast<const BDataType*>(p_b),
                        p_ds,
                        static_cast<EDataType*>(p_e),
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideDs,
                        StrideE,
                        BatchStrideA,
                        BatchStrideB,
                        BatchStrideDs,
                        BatchStrideE,
                        Batch,
                        a_element_op,
                        b_element_op,
                        cde_element_op,
                        KBatch};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const std::array<const void*, GridwiseGemm::NumDTensor>& p_ds,
                        void* p_e,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t Batch,
                        index_t StrideA,
                        index_t StrideB,
                        const std::array<ck::index_t, GridwiseGemm::NumDTensor>& StrideDs,
                        index_t StrideE,
                        index_t BatchStrideA,
                        index_t BatchStrideB,
                        const std::array<ck::index_t, GridwiseGemm::NumDTensor>& BatchStrideDs,
                        index_t BatchStrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op,
                        index_t KBatch = 1) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          p_ds,
                                          static_cast<EDataType*>(p_e),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideE,
                                          BatchStrideA,
                                          BatchStrideB,
                                          BatchStrideDs,
                                          BatchStrideE,
                                          Batch,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op,
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
        str << "DeviceBatchedGemmMultipleD_Wmma_CShuffleV3"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(ELayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock<<"x"<<NPerBlock<<"x"<<KPerBlock << ", "
            << "WaveTile: "
            << MPerWmma<<"x"<<NPerWmma << ", "
            << "WaveMap: "
            << MRepeat<<"x" << NRepeat<<", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector<<"x"<<BBlockTransferSrcScalarPerVector<<", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm::BlockwiseGemmPipe::PrefetchStages;
        // clang-format on

        return str.str();
    }

    static ck::index_t GetMaxOccupancy()
    {
        static ActiveWorkgroupsPerCU active_workgroups_per_cu;
        return active_workgroups_per_cu.max_occupancy_;
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
