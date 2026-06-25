// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <tuple>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/stream_utility.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp" // stare wywalic
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

///
/// @brief      Entry point kernel for device-wide Grouped GEMM operation.
///
/// @param[in]  gemm_descs_const  The pointer to the array of GEMM descriptor structures.
/// @param[in]  group_count       The number of together processed GEMMs.
///
/// @tparam     GridwiseGemm                The specific GridwiseGEMM algorithm implementation.
/// @tparam     GemmDesc                    The structure holding all necessary descriptors and
///                                         other data needed for grouped gemm calculation and work
///                                         distribution.
/// @tparam     LocalBlock2ETileMap         The structure providing mapping between workgroup ids,
///                                         the data tiles to process and the output tiles.
///
template <typename GridwiseGemm,
          typename GemmDesc,
          GemmSpecialization GemmSpec,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          index_t KPerBlock,
          typename OffsettedBlockToCTileMap,
          typename LocalBlock2ETileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_gemm_multiple_d_xdl(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                       const index_t group_count,
                                       const AElementwiseOperation a_element_op,
                                       const BElementwiseOperation b_element_op,
                                       const CDEElementwiseOperation cde_element_op)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        constexpr index_t shared_size = GridwiseGemm::GetSharedMemoryNumberOfByte();
        __shared__ uint8_t p_shared[shared_size];
        __shared__ uint8_t p_shared1[shared_size];

        const auto gemm_desc_ptr = reinterpret_cast<const GemmDesc*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        constexpr auto NumDTensor = DsDataType::Size();
        index_t tile_id           = get_block_1d_id();
        index_t tile_offset       = 0;
        index_t group_id          = -1;
        index_t group_offset      = 0;
        index_t grid_size_grp     = 0;

        index_t gemm_tile_id_start = 0;
        index_t gemm_tile_id_end   = 0;

        index_t M = 0, N = 0, K = 0;

        auto b2c_tile_map = OffsettedBlockToCTileMap(LocalBlock2ETileMap(1, 1), 1, 1);

        do
        {
            // Find corresponding GEMM group for our tile
            while(!(tile_id >= gemm_tile_id_start && tile_id < gemm_tile_id_end) &&
                  group_id < group_count)
            {
                group_offset += grid_size_grp;
                group_id++;

                if(group_id >= group_count)
                    return;

                M = gemm_desc_ptr[group_id].M;
                N = gemm_desc_ptr[group_id].N;
                K = gemm_desc_ptr[group_id].K;

                if(M == 0 || N == 0 || K == 0)
                {
                    grid_size_grp = 0;
                    continue;
                }

                b2c_tile_map = OffsettedBlockToCTileMap(
                    LocalBlock2ETileMap(M, N, 4), group_offset, tile_offset);
                grid_size_grp = b2c_tile_map.CalculateGridSize(M, N);

                gemm_tile_id_start = group_offset;
                gemm_tile_id_end   = group_offset + grid_size_grp;
            }

            using DsGridPointer = decltype(GridwiseGemm::MakeDsGridPointer());
            DsGridPointer p_ds_grid;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                p_ds_grid(i) = static_cast<const DDataType*>(gemm_desc_ptr[group_id].p_ds_grid[i]);
            });

            static constexpr index_t kbatch  = 1;
            static constexpr index_t k_grain = kbatch * KPerBlock;
            index_t K_split                  = (K + k_grain - 1) / k_grain * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            // Update tile offset if we have moved within group
            b2c_tile_map.UpdateTileOffset(tile_offset);

            using Problem = typename GridwiseGemm::Problem;
            auto problem  = Problem(gemm_desc_ptr[group_id].M,
                                   gemm_desc_ptr[group_id].N,
                                   gemm_desc_ptr[group_id].K,
                                   gemm_desc_ptr[group_id].StrideA,
                                   gemm_desc_ptr[group_id].StrideB,
                                   gemm_desc_ptr[group_id].StrideDs,
                                   gemm_desc_ptr[group_id].StrideE,
                                   kbatch);

            if(has_main_k_block_loop)
            {
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                               true,
                                               InMemoryDataOperationEnum::Set,
                                               TailNumber::Full>(
                        static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                        static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                        p_ds_grid,
                        static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                        static_cast<void*>(p_shared),
                        problem,
                        a_element_op,
                        b_element_op,
                        cde_element_op,
                        b2c_tile_map);
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                    {
                        GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                                   true,
                                                   InMemoryDataOperationEnum::Set,
                                                   TailNumber::One>(
                            static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                            static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                            p_ds_grid,
                            static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                            static_cast<void*>(p_shared),
                            problem,
                            a_element_op,
                            b_element_op,
                            cde_element_op,
                            b2c_tile_map);
                    }
                    else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Full)
                    {
                        GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                                   true,
                                                   InMemoryDataOperationEnum::Set,
                                                   TailNumber::Full>(
                            static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                            static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                            p_ds_grid,
                            static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                            static_cast<void*>(p_shared),
                            problem,
                            a_element_op,
                            b_element_op,
                            cde_element_op,
                            b2c_tile_map);
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                        {
                            GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                                       true,
                                                       InMemoryDataOperationEnum::Set,
                                                       TailNumber::Two>(
                                static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                                static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                                p_ds_grid,
                                static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                                static_cast<void*>(p_shared),
                                problem,
                                a_element_op,
                                b_element_op,
                                cde_element_op,
                                b2c_tile_map);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Three)
                        {
                            GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                                       true,
                                                       InMemoryDataOperationEnum::Set,
                                                       TailNumber::Three>(
                                static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                                static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                                p_ds_grid,
                                static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                                static_cast<void*>(p_shared),
                                problem,
                                a_element_op,
                                b_element_op,
                                cde_element_op,
                                b2c_tile_map);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Four)
                        {
                            GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                                       true,
                                                       InMemoryDataOperationEnum::Set,
                                                       TailNumber::Four>(
                                static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                                static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                                p_ds_grid,
                                static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                                static_cast<void*>(p_shared),
                                problem,
                                a_element_op,
                                b_element_op,
                                cde_element_op,
                                b2c_tile_map);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Five)
                        {
                            GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                                       true,
                                                       InMemoryDataOperationEnum::Set,
                                                       TailNumber::Five>(
                                static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                                static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                                p_ds_grid,
                                static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                                static_cast<void*>(p_shared),
                                problem,
                                a_element_op,
                                b_element_op,
                                cde_element_op,
                                b2c_tile_map);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                        {
                            GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                                       true,
                                                       InMemoryDataOperationEnum::Set,
                                                       TailNumber::Six>(
                                static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                                static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                                p_ds_grid,
                                static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                                static_cast<void*>(p_shared),
                                problem,
                                a_element_op,
                                b_element_op,
                                cde_element_op,
                                b2c_tile_map);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Seven)
                        {
                            GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                                       true,
                                                       InMemoryDataOperationEnum::Set,
                                                       TailNumber::Seven>(
                                static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                                static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                                p_ds_grid,
                                static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                                static_cast<void*>(p_shared),
                                problem,
                                a_element_op,
                                b_element_op,
                                cde_element_op,
                                b2c_tile_map);
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                    {
                        GridwiseGemm::template Run_2Lds<OffsettedBlockToCTileMap,
                                                        true,
                                                        InMemoryDataOperationEnum::Set,
                                                        TailNumber::Odd>(
                            static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                            static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                            p_ds_grid,
                            static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                            static_cast<void*>(p_shared),
                            static_cast<void*>(p_shared1),
                            problem,
                            a_element_op,
                            b_element_op,
                            cde_element_op,
                            b2c_tile_map);
                    }
                    else
                    {
                        GridwiseGemm::template Run_2Lds<OffsettedBlockToCTileMap,
                                                        true,
                                                        InMemoryDataOperationEnum::Set,
                                                        TailNumber::Even>(
                            static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                            static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                            p_ds_grid,
                            static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                            static_cast<void*>(p_shared),
                            static_cast<void*>(p_shared1),
                            problem,
                            a_element_op,
                            b_element_op,
                            cde_element_op,
                            b2c_tile_map);
                    }
                }
            }
            else
            {
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    GridwiseGemm::template Run<OffsettedBlockToCTileMap,
                                               false,
                                               InMemoryDataOperationEnum::Set,
                                               TailNumber::Full>(
                        static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid),
                        static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid),
                        p_ds_grid,
                        static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                        static_cast<void*>(p_shared),
                        problem,
                        a_element_op,
                        b_element_op,
                        cde_element_op,
                        b2c_tile_map);
                }
            }

            tile_id += get_grid_size();
            tile_offset += get_grid_size();

        } while(group_id < group_count);
    }
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
#endif // end of if (defined(__gfx9__))
}

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
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
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
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA>

struct DeviceGroupedGemmMultipleDXdlCShuffleTileLoop
    : public DeviceGroupedGemmTileLoop<ALayout,
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
    using DeviceOp = DeviceGroupedGemmMultipleDXdlCShuffleTileLoop;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();
    static constexpr index_t NumDTensor = DsDataType::Size();

    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemmMultiD_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        ADataType,
        BDataType,
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
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    using KernelArguments = GroupedGemmKernelArgument<NumDTensor>;
    using Block2ETileMap  = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;
    using OffsettedLocalBlock2ETileMap = OffsettedBlockToCTileMap2<Block2ETileMap>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(std::vector<const void*>& /* p_As */,
                 std::vector<const void*>& /* p_Bs */,
                 std::vector<std::array<const void*, NumDTensor>>& /* p_Ds */,
                 std::vector<void*>& /* p_Es */,
                 const std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op,
                 int occupancy_num_blocks,
                 int gpu_cu_count)
            : group_count_{static_cast<index_t>(gemm_descs.size())},
              occupancy_num_blocks_{occupancy_num_blocks},
              gpu_cu_count_{gpu_cu_count},
              gemm_descs_{gemm_descs},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              tile_count_{0}
        {
            for(const auto& desc : gemm_descs)
            {
                const auto M            = desc.M_;
                const auto N            = desc.N_;
                const auto b2c_tile_map = Block2ETileMap(M, N);
                tile_count_ += b2c_tile_map.CalculateGridSize(M, N);
            }
        }

        index_t group_count_;
        const void* p_dev_gemm_args_;
        int occupancy_num_blocks_;
        int gpu_cu_count_;
        const std::vector<GemmDesc>& gemm_descs_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
        index_t tile_count_;
    };

    struct KernelConfig
    {
        // The oversubscription factor for the number of blocks that can simultaneously reside on
        // GPU.
        static constexpr int BLOCK_SUBSCRIPTION_FACTOR = 1;
        // static constexpr int BLOCK_WAVES               = BlockSize / get_warp_size();
        static constexpr int CU_SIMDS = 4;
        // Assume we want to have at most 2 waves per SIMD
        // static constexpr int CU_BLOCKS = math::integer_divide_floor(2 * CU_SIMDS, BLOCK_WAVES);
        static int GetCuBlocks()
        {
            int BLOCK_WAVES = BlockSize / get_warp_size();
            return math::integer_divide_floor(2 * CU_SIMDS, BLOCK_WAVES);
        }
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        ///
        /// @brief      Launch Grouped Gemm kernel.
        ///
        /// @note       This function overload is using user provided device buffer for kernel
        ///             arguments.
        ///
        /// @param[in]  arg                 The structure containing kernel arguments (in host
        ///                                 memory).
        /// @param[in]  dev_gemm_args       The pointer to device memory with kernel arguments.
        /// @param[in]  stream_config       The device stream configuration.
        ///
        /// @return     The average kernel execution time (if time measurement is enabled.)
        ///
        template <typename GridwiseGemm>
        float Run(const Argument& arg,
                  const void* dev_gemm_args,
                  const StreamConfig& stream_config = StreamConfig{})
        {
            if(dev_gemm_args == nullptr)
            {
                std::ostringstream err;
                err << "The gemm arguments device buffer is not allocated!" << " In " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            float ave_time = 0;
            ave_time       = DispatchKernel<GridwiseGemm>(arg, dev_gemm_args, stream_config);

            return ave_time;
        }

        ///
        /// @brief      Launch Grouped Gemm kernel.
        ///
        /// @note       This function overload is using device buffers (for kernel arguments and
        ///             for kernel auxiliary workspace) provided with an argument. The user should
        ///             call @see GetDeviceKernelArgSize, and @see SetDeviceKernelArgs, on arg
        ///             parameter to properly allocate those buffers.
        ///
        /// @param[in]  arg            The structure containing kernel arguments (in host memory).
        /// @param[in]  stream_config  The device stream configuration.
        ///
        /// @return     The average kernel execution time (if time measurement is enabled.)
        ///
        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(arg.p_dev_gemm_args_ == nullptr)
            {
                std::ostringstream err;
                err << "The gemm arguments device buffer is not allocated!" << " In " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            return Run<GridwiseGemm>(arg, arg.p_dev_gemm_args_, stream_config);
        }

        INVOKER_RUN_IMPL

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }

        private:
        template <typename GridwiseGemm>
        float DispatchKernel(const Argument& arg,
                             const void* dev_gemm_args,
                             const StreamConfig& stream_config) const
        {
            const auto kernel = kernel_grouped_gemm_multiple_d_xdl<GridwiseGemm,
                                                                   KernelArguments,
                                                                   GemmSpec,
                                                                   ADataType,
                                                                   BDataType,
                                                                   DsDataType,
                                                                   EDataType,
                                                                   ALayout,
                                                                   BLayout,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   KPerBlock,
                                                                   OffsettedLocalBlock2ETileMap,
                                                                   Block2ETileMap,
                                                                   AElementwiseOperation,
                                                                   BElementwiseOperation,
                                                                   CDEElementwiseOperation,
                                                                   BlkGemmPipeSched,
                                                                   BlkGemmPipelineVer>;
            return LaunchKernel(kernel, arg, dev_gemm_args, stream_config);
        }

        template <typename KernelFunction>
        int CalculateMaxOccupancyGridSize(const KernelFunction& kernel,
                                          const StreamConfig& stream_config) const
        {
            // Calculate max number of workgroups that can simultaneously reside on the CU.
            int occ_num_blocks            = 0;
            size_t dyn_shared_mem_per_blk = 0;
            hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &occ_num_blocks, kernel, BlockSize, dyn_shared_mem_per_blk));

            int cu_count = getAvailableComputeUnitCount(stream_config);

            if(stream_config.log_level_ > 0)
            {
                std::cout << "MaxActiveBlocksPerCU: " << occ_num_blocks
                          << ", available CUs count: " << cu_count << ", occup. grid size: "
                          << ck::math::min(occ_num_blocks, KernelConfig::GetCuBlocks()) * cu_count
                          << std::endl;
            }

            return cu_count * ck::math::min(occ_num_blocks, KernelConfig::GetCuBlocks());
        }

        template <typename KernelFunction>
        float LaunchKernel(const KernelFunction& kernel,
                           const Argument& arg,
                           const void* dev_gemm_args,
                           const StreamConfig& stream_config) const
        {
            int grid_size = CalculateMaxOccupancyGridSize(kernel, stream_config);

            if(stream_config.log_level_ > 0)
            {
                std::cout << "grid_size: " << grid_size << " tile_count: " << arg.tile_count_
                          << std::endl;
            }

            // run multiple kernels

            return launch_and_time_kernel(stream_config,
                                          kernel,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          cast_pointer_to_constant_address_space(dev_gemm_args),
                                          arg.group_count_,
                                          arg.a_element_op_,
                                          arg.b_element_op_,
                                          arg.cde_element_op_);
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
        bool supported = true;

        constexpr index_t k_batch = 1;
        bool isWave64             = get_warp_size() == 64;
        for(index_t i = 0; i < arg.group_count_; ++i)
        {
            std::array<const void*, NumDTensor> placeholder_p_ds_grid{};
            std::array<index_t, NumDTensor> stride_Ds;
            std::copy_n(arg.gemm_descs_[i].stride_Ds_.begin(), NumDTensor, stride_Ds.begin());
            if((arg.gemm_descs_[i].K_ % AK1 != 0 || arg.gemm_descs_[i].K_ % BK1 != 0) &&
               !(GemmSpec == GemmSpecialization::MKPadding ||
                 GemmSpec == GemmSpecialization::NKPadding ||
                 GemmSpec == GemmSpecialization::MNKPadding ||
                 GemmSpec == GemmSpecialization::KPadding))
            {
                return false;
            }
            if(isWave64)
            {
                if constexpr(NXdlPerWave64 > 0)
                {
                    using GridArg = typename GridwiseGemm64::Argument;
                    GridArg gridwise_arg(nullptr,               // p_a_grid,
                                         nullptr,               // p_b_grid,
                                         placeholder_p_ds_grid, // p_ds_grid,
                                         nullptr,               // p_e_grid  ,
                                         arg.gemm_descs_[i].M_,
                                         arg.gemm_descs_[i].N_,
                                         arg.gemm_descs_[i].K_,
                                         arg.gemm_descs_[i].stride_A_,
                                         arg.gemm_descs_[i].stride_B_,
                                         stride_Ds,
                                         arg.gemm_descs_[i].stride_C_,
                                         k_batch,
                                         arg.a_element_op_,
                                         arg.b_element_op_,
                                         arg.cde_element_op_);

                    supported = supported && GridwiseGemm64::CheckValidity(gridwise_arg);
                }
                else
                {
                    supported = false;
                }
            }
            else
            {
                if constexpr(NXdlPerWave32 > 0)
                {
                    using GridArg = typename GridwiseGemm32::Argument;
                    GridArg gridwise_arg(nullptr,               // p_a_grid,
                                         nullptr,               // p_b_grid,
                                         placeholder_p_ds_grid, // p_ds_grid,
                                         nullptr,               // p_e_grid  ,
                                         arg.gemm_descs_[i].M_,
                                         arg.gemm_descs_[i].N_,
                                         arg.gemm_descs_[i].K_,
                                         arg.gemm_descs_[i].stride_A_,
                                         arg.gemm_descs_[i].stride_B_,
                                         stride_Ds,
                                         arg.gemm_descs_[i].stride_C_,
                                         k_batch,
                                         arg.a_element_op_,
                                         arg.b_element_op_,
                                         arg.cde_element_op_);

                    supported = supported && GridwiseGemm32::CheckValidity(gridwise_arg);
                }
                else
                {
                    supported = false;
                }
            }
        }

        return supported;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static int GetKernelOccupancy()
    {
        int occupancy = 0;
        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                const auto kernel = kernel_grouped_gemm_multiple_d_xdl<GridwiseGemm64,
                                                                       KernelArguments,
                                                                       GemmSpec,
                                                                       ADataType,
                                                                       BDataType,
                                                                       DsDataType,
                                                                       EDataType,
                                                                       ALayout,
                                                                       BLayout,
                                                                       DsLayout,
                                                                       ELayout,
                                                                       KPerBlock,
                                                                       OffsettedLocalBlock2ETileMap,
                                                                       Block2ETileMap,
                                                                       AElementwiseOperation,
                                                                       BElementwiseOperation,
                                                                       CDEElementwiseOperation,
                                                                       BlkGemmPipeSched,
                                                                       BlkGemmPipelineVer>;
                hip_check_error(
                    hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize, 0));
            }
        }
        else
        {

            if constexpr(NXdlPerWave32 > 0)
            {
                const auto kernel = kernel_grouped_gemm_multiple_d_xdl<GridwiseGemm32,
                                                                       KernelArguments,
                                                                       GemmSpec,
                                                                       ADataType,
                                                                       BDataType,
                                                                       DsDataType,
                                                                       EDataType,
                                                                       ALayout,
                                                                       BLayout,
                                                                       DsLayout,
                                                                       ELayout,
                                                                       KPerBlock,
                                                                       OffsettedLocalBlock2ETileMap,
                                                                       Block2ETileMap,
                                                                       AElementwiseOperation,
                                                                       BElementwiseOperation,
                                                                       CDEElementwiseOperation,
                                                                       BlkGemmPipeSched,
                                                                       BlkGemmPipelineVer>;
                hip_check_error(
                    hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize, 0));
            }
        }
        return occupancy;
    }

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc>& gemm_descs,
                             AElementwiseOperation a_elementwise_op,
                             BElementwiseOperation b_elementwise_op,
                             CDEElementwiseOperation cde_elementwise_op)
    {
        int occupancy = GetKernelOccupancy();
        int num_cu;

        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        hip_check_error(hipGetDevice(&dev));
        hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;

        return Argument{p_As,
                        p_Bs,
                        p_Ds,
                        p_Es,
                        gemm_descs,
                        a_elementwise_op,
                        b_elementwise_op,
                        cde_elementwise_op,
                        occupancy,
                        num_cu};
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation a_elementwise_op,
                        BElementwiseOperation b_elementwise_op,
                        CDEElementwiseOperation cde_elementwise_op) override
    {
        int occupancy = GetKernelOccupancy();
        int num_cu;

        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        hip_check_error(hipGetDevice(&dev));
        hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;

        return std::make_unique<Argument>(p_As,
                                          p_Bs,
                                          p_Ds,
                                          p_Es,
                                          gemm_descs,
                                          a_elementwise_op,
                                          b_elementwise_op,
                                          cde_elementwise_op,
                                          occupancy,
                                          num_cu);
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::ostringstream();

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
        str << "DeviceGroupedGemmMultipleDXdlCShuffleTileLoop"
            << "<"
            << std::string(ALayout::name)[0] << ","
            << std::string(BLayout::name)[0] << ","
            << std::string(ELayout::name)[0] << ","
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
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << getGemmSpecializationString(GemmSpec) << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer]
            << ">";
        // clang-format on

        return str.str();
    }

    void SetDeviceKernelArgs(Argument& arg,
                             void* p_dev_kernel_args,
                             const void* p_host_kernel_args) const
    {
        arg.p_dev_gemm_args_ = p_dev_kernel_args;
        hip_check_error(hipMemcpyAsync(p_dev_kernel_args,
                                       p_host_kernel_args,
                                       GetDeviceKernelArgSize(&arg),
                                       hipMemcpyHostToDevice));
    }

    virtual void SetDeviceKernelArgs(BaseArgument* p_arg,
                                     void* p_dev_kernel_args,
                                     const void* p_host_kernel_args) const override
    {
        return SetDeviceKernelArgs(
            *dynamic_cast<Argument*>(p_arg), p_dev_kernel_args, p_host_kernel_args);
    }

    void SetDeviceKernelArgs(Argument& arg, void* p_dev_kernel_args) const
    {
        arg.p_dev_gemm_args_ = p_dev_kernel_args;
    }

    virtual void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const override
    {
        return SetDeviceKernelArgs(*dynamic_cast<Argument*>(p_arg), p_dev_kernel_args);
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->group_count_ * sizeof(KernelArguments);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
