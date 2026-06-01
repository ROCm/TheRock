// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp"
#include "ck_tile/host.hpp"

#include <hip/hip_runtime.h>

namespace ck_tile {

/// @brief The Grouped GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref GroupedGemmKernel "GroupedGemmKernel" when creating kernel
///      arguments object. It contain all necessary information required to build proper kernel
///      argument and launch kernel on GPU. This structure defines the GEMM problem configuration by
///      stating all required information like M,N,K sizes and respective strides.
struct QuantGroupedGemmHostArgs
{
    CK_TILE_HOST QuantGroupedGemmHostArgs(const void* a_ptr_,
                                          const void* b_ptr_,
                                          void* e_ptr_,
                                          const void* aq_ptr_,
                                          const void* bq_ptr_,
                                          index_t k_batch_,
                                          index_t M_,
                                          index_t N_,
                                          index_t K_,
                                          index_t QK_A_,
                                          index_t QK_B_,
                                          index_t stride_A_,
                                          index_t stride_B_,
                                          index_t stride_E_,
                                          index_t stride_AQ_,
                                          index_t stride_BQ_)
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          aq_ptr(aq_ptr_),
          bq_ptr(bq_ptr_),
          e_ptr(e_ptr_),
          M(M_),
          N(N_),
          K(K_),
          QK_A(QK_A_),
          QK_B(QK_B_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_AQ(stride_AQ_),
          stride_BQ(stride_BQ_),
          stride_E(stride_E_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    const void* aq_ptr;
    const void* bq_ptr;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };

    index_t M;
    index_t N;
    index_t K;
    index_t QK_A;
    index_t QK_B;
    index_t stride_A;
    index_t stride_B;
    index_t stride_AQ;
    index_t stride_BQ;

    union
    {
        index_t stride_E;
        index_t stride_C;
    };

    index_t k_batch;
};

using QuantGroupedGemmKernelArgs = QuantGemmKernelArgs;

struct QuantGemmTransKernelArg
{
    QuantGroupedGemmKernelArgs group_karg;
    ck_tile::index_t block_start;
    ck_tile::index_t block_end;

    QuantGemmTransKernelArg() = delete;
    QuantGemmTransKernelArg(QuantGroupedGemmKernelArgs&& karg, index_t bl_start, index_t bl_end)
        : group_karg{karg}, block_start{bl_start}, block_end{bl_end}
    {
    }

    QuantGemmTransKernelArg(QuantGroupedGemmKernelArgs&& karg)
        : group_karg{karg}, block_start{0}, block_end{0}
    {
    }
};

template <typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_,
          QuantType QuantType_>
struct QuantGroupedGemmKernel
{
    /// @brief Inject the UniversalGemmKernel base class to support execution of all necessary
    /// functions.
    using Base = QuantGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_, QuantType_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    //// @brief Specify the layout configurations for A, B, C/E
    using ALayout = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout = remove_cvref_t<typename GemmPipeline::CLayout>;

    /// @brief Specify the data type configurations for A, B, C/E
    using ADataType   = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType   = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType   = remove_cvref_t<typename EpiloguePipeline::ODataType>;
    using AccDataType = remove_cvref_t<typename EpiloguePipeline::AccDataType>;

    using AQDataType =
        remove_cvref_t<typename detail::get_aq_data_type_or<GemmPipeline, AccDataType>::type>;
    using BQDataType =
        remove_cvref_t<typename detail::get_bq_data_type_or<GemmPipeline, AccDataType>::type>;

    static constexpr auto kQuantType = QuantType_;

    /// @brief ALayout and ADataType are expected to be scalars, not a tuple.
    static_assert(
        !is_detected<is_tuple, ALayout>::value && !is_detected<is_tuple, ADataType>::value,
        "ALayout and ADataType must be scalars. Multiple parameters are not currently supported.");

    /// @brief  BLayout and BDataType are expected to be scalars, not a tuple.
    static_assert(
        !is_detected<is_tuple, BLayout>::value && !is_detected<is_tuple, BDataType>::value,
        "BLayout and BDataType must be scalars. Multiple parameters are not currently supported.");

    /// @brief  C/ELayout and C/EDataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, CLayout>::value &&
                      !is_detected<is_tuple, CDataType>::value,
                  "C/ELayout and C/EDataType must be scalars.");

    using OffsetTile1DPartitioner = OffsettedTile1DPartitioner<TilePartitioner>;
    using Kernel =
        QuantGroupedGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline, kQuantType>;

    static constexpr index_t kBlockSize       = GemmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = GemmPipeline::UsePersistentKernel;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;

        return concat('_', "gemm_grouped", gemm_prec_str<ADataType, BDataType>(),
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK),
                      (UsePersistentKernel ? "Persistent" : "NonPersistent"));
        // clang-format on
    }

    CK_TILE_HOST static auto
    GetWorkSpaceSize(const std::vector<QuantGroupedGemmHostArgs>& gemm_descs) -> std::size_t
    {
        return gemm_descs.size() * sizeof(QuantGemmTransKernelArg);
    }

    CK_TILE_HOST static auto GetWorkSpaceSize(index_t group_count) -> std::size_t
    {
        return group_count * sizeof(QuantGemmTransKernelArg);
    }

    CK_TILE_HOST static auto BlockSize() -> dim3
    {
        if(is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    /**
     * @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
     * @return The maximum occupancy grid size.
     * @note This function queries the maximum occupancy of the kernel using
     *       `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        using ConstantPointer  = const void CK_TILE_CONSTANT_ADDRESS_SPACE*;
        const auto kernel_func = kentry<1, Kernel, ConstantPointer, index_t>;
        int occupancy;
        HIP_CHECK_ERROR(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel_func, kBlockSize, 0));
        const int grid_size = get_available_compute_units(s) * occupancy;
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto GridSize(const std::vector<QuantGroupedGemmHostArgs>& gemm_descs)
    {
        index_t grid_size = 0;
        for(const auto& it_desc : gemm_descs)
        {
            const auto local_grid_size = TilePartitioner::GridSize(it_desc.M, it_desc.N);
            grid_size += local_grid_size * it_desc.k_batch;
        }
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto MakeKargs(const std::vector<QuantGroupedGemmHostArgs>& gemm_descs)
        -> std::vector<QuantGemmTransKernelArg>
    {
        std::vector<QuantGemmTransKernelArg> gemm_kernel_args_;
        index_t group_count = ck_tile::type_convert<ck_tile::index_t>(gemm_descs.size());
        index_t grid_size   = 0;
        gemm_kernel_args_.reserve(group_count);

        for(std::size_t i = 0; i < gemm_descs.size(); ++i)
        {
            const index_t M = gemm_descs[i].M;
            const index_t N = gemm_descs[i].N;
            const index_t K = gemm_descs[i].K;

            if(M == 0 || N == 0 || K == 0)
            {
                continue;
            }

            const index_t stride_a = gemm_descs[i].stride_A;
            const index_t stride_b = gemm_descs[i].stride_B;
            const index_t stride_e = gemm_descs[i].stride_C;

            const index_t grid_size_grp = TilePartitioner::GridSize(M, N) * gemm_descs[i].k_batch;

            const index_t block_start = grid_size;
            const index_t block_end   = grid_size + grid_size_grp;

            grid_size += grid_size_grp;

            auto karg =
                QuantGroupedGemmKernelArgs{type_convert<const ADataType*>(gemm_descs[i].a_ptr),
                                           type_convert<const BDataType*>(gemm_descs[i].b_ptr),
                                           type_convert<const AQDataType*>(gemm_descs[i].aq_ptr),
                                           type_convert<const BQDataType*>(gemm_descs[i].bq_ptr),
                                           type_convert<CDataType*>(gemm_descs[i].e_ptr),
                                           M,
                                           N,
                                           K,
                                           gemm_descs[i].QK_A,
                                           gemm_descs[i].QK_B,
                                           stride_a,
                                           stride_b,
                                           stride_e,
                                           gemm_descs[i].stride_AQ,
                                           gemm_descs[i].stride_BQ,
                                           gemm_descs[i].k_batch};

            gemm_kernel_args_.emplace_back(std::move(karg), block_start, block_end);
        }

        return gemm_kernel_args_;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const std::vector<QuantGemmTransKernelArg>& kargs)
    {
        for(const auto& karg : kargs)
        {
            if(!Base::IsSupportedArgument(karg.group_karg))
            {
                return false;
            }
        }
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetSmemSize() -> index_t
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void Run(const QuantGroupedGemmKernelArgs& kargs,
                            const tuple<index_t, index_t>& block_idx_2d,
                            const index_t block_idx_z) const
    {
        const auto [iM, iN] = block_idx_2d;

        const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

        const typename Base::SplitKBatchOffset splitk_batch_offset(kargs, block_idx_z);

        // options
        const ADataType* a_ptr   = static_cast<const ADataType*>(kargs.a_ptr);
        const BDataType* b_ptr   = static_cast<const BDataType*>(kargs.b_ptr);
        const AQDataType* aq_ptr = static_cast<const AQDataType*>(kargs.aq_ptr);
        const BQDataType* bq_ptr = static_cast<const BQDataType*>(kargs.bq_ptr);
        CDataType* c_ptr         = static_cast<CDataType*>(kargs.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // Only for BQuantGrouped DoubleSmemBuffer is supported
        if constexpr(GemmPipeline::DoubleSmemBuffer == true &&
                     kQuantType == QuantType::BQuantGrouped)
        {
            RunGemmWithPipelineSelection2LDS(a_ptr,
                                             b_ptr,
                                             aq_ptr,
                                             bq_ptr,
                                             c_ptr,
                                             smem_ptr,
                                             kargs,
                                             splitk_batch_offset,
                                             i_m,
                                             i_n);
        }
        else
        {

            if constexpr(UsePersistentKernel)
            {
                RunGemmWithPipelineSelection(a_ptr,
                                             b_ptr,
                                             aq_ptr,
                                             bq_ptr,
                                             c_ptr,
                                             smem_ptr,
                                             kargs,
                                             splitk_batch_offset,
                                             i_m,
                                             i_n);
            }
            else // Non-persistent kernel
            {
                Base::RunGemm({a_ptr},
                              {b_ptr},
                              aq_ptr,
                              bq_ptr,
                              c_ptr,
                              smem_ptr,
                              kargs,
                              splitk_batch_offset,
                              i_m,
                              i_n);
            }
        }
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static void
    RunGemmWithPipelineSelection2LDS(const ADataType* a_ptr,
                                     const BDataType* b_ptr,
                                     [[maybe_unused]] const AQDataType* aq_ptr,
                                     const BQDataType* bq_ptr,
                                     CDataType* c_ptr,
                                     void* smem_ptr,
                                     const QuantGroupedGemmKernelArgs& kargs,
                                     const typename Base::SplitKBatchOffset& splitk_batch_offset,
                                     const index_t block_idx_m,
                                     const index_t block_idx_n)
    {
        static_assert(kQuantType == QuantType::BQuantGrouped, "kQuantType must be BQuantGrouped");

        // Create block windows using specialized methods
        const auto& a_block_window =
            Base::MakeABlockWindow(a_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_m);
        const auto& b_block_window =
            Base::MakeBBlockWindow(b_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_n);
        const auto& bq_block_window =
            Base::MakeBQBlockWindow(bq_ptr, kargs, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));
        const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);

        // Run GEMM cooperatively by whole workgroup
        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, bq_block_window, num_loop, tail_num, smem_ptr);

        // Run Epilogue Pipeline with split_k dispatch
        if(kargs.k_batch == 1)
        {
            auto c_block_window = Base::template MakeCBlockWindow<memory_operation_enum::set>(
                c_ptr, kargs, block_idx_m, block_idx_n);
            EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr);
        }
        else
        {
            auto c_block_window =
                Base::template MakeCBlockWindow<memory_operation_enum::atomic_add>(
                    c_ptr, kargs, block_idx_m, block_idx_n);
            EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr);
        }
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @note The GEMM pipeline is selected in-kernel based on the number of K-loops
     *       and the tail-number. This is needed for the persistent tile-loop when
     *       we didn't have access to the K dimension on the host.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param aq_ptr input AQ pointer
     * @param bq_ptr input BQ pointer
     * @param c_ptr output C pointer
     * @param smem_ptr The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k
     * batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void
    RunGemmWithPipelineSelection(const ADataType* a_ptr,
                                 const BDataType* b_ptr,
                                 const AQDataType* aq_ptr,
                                 const BQDataType* bq_ptr,
                                 CDataType* c_ptr,
                                 void* smem_ptr,
                                 const QuantGroupedGemmKernelArgs& kargs,
                                 const typename Base::SplitKBatchOffset& splitk_batch_offset,
                                 const index_t block_idx_m,
                                 const index_t block_idx_n)
    {
        // Create block windows using specialized methods
        const auto& a_block_window =
            Base::MakeABlockWindow(a_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_m);
        const auto& b_block_window =
            Base::MakeBBlockWindow(b_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_n);
        const auto& aq_block_window =
            Base::MakeAQBlockWindow(aq_ptr, kargs, block_idx_m, block_idx_n);
        const auto& bq_block_window =
            Base::MakeBQBlockWindow(bq_ptr, kargs, block_idx_m, block_idx_n);

        // Get hot-loop and tail configuration
        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));
        const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);

        // Run GEMM cooperatively by whole workgroup
        const auto& c_block_tile = [&]() {
            if constexpr(kQuantType == QuantType::AQuantGrouped)
            {
                return GemmPipeline{}.template operator()(a_block_window,
                                                          b_block_window,
                                                          aq_block_window,
                                                          num_loop,
                                                          has_hot_loop,
                                                          tail_num,
                                                          smem_ptr);
            }
            else if constexpr(kQuantType == QuantType::BQuantGrouped)
            {
                return GemmPipeline{}.template operator()(a_block_window,
                                                          b_block_window,
                                                          bq_block_window,
                                                          num_loop,
                                                          has_hot_loop,
                                                          tail_num,
                                                          smem_ptr);
            }
            else if constexpr(kQuantType == QuantType::ABQuantGrouped)
            {
                return GemmPipeline{}.template operator()(a_block_window,
                                                          b_block_window,
                                                          aq_block_window,
                                                          bq_block_window,
                                                          num_loop,
                                                          has_hot_loop,
                                                          tail_num,
                                                          smem_ptr);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant ||
                              kQuantType == QuantType::TensorQuant)
            {
                return GemmPipeline{}.template operator()(
                    a_block_window, b_block_window, num_loop, has_hot_loop, tail_num, smem_ptr);
            }
        }();

        // Run Epilogue Pipeline with split_k dispatch
        if(kargs.k_batch == 1)
        {
            auto c_block_window = Base::template MakeCBlockWindow<memory_operation_enum::set>(
                c_ptr, kargs, block_idx_m, block_idx_n);

            if constexpr(kQuantType == QuantType::AQuantGrouped ||
                         kQuantType == QuantType::BQuantGrouped ||
                         kQuantType == QuantType::ABQuantGrouped)
            {
                EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                EpiloguePipeline{}(c_block_window,
                                   c_block_tile,
                                   c_block_window,
                                   smem_ptr,
                                   aq_block_window,
                                   bq_block_window);
            }
            else if constexpr(kQuantType == QuantType::TensorQuant)
            {
                const AccDataType aq_scale = type_convert<AccDataType>(*aq_ptr);
                const AccDataType bq_scale = type_convert<AccDataType>(*bq_ptr);
                EpiloguePipeline{}(
                    c_block_window, c_block_tile, c_block_window, smem_ptr, aq_scale, bq_scale);
            }
        }
        else
        {
            auto c_block_window =
                Base::template MakeCBlockWindow<memory_operation_enum::atomic_add>(
                    c_ptr, kargs, block_idx_m, block_idx_n);

            if constexpr(kQuantType == QuantType::AQuantGrouped ||
                         kQuantType == QuantType::BQuantGrouped ||
                         kQuantType == QuantType::ABQuantGrouped)
            {
                EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                EpiloguePipeline{}(c_block_window,
                                   c_block_tile,
                                   c_block_window,
                                   smem_ptr,
                                   aq_block_window,
                                   bq_block_window);
            }
            else if constexpr(kQuantType == QuantType::TensorQuant)
            {
                const AccDataType aq_scale = type_convert<AccDataType>(*aq_ptr);
                const AccDataType bq_scale = type_convert<AccDataType>(*bq_ptr);
                EpiloguePipeline{}(
                    c_block_window, c_block_tile, c_block_window, smem_ptr, aq_scale, bq_scale);
            }
        }
    }

    CK_TILE_DEVICE index_t FindGroupId(const QuantGemmTransKernelArg* gemm_desc_ptr,
                                       index_t block_id,
                                       index_t group_count) const
    {
        index_t left     = 0;
        index_t right    = group_count;
        index_t group_id = index_t((left + right) >> 1);

        while((!(block_id >= gemm_desc_ptr[group_id].block_start &&
                 block_id < gemm_desc_ptr[group_id].block_end)) &&
              left <= right)
        {
            if(block_id < gemm_desc_ptr[group_id].block_start)
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) >> 1);
        }

        return group_id;
    }

    // For non-persistent kernels
    template <bool U = UsePersistentKernel, typename = std::enable_if_t<!U>>
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   index_t group_count) const
    {
        const index_t block_id   = ck_tile::get_block_1d_id();
        const auto gemm_desc_ptr = reinterpret_cast<const QuantGemmTransKernelArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        const index_t group_id = FindGroupId(gemm_desc_ptr, block_id, group_count);
        const auto& kargs      = gemm_desc_ptr[group_id];

        const auto grid_size_2d = TilePartitioner::GridSize(kargs.group_karg.M, kargs.group_karg.N);
        const auto block_idx_2d = OffsetTile1DPartitioner::GetOffsetedTileIndex(
            0,
            kargs.group_karg.M,
            kargs.group_karg.N,
            (block_id - kargs.block_start) % grid_size_2d);
        Run(kargs.group_karg, block_idx_2d, (block_id - kargs.block_start) / grid_size_2d);
    }

    // For persistent kernels
    template <bool U   = UsePersistentKernel,
              typename = std::enable_if_t<U>,
              typename = void> // extra template parameter to avoid redefinition
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   const index_t group_count) const
    {
        const index_t grid_size  = ck_tile::get_grid_size();
        const auto gemm_desc_ptr = reinterpret_cast<const QuantGemmTransKernelArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));
        index_t block_id      = ck_tile::get_block_1d_id(); // initial block_id
        index_t cum_grid_size = 0;
        for(index_t group_id = 0; group_id < group_count; ++group_id)
        {
            const auto& kargs      = gemm_desc_ptr[group_id].group_karg;
            const auto& k_batch    = kargs.k_batch;
            const auto block_start = cum_grid_size;
            cum_grid_size += TilePartitioner::GridSize(kargs.M, kargs.N) * k_batch;
            while(block_id < cum_grid_size)
            {
                const auto grid_size_2d = TilePartitioner::GridSize(kargs.M, kargs.N);
                const auto block_idx_2d = OffsetTile1DPartitioner::GetOffsetedTileIndex(
                    0, kargs.M, kargs.N, (block_id - block_start) % grid_size_2d);
                Run(kargs, block_idx_2d, (block_id - block_start) / grid_size_2d);
                block_id = block_id + grid_size; // advance to next block
                // NOTE: this check is redundant but helps the compiler avoid spilling some VGPR
                if(block_id >= cum_grid_size)
                {
                    break; // exit the loop if all blocks are processed
                }
            }
        }
    }
};

} // namespace ck_tile
