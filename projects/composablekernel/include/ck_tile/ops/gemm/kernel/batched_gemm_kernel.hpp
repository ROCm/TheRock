// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

/// @brief The Batched GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref BatchedGemmKernel "BatchedGemmKernel" when creating kernel
///      arguments object. It contain all necessary information required to build proper kernel
///      argument and launch kernel on GPU. This structure defines the GEMM problem configuration by
///      stating all required information like M,N,K sizes and respective strides.
struct BatchedGemmHostArgs : public ck_tile::UniversalGemmHostArgs<>
{
    CK_TILE_HOST explicit BatchedGemmHostArgs(const void* a_ptr_,
                                              const void* b_ptr_,
                                              void* c_ptr_,
                                              ck_tile::index_t k_batch_,
                                              ck_tile::index_t M_,
                                              ck_tile::index_t N_,
                                              ck_tile::index_t K_,
                                              ck_tile::index_t stride_A_,
                                              ck_tile::index_t stride_B_,
                                              ck_tile::index_t stride_C_,
                                              ck_tile::index_t batch_stride_A_,
                                              ck_tile::index_t batch_stride_B_,
                                              ck_tile::index_t batch_stride_C_,
                                              ck_tile::index_t batch_count_)
        : UniversalGemmHostArgs<>({a_ptr_},
                                  {b_ptr_},
                                  {/*ds_ptr*/},
                                  c_ptr_,
                                  k_batch_,
                                  M_,
                                  N_,
                                  K_,
                                  {stride_A_},
                                  {stride_B_},
                                  {/*stride_Ds_*/},
                                  stride_C_),
          batch_stride_A(batch_stride_A_),
          batch_stride_B(batch_stride_B_),
          batch_stride_E(batch_stride_C_),
          batch_count(batch_count_)
    {
    }

    ck_tile::index_t batch_stride_A;
    ck_tile::index_t batch_stride_B;
    ck_tile::index_t batch_stride_E;
    ck_tile::index_t batch_count;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct BatchedGemmKernel
{
    /// @brief Inject the UniversalGemmKernel base class to support execution of all necessary
    /// functions.
    using UniversalGemmKernel =
        UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;
    static constexpr index_t kBlockSize = UniversalGemmKernel::kBlockSize;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    /// @brief Specify the layout configurations for A, B, E and D
    using ALayout = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout = remove_cvref_t<typename GemmPipeline::CLayout>;

    /// @brief Specify the data type configurations for A, B, E and D
    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    /// @brief ALayout and ADataType are expected to be scalars, not a tuple.
    static_assert(
        !is_detected<is_tuple, ALayout>::value && !is_detected<is_tuple, ADataType>::value,
        "ALayout and ADataType must be scalars. Multiple parameters are not currently supported.");

    /// @brief  BLayout and BDataType are expected to be scalars, not a tuple.
    static_assert(
        !is_detected<is_tuple, BLayout>::value && !is_detected<is_tuple, BDataType>::value,
        "BLayout and BDataType must be scalars. Multiple parameters are not currently supported.");

    /// @brief  C/CLayout and C/EDataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, CLayout>::value &&
                      !is_detected<is_tuple, CDataType>::value,
                  "C/CLayout and C/EDataType must be scalars.");

    struct BatchedGemmKernelArgs : ck_tile::UniversalGemmKernelArgs<>
    {
        index_t batch_stride_A;
        index_t batch_stride_B;
        index_t batch_stride_E;
        index_t batch_count;
    };

    using KernelArgs = BatchedGemmKernelArgs;

    [[nodiscard]] CK_TILE_HOST static auto GetName() -> const std::string
    {
        // clang-format off
        using P_ = GemmPipeline;
        return concat('_', "gemm_batched", gemm_prec_str<ADataType, BDataType>(),
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock), 
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK));
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto
    GridSize(index_t M, index_t N, index_t KBatch, index_t batch_count) -> dim3
    {
        return dim3(TilePartitioner::GridSize(M, N), batch_count, KBatch);
    }

    CK_TILE_HOST static auto BlockSize() -> dim3
    {
        if(ck_tile::is_wave32())
        {
            return dim3(UniversalGemmKernel::kBlockSize / 2);
        }
        else
        {
            return dim3(UniversalGemmKernel::kBlockSize);
        }
    }

    CK_TILE_HOST static constexpr BatchedGemmKernelArgs
    MakeKernelArgs(const BatchedGemmHostArgs& hostArgs)
    {
        return BatchedGemmKernelArgs{{hostArgs.as_ptr,
                                      hostArgs.bs_ptr,
                                      hostArgs.ds_ptr,
                                      hostArgs.e_ptr,
                                      hostArgs.M,
                                      hostArgs.N,
                                      hostArgs.K,
                                      hostArgs.stride_As,
                                      hostArgs.stride_Bs,
                                      hostArgs.stride_Ds,
                                      hostArgs.stride_E,
                                      hostArgs.k_batch},
                                     hostArgs.batch_stride_A,
                                     hostArgs.batch_stride_B,
                                     hostArgs.batch_stride_E,
                                     hostArgs.batch_count};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST static auto
    IsSupportedArgument(const typename BatchedGemmKernel::KernelArgs& kargs) -> bool
    {
        if(kargs.batch_count < 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Conditions not met: batch_count must be at least 1 !");
            }
            return false;
        }
        if(kargs.batch_stride_A < 0 || kargs.batch_stride_A < kargs.M * kargs.K)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(
                    "Conditions not met: batch_stride_A must be non-negative and at least K * M!");
            }
            return false;
        }
        if(kargs.batch_stride_B < 0 || kargs.batch_stride_B < kargs.K * kargs.N)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(
                    "Conditions not met: batch_stride_B must be non-negative and at least K * N!");
            }
            return false;
        }
        if(kargs.batch_stride_E < 0 || kargs.batch_stride_E < kargs.M * kargs.N)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(
                    "Conditions not met: batch_stride_E must be non-negative and at least M * N!");
            }
            return false;
        }
        return UniversalGemmKernel::IsSupportedArgument(kargs);
    }

    CK_TILE_DEVICE void operator()(BatchedGemmKernelArgs kargs) const
    {
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockIdx.x);
        const index_t i_m   = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

        const auto i_batch  = amd_wave_read_first_lane(blockIdx.y);
        const auto i_splitk = amd_wave_read_first_lane(blockIdx.z);

        const typename UniversalGemmKernel::SplitKBatchOffset splitk_batch_offset(kargs, i_splitk);

        //  options
        const auto batch_stride_A = amd_wave_read_first_lane(kargs.batch_stride_A);
        const auto batch_offset_A = amd_wave_read_first_lane(i_batch * batch_stride_A);
        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.as_ptr[0]) + batch_offset_A +
                                 splitk_batch_offset.as_k_split_offset[0];

        const auto batch_stride_B = amd_wave_read_first_lane(kargs.batch_stride_B);
        const auto batch_offset_B = amd_wave_read_first_lane(i_batch * batch_stride_B);
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.bs_ptr[0]) + batch_offset_B +
                                 splitk_batch_offset.bs_k_split_offset[0];

        const auto batch_stride_E = amd_wave_read_first_lane(kargs.batch_stride_E);
        const auto batch_offset_C = amd_wave_read_first_lane(i_batch * batch_stride_E);
        CDataType* c_ptr          = static_cast<CDataType*>(kargs.e_ptr) + batch_offset_C;

        // allocate LDS
        __shared__ char smem_ptr0[GetSmemSize()];

        if constexpr(GemmPipeline::DoubleSmemBuffer == true)
        {
            __shared__ char smem_ptr1[GemmPipeline::GetSmemSize()];
            UniversalGemmKernel::RunGemm2LDS({a_ptr},
                                             {b_ptr},
                                             {/*ds_ptr*/},
                                             c_ptr,
                                             smem_ptr0,
                                             smem_ptr1,
                                             kargs,
                                             splitk_batch_offset,
                                             i_m,
                                             i_n);
        }
        else
        {
            UniversalGemmKernel::RunGemm({a_ptr},
                                         {b_ptr},
                                         {/*ds_ptr*/},
                                         c_ptr,
                                         smem_ptr0,
                                         kargs,
                                         splitk_batch_offset,
                                         i_m,
                                         i_n);
        }
    }
};

} // namespace ck_tile
