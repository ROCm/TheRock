// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

/// @brief The MultiABD GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref GemmKernelMultiABD "GemmKernelMultiABD" when creating
///      kernel arguments object. It contain all necessary information required to build proper
///      kernel argument and launch kernel on GPU. This structure defines the GEMM problem
///      configuration by stating all required information like M,N,K sizes and respective strides.
///      NumATensor describes the number of A tensors. The minimum number of tensors is 1(required).
///      NumBTensor describes the number of B tensors. The minimum number of tensors is 1(required).
///      NumDTensor describes the number of D tensors. The minimum number of tensors is 0(not
///      required).
template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct GemmMultiABDHostArgs
{
    CK_TILE_HOST GemmMultiABDHostArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                      const std::array<const void*, NumBTensor>& bs_ptr_,
                                      const std::array<const void*, NumDTensor>& ds_ptr_,
                                      void* e_ptr_,
                                      index_t k_batch_,
                                      index_t M_,
                                      index_t N_,
                                      index_t K_,
                                      const std::array<index_t, NumATensor>& stride_As_,
                                      const std::array<index_t, NumBTensor>& stride_Bs_,
                                      const std::array<index_t, NumDTensor>& stride_Ds_,
                                      index_t stride_E_)
        : as_ptr(as_ptr_),
          bs_ptr(bs_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          M(M_),
          N(N_),
          K(K_),
          stride_As(stride_As_),
          stride_Bs(stride_Bs_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          k_batch(k_batch_)
    {
    }

    const std::array<const void*, NumATensor> as_ptr;
    const std::array<const void*, NumBTensor> bs_ptr;
    const std::array<const void*, NumDTensor> ds_ptr;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };
    index_t M;
    index_t N;
    index_t K;
    const std::array<index_t, NumATensor> stride_As;
    const std::array<index_t, NumBTensor> stride_Bs;
    const std::array<index_t, NumDTensor> stride_Ds;
    union
    {
        index_t stride_E;
        index_t stride_C;
    };

    index_t k_batch;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GemmKernelMultiABD
{
    /// @brief Inject the UniversalGemmKernel base class to support execution of all necessary
    /// functions.
    using UniversalGemmKernel =
        UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;
    static constexpr index_t kBlockSize = UniversalGemmKernel::kBlockSize;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    /// @brief  Specify the layout configurations for A, B, E and D
    using AsLayout = remove_cvref_t<typename GemmPipeline::AsLayout>;
    using BsLayout = remove_cvref_t<typename GemmPipeline::BsLayout>;
    using CLayout  = remove_cvref_t<typename GemmPipeline::CLayout>;
    using DsLayout = remove_cvref_t<typename EpiloguePipeline::DsLayout>;

    /// @brief  Specify the data type configurations for A, B, E and D
    using AsDataType = remove_cvref_t<typename GemmPipeline::AsDataType>;
    using BsDataType = remove_cvref_t<typename GemmPipeline::BsDataType>;
    using EDataType  = remove_cvref_t<typename EpiloguePipeline::ODataType>;
    using DsDataType = remove_cvref_t<typename EpiloguePipeline::DsDataType>;

    /// @brief  ALayout and ADataType are expected to be a tuple, not a scalar.
    static_assert(is_detected<is_tuple, AsLayout>::value &&
                      is_detected<is_tuple, AsDataType>::value,
                  "ALayout and ADataType must be a tuple.");

    /// @brief  BLayout and BDataType are expected to be a tuple, not a scalar.
    static_assert(is_detected<is_tuple, BsLayout>::value &&
                      is_detected<is_tuple, BsDataType>::value,
                  "BLayout and BDataType must be a tuple.");

    /// @brief  CLayout and EDataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, CLayout>::value &&
                      !is_detected<is_tuple, EDataType>::value,
                  "CLayout and EDataType must be a scalar.");

    /// @brief  DsLayout and DsDataType are expected to be tuple, not a scalar.
    static_assert(is_detected<is_tuple, DsLayout>::value &&
                      is_detected<is_tuple, DsDataType>::value &&
                      DsLayout::size() == DsDataType::size() && DsLayout::size() > 0,
                  "DsLayout and DsDataType must be tuples and must have the same size.");

    /// @brief The sizes of NumATensor, NumBTensor and NumDTensor is set by the user."
    static constexpr index_t NumATensor = AsDataType::size();
    static constexpr index_t NumBTensor = BsDataType::size();
    static constexpr index_t NumDTensor = DsDataType::size();

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;
    using DDataType = remove_cvref_t<std::tuple_element_t<0, DsDataType>>;

    CK_TILE_HOST static auto GetName() -> const std::string
    {
        return UniversalGemmKernel::GetName();
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch) -> dim3
    {
        return UniversalGemmKernel::GridSize(M, N, KBatch);
    }

    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        return UniversalGemmKernel::MaxOccupancyGridSize(s);
    }

    CK_TILE_HOST static constexpr auto BlockSize() -> dim3
    {
        return UniversalGemmKernel::BlockSize();
    }

    CK_TILE_HOST static constexpr auto
    MakeKernelArgs(const GemmMultiABDHostArgs<NumATensor, NumBTensor, NumDTensor>& hostArgs) ->
        typename UniversalGemmKernel::KernelArgs
    {
        /// @brief  Universal GEMM requires array objects and corresponding stride information for
        /// matrices A, B, and D.
        return UniversalGemmKernel::MakeKernelArgs(
            UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>(hostArgs.as_ptr,
                                                                      hostArgs.bs_ptr,
                                                                      hostArgs.ds_ptr,
                                                                      hostArgs.e_ptr,
                                                                      hostArgs.k_batch,
                                                                      hostArgs.M,
                                                                      hostArgs.N,
                                                                      hostArgs.K,
                                                                      hostArgs.stride_As,
                                                                      hostArgs.stride_Bs,
                                                                      hostArgs.stride_Ds,
                                                                      hostArgs.stride_E));
    }

    CK_TILE_HOST static auto
    IsSupportedArgument(const typename UniversalGemmKernel::KernelArgs& kargs) -> bool
    {
        // Currently MultiABD kernel doesn't support k_batch > 1
        if(kargs.k_batch > 1)
        {
            return false;
        }

        return UniversalGemmKernel::IsSupportedArgument(kargs);
    }

    CK_TILE_DEVICE auto operator()(typename UniversalGemmKernel::KernelArgs kargs) const -> void
    {
        UniversalGemmKernel{}.template operator()(kargs);
    }
};
} // namespace ck_tile
