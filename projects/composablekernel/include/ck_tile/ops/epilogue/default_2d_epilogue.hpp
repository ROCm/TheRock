// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

// this epilogue just store out a M*N matrix, row major

template <typename AccDataType_,
          typename ODataType_,
          bool kPadM_,
          bool kPadN_,
          bool UseRawStore_ = true>
struct Default2DEpilogueProblem
{
    using AccDataType                   = remove_cvref_t<AccDataType_>;
    using ODataType                     = remove_cvref_t<ODataType_>;
    static constexpr bool kPadM         = kPadM_;
    static constexpr bool kPadN         = kPadN_;
    static constexpr bool UseRawStore   = UseRawStore_;
    static constexpr index_t NumDTensor = 0;
};

template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename DsLayout_,
          typename CLayout_,
          typename CDElementwise_,
          index_t kM_,
          index_t kN_,
          bool kPadM_,
          bool kPadN_,
          index_t kMPerXdl_,
          index_t kNPerXdl_,
          index_t kKPerXdl_,
          bool isCTransposed_,
          bool UseRawStore_ = true>
struct DefaultGemm2DEpilogueProblem
    : public Default2DEpilogueProblem<AccDataType_, ODataType_, kPadM_, kPadN_, UseRawStore_>
{
    using AsDataType                       = remove_cvref_t<AsDataType_>;
    using BsDataType                       = remove_cvref_t<BsDataType_>;
    using CLayout                          = remove_cvref_t<CLayout_>;
    using DsDataType                       = remove_cvref_t<DsDataType_>;
    using CDElementwise                    = remove_cvref_t<CDElementwise_>;
    using DsLayout                         = remove_cvref_t<DsLayout_>;
    static constexpr index_t kMPerBlock    = kM_;
    static constexpr index_t kNPerBlock    = kN_;
    static constexpr index_t kMPerXdl      = kMPerXdl_;
    static constexpr index_t kNPerXdl      = kNPerXdl_;
    static constexpr index_t kKPerXdl      = kKPerXdl_;
    static constexpr index_t isCTransposed = isCTransposed_;

    static constexpr index_t NumDTensor = DsDataType::size();

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
};

template <typename Problem_, typename Policy_ = void>
struct Default2DEpilogue
{
    using Problem                     = remove_cvref_t<Problem_>;
    using AccDataType                 = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                   = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM       = Problem::kPadM;
    static constexpr bool kPadN       = Problem::kPadN;
    static constexpr bool UseRawStore = Problem::UseRawStore;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    // TODO: this function assume store out vector size is the same as OAccTile last dimension size
    //       how do we fix this ?
    template <typename ODramWindowTmp, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* = nullptr) const
    {
        constexpr bool is_partition_index =
            std::is_convertible_v<decltype(ds_dram_windows),
                                  decltype(get_partition_index(
                                      o_acc_tile.get_tile_distribution()))>;

        const auto storeOrUpdateTile = [&](const auto& o_tile) {
            // TODO: this is ugly
            if constexpr(UseRawStore && (kPadM || kPadN))
            {
                // FIXME?
                // if constexpr(decltype(o_dram_window_tmp.get_bottom_tensor_view())::DstInMemOp ==
                // memory_operation_enum::set)
                if constexpr(true)
                {
                    if constexpr(is_partition_index)
                    {
                        store_tile_raw(o_dram_window_tmp,
                                       cast_tile<ODataType>(o_tile),
                                       /*partition_index=*/ds_dram_windows);
                    }
                    else
                    {
                        store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_tile));
                    }
                }
                else
                {
                    update_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_tile));
                }
                buffer_store_fence();
            }
            else
            {
                // FIXME?
                // if constexpr(decltype(o_dram_window_tmp.get_bottom_tensor_view())::DstInMemOp ==
                // memory_operation_enum::set)
                if constexpr(true)
                {
                    if constexpr(is_partition_index)
                    {
                        store_tile(o_dram_window_tmp,
                                   cast_tile<ODataType>(o_tile),
                                   /*partition_index=*/ds_dram_windows);
                    }
                    else
                    {
                        store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_tile));
                    }
                }
                else
                {
                    if constexpr(is_partition_index)
                    {
                        update_tile(o_dram_window_tmp,
                                    cast_tile<ODataType>(o_tile),
                                    /*partition_index=*/ds_dram_windows);
                    }
                    else
                    {
                        update_tile(o_dram_window_tmp, cast_tile<ODataType>(o_tile));
                    }
                }
            }
        };

        if constexpr(!std::is_same_v<DsDramWindows, std::nullptr_t> && !is_partition_index &&
                     Problem::NumDTensor >= 1)
        {
            using elementwise_result_t = decltype(load_tile(
                make_tile_window(ds_dram_windows[number<0>{}].get_bottom_tensor_view(),
                                 make_tuple(Problem::kMPerBlock, Problem::kNPerBlock),
                                 ds_dram_windows[number<0>{}].get_window_origin(),
                                 o_acc_tile.get_tile_distribution())));

            elementwise_result_t elementwise_result;

            const auto d_tensor_tuple = generate_tuple(
                [&](auto idx) {
                    const auto d_tile_window =
                        make_tile_window(ds_dram_windows[idx], o_acc_tile.get_tile_distribution());
                    return load_tile(d_tile_window);
                },
                number<Problem::NumDTensor>{});

            const auto c_d_tuple = concat_tuple_of_reference(
                tie(elementwise_result, o_acc_tile),
                generate_tie([&](auto idx) -> const auto& { return d_tensor_tuple[idx]; },
                             number<Problem::NumDTensor>{}));

            tile_elementwise_inout_unpack(typename Problem::CDElementwise{}, c_d_tuple);

            storeOrUpdateTile(elementwise_result);
        }
        else
        {
            storeOrUpdateTile(o_acc_tile);
        }
    }
};

template <typename Problem_, typename Policy_ = void>
struct DefaultGemm2DEpilogue : public Default2DEpilogue<Problem_, Policy_>
{
    using Problem                          = remove_cvref_t<Problem_>;
    using AsDataType                       = remove_cvref_t<typename Problem::AsDataType>;
    using BsDataType                       = remove_cvref_t<typename Problem::BsDataType>;
    using AccDataType                      = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                        = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool ADataTypeIsTuple = is_detected<is_tuple, AsDataType>::value;
    static constexpr bool BDataTypeIsTuple = is_detected<is_tuple, BsDataType>::value;

    using AsDataTypeTuple = std::conditional_t<ADataTypeIsTuple,
                                               remove_cvref_t<AsDataType>,
                                               remove_cvref_t<tuple<AsDataType>>>;

    using BsDataTypeTuple = std::conditional_t<BDataTypeIsTuple,
                                               remove_cvref_t<BsDataType>,
                                               remove_cvref_t<tuple<BsDataType>>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataTypeTuple>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataTypeTuple>>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;

    using DsDataType                       = remove_cvref_t<typename Problem::DsDataType>;
    using DsLayout                         = remove_cvref_t<typename Problem::DsLayout>;
    using CDElementwise                    = remove_cvref_t<typename Problem::CDElementwise>;
    using CLayout                          = remove_cvref_t<typename Problem::CLayout>;
    static constexpr index_t kMPerXdl      = Problem::kMPerXdl;
    static constexpr index_t kNPerXdl      = Problem::kNPerXdl;
    static constexpr index_t kKPerXdl      = Problem::kKPerXdl;
    static constexpr index_t isCTransposed = Problem::isCTransposed;

    using WG = WarpGemmDispatcher<ADataType,
                                  BTypeToUse,
                                  AccDataType,
                                  kMPerXdl,
                                  kNPerXdl,
                                  kKPerXdl,
                                  isCTransposed>;

    using CWarpDstr = typename WG::CWarpDstr;

    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if constexpr(isCTransposed)
            {
                // In this case each thread has multiple consecutive elements in
                // N dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
            else
            {
                // In this case each thread has just a single item in Ndim
                return (WG::WarpGemmAttribute::Impl::kCNLane *
                        WG::WarpGemmAttribute::Impl::kBNBlock) /
                       WG::kN;
            }
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            if constexpr(isCTransposed)
            {
                // In this case each thread has just a single item in Mdim
                return (WG::WarpGemmAttribute::Impl::kCNLane *
                        WG::WarpGemmAttribute::Impl::kAMBlock) /
                       WG::kN;
            }
            else
            {
                // In this case each thread has multiple consecutive elements in
                // M dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeD([[maybe_unused]] number<I> index)
    {
        return GetVectorSizeC();
    }
};

} // namespace ck_tile
