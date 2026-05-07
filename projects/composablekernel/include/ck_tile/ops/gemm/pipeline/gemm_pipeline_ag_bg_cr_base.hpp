// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmPipelineAgBgCrImplBase
{
    using AsDataType     = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType     = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using AsLayout       = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout       = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ADataType   = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
    using ALayout     = remove_cvref_t<std::tuple_element_t<number<0>{}, AsLayout>>;
    using BInDataType = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;
    using BDataType =
        std::conditional_t<std::is_same_v<BInDataType, pk_fp4_raw_t>, ADataType, BInDataType>;
    using BLayout = remove_cvref_t<std::tuple_element_t<number<0>{}, BsLayout>>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;
#if defined(__gfx950__)
    // The combination of pk_int4_t and transposed loading causes compilation errors.
    // Therefore do not use transposed loading in this case.
    // Also, transpose load (ds_read_tr) requires specific tile distribution patterns
    // that only work for certain K warp tile sizes based on data type size:
    // - For 1-byte types (fp8/bf8): K warp tile <= 64
    // - For 2-byte types (fp16/bf16): K warp tile <= 32
    static constexpr bool is_a_load_tr = []() {
        using WarpTile                  = typename BlockGemmShape::WarpTile;
        constexpr index_t kKWarpTile    = WarpTile::at(number<2>{});
        constexpr index_t kMaxKWarpTile = (sizeof(ADataType) == 1) ? 64 : 32;
        if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            return false;
        else if constexpr(kKWarpTile > kMaxKWarpTile)
            return false;
        else
            return std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
    }();

    static constexpr bool is_b_load_tr = []() {
        using WarpTile                  = typename BlockGemmShape::WarpTile;
        constexpr index_t kKWarpTile    = WarpTile::at(number<2>{});
        constexpr index_t kMaxKWarpTile = (sizeof(BDataType) == 1) ? 64 : 32;
        if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            return false;
        else if constexpr(kKWarpTile > kMaxKWarpTile)
            return false;
        else
            return std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
    }();
#else
    static constexpr bool is_a_load_tr = false;
    static constexpr bool is_b_load_tr = false;
#endif

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    template <typename SrcDataType = void,
              typename DstDataType = void,
              index_t UnaryOpSize  = 8,
              typename DstBlockTile,
              typename SrcTileWindow,
              typename DramTileWindowStep>
    CK_TILE_DEVICE void GlobalPrefetch(DstBlockTile& dst_block_tile,
                                       SrcTileWindow& dram_tile_window,
                                       const DramTileWindowStep& dram_tile_window_step) const
    {
        load_int4_tile<SrcDataType, DstDataType, UnaryOpSize>(dst_block_tile, dram_tile_window);
        move_tile_window(dram_tile_window, dram_tile_window_step);
    }

    template <typename DstBlockWindow, typename SrcTileWindow, typename DramTileWindowStep>
    CK_TILE_DEVICE void GlobalPrefetchAsync(DstBlockWindow& dst_block_window,
                                            SrcTileWindow& dram_tile_window,
                                            const DramTileWindowStep& dram_tile_window_step) const
    {
        async_load_tile(dst_block_window, dram_tile_window);
        move_tile_window(dram_tile_window, dram_tile_window_step);
    }

    template <typename DstTileWindow, typename SrcBlockTile, typename ElementFunction>
    CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                     const SrcBlockTile& src_block_tile,
                                     const ElementFunction& element_func) const
    {
        const auto block_tile_tmp = tile_elementwise_in(element_func, src_block_tile);
        store_tile(lds_tile_window, block_tile_tmp);
    }

    template <typename DstTileWindow, typename SrcBlockTile>
    CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                     const SrcBlockTile& src_block_tile) const
    {
        store_tile(lds_tile_window, src_block_tile);
    }

    template <typename DstBlockTile, typename SrcTileWindow, bool LoadTranspose = false>
    CK_TILE_DEVICE void LocalPrefetch(DstBlockTile& dst_block_tile,
                                      const SrcTileWindow& lds_tile_window,
                                      bool_constant<LoadTranspose> = {}) const
    {
        if constexpr(LoadTranspose)
            dst_block_tile = load_tile_transpose(lds_tile_window);
        else
            load_tile(dst_block_tile, lds_tile_window);
    }

    template <typename OverrideADataType = ADataType, typename OverrideBDataType = BDataType>
    CK_TILE_DEVICE auto GetABLdsTensorViews(void* p_smem) const
    {
        // A tile in LDS
        OverrideADataType* __restrict__ p_a_lds = static_cast<OverrideADataType*>(p_smem);
        constexpr auto a_lds_block_desc =
            Policy::template MakeALdsBlockDescriptor<Problem, OverrideADataType>();
        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        // TODO: LDS alignment should come from Policy!
        constexpr index_t a_lds_block_space_size_aligned = integer_least_multiple(
            sizeof(OverrideADataType) * a_lds_block_desc.get_element_space_size(), 16);

        // B tile in LDS
        OverrideBDataType* __restrict__ p_b_lds = static_cast<OverrideBDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        return make_tuple(std::move(a_lds_block), std::move(b_lds_block));
    }

    template <typename DramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, DramBlockWindowTmp>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto CopyADramWindow(const DramBlockWindowTmp& dram_block_window_tmp,
                                                  const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;

        using YPerTile = std::conditional_t<is_col_major, number<KPerBlock>, number<MPerBlock>>;
        using XPerTile = std::conditional_t<is_col_major, number<MPerBlock>, number<KPerBlock>>;
        // A DRAM tile window for load
        auto a_copy_dram_window = generate_tuple(
            [&](auto idx) {
                return make_tile_window(
                    dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                    make_tuple(YPerTile{}, XPerTile{}),
                    dram_block_window_tmp[number<idx>{}].get_window_origin() + offset,
                    Policy::template MakeADramTileDistribution<Problem>());
            },
            number<DramBlockWindowTmp::size()>{});
        return std::move(a_copy_dram_window);
    }

    template <typename DramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, DramBlockWindowTmp>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto CopyADramWindow(const DramBlockWindowTmp& dram_block_window_tmp,
                                                  const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;

        using YPerTile = std::conditional_t<is_col_major, number<KPerBlock>, number<MPerBlock>>;
        using XPerTile = std::conditional_t<is_col_major, number<MPerBlock>, number<KPerBlock>>;
        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile{}, XPerTile{}),
                             dram_block_window_tmp.get_window_origin() + offset,
                             Policy::template MakeADramTileDistribution<Problem>());

        return std::move(a_copy_dram_window);
    }

    template <typename DramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, DramBlockWindowTmp>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto CopyBDramWindow(const DramBlockWindowTmp& dram_block_window_tmp,
                                                  const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        using YPerTile = std::conditional_t<is_row_major, number<KPerBlock>, number<NPerBlock>>;
        using XPerTile = std::conditional_t<is_row_major, number<NPerBlock>, number<KPerBlock>>;
        // A DRAM tile window for load
        auto a_copy_dram_window = generate_tuple(
            [&](auto idx) {
                return make_tile_window(
                    dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                    make_tuple(YPerTile{}, XPerTile{}),
                    dram_block_window_tmp[number<idx>{}].get_window_origin() + offset,
                    Policy::template MakeBDramTileDistribution<Problem>());
            },
            number<DramBlockWindowTmp::size()>{});
        return std::move(a_copy_dram_window);
    }

    template <typename DramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, DramBlockWindowTmp>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto CopyBDramWindow(const DramBlockWindowTmp& dram_block_window_tmp,
                                                  const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        using YPerTile = std::conditional_t<is_row_major, number<KPerBlock>, number<NPerBlock>>;
        using XPerTile = std::conditional_t<is_row_major, number<NPerBlock>, number<KPerBlock>>;
        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile{}, XPerTile{}),
                             dram_block_window_tmp.get_window_origin() + offset,
                             Policy::template MakeBDramTileDistribution<Problem>());

        return std::move(a_copy_dram_window);
    }

    template <typename ALdsTensorView, typename ALdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto MakeALdsWindows(const ALdsTensorView& a_lds_block_view,
                                                  const ALdsLoadTileDistr&) const
    {
        auto a_lds_shape = []() {
            if constexpr(is_a_load_tr)
                return make_tuple(number<KPerBlock>{}, number<MPerBlock>{});
            else
                return make_tuple(number<MPerBlock>{}, number<KPerBlock>{});
        }();

        auto a_copy_lds_window = make_tile_window(a_lds_block_view, a_lds_shape, {0, 0});

        auto a_lds_load_tile_distr = []() {
            if constexpr(is_a_load_tr)
                return make_static_tile_distribution(
                    typename InputTileDistributionTraits<
                        typename ALdsLoadTileDistr::DstrEncode,
                        typename Problem::ADataType>::TransposedDstrEncode{});
            else
                return ALdsLoadTileDistr{};
        }();

        auto a_lds_gemm_window =
            make_tile_window(a_lds_block_view, a_lds_shape, {0, 0}, a_lds_load_tile_distr);

        return make_tuple(std::move(a_copy_lds_window), std::move(a_lds_gemm_window));
    }

    template <
        typename ADramBlockWindowTmp,
        typename ALdsTensorView,
        typename ALdsLoadTileDistr,
        typename std::enable_if_t<!is_detected<is_tuple, ALdsTensorView>::value, bool>* = nullptr>
    CK_TILE_DEVICE constexpr auto GetAWindows(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                              const ALdsTensorView& a_lds_block_view,
                                              const ALdsLoadTileDistr& a_lds_load_tile_distr,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        // A DRAM tile window for load
        auto a_copy_dram_window = CopyADramWindow(a_dram_block_window_tmp, offset);

        // Create LDS windows
        auto [a_copy_lds_window, a_lds_gemm_window] =
            MakeALdsWindows(a_lds_block_view, a_lds_load_tile_distr);

        return make_tuple(std::move(a_copy_dram_window),
                          std::move(a_copy_lds_window),
                          std::move(a_lds_gemm_window));
    }

    // Unified GetAWindows that supports 1, 2, or 3 LDS buffers
    template <typename ADramBlockWindowTmp,
              typename ALdsTensorViewsTuple,
              typename ALdsLoadTileDistr,
              typename std::enable_if_t<is_detected<is_tuple, ALdsTensorViewsTuple>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto GetAWindows(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                              const ALdsTensorViewsTuple& a_lds_block_views_tuple,
                                              const ALdsLoadTileDistr& a_lds_load_tile_distr,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        // A DRAM tile window for load
        auto a_copy_dram_window = CopyADramWindow(a_dram_block_window_tmp, offset);

        // Create LDS windows for each buffer
        constexpr index_t num_buffers = ALdsTensorViewsTuple::size();
        auto a_lds_windows            = generate_tuple(
            [&](auto i) {
                return MakeALdsWindows(a_lds_block_views_tuple[i], a_lds_load_tile_distr);
            },
            number<num_buffers>{});

        // Return: (dram_window, lds_windows_tuple)
        // lds_windows_tuple[i] = (copy_lds_window_i, lds_gemm_window_i)
        return make_tuple(std::move(a_copy_dram_window), std::move(a_lds_windows));
    }

    template <typename BLdsTensorView, typename BLdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto MakeBLdsWindows(const BLdsTensorView& b_lds_block_view,
                                                  const BLdsLoadTileDistr&) const
    {
        auto b_lds_shape = []() {
            if constexpr(is_b_load_tr)
                return make_tuple(number<KPerBlock>{}, number<NPerBlock>{});
            else
                return make_tuple(number<NPerBlock>{}, number<KPerBlock>{});
        }();

        auto b_copy_lds_window = make_tile_window(b_lds_block_view, b_lds_shape, {0, 0});

        using BLdsDataType =
            std::conditional_t<std::is_same_v<typename Problem::BDataType, pk_fp4_raw_t>,
                               typename Problem::ADataType,
                               typename Problem::BDataType>;

        auto b_lds_load_tile_distr = []() {
            if constexpr(is_b_load_tr)
                return make_static_tile_distribution(
                    typename InputTileDistributionTraits<typename BLdsLoadTileDistr::DstrEncode,
                                                         BLdsDataType>::TransposedDstrEncode{});

            else
                return BLdsLoadTileDistr{};
        }();

        auto b_lds_gemm_window =
            make_tile_window(b_lds_block_view, b_lds_shape, {0, 0}, b_lds_load_tile_distr);

        return make_tuple(std::move(b_copy_lds_window), std::move(b_lds_gemm_window));
    }

    template <
        typename BDramBlockWindowTmp,
        typename BLdsTensorView,
        typename BLdsLoadTileDistr,
        typename std::enable_if_t<!is_detected<is_tuple, BLdsTensorView>::value, bool>* = nullptr>
    CK_TILE_DEVICE constexpr auto GetBWindows(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                              const BLdsTensorView& b_lds_block_view,
                                              const BLdsLoadTileDistr& b_lds_load_tile_distr,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        // A DRAM tile window for load
        auto b_copy_dram_window = CopyBDramWindow(b_dram_block_window_tmp, offset);

        // Create LDS windows
        auto [b_copy_lds_window, b_lds_gemm_window] =
            MakeBLdsWindows(b_lds_block_view, b_lds_load_tile_distr);

        return make_tuple(std::move(b_copy_dram_window),
                          std::move(b_copy_lds_window),
                          std::move(b_lds_gemm_window));
    }

    // Unified GetBWindows that supports 1, 2, or 3 LDS buffers
    template <typename BDramBlockWindowTmp,
              typename BLdsTensorViewsTuple,
              typename BLdsLoadTileDistr,
              typename std::enable_if_t<is_detected<is_tuple, BLdsTensorViewsTuple>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto GetBWindows(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                              const BLdsTensorViewsTuple& b_lds_block_views_tuple,
                                              const BLdsLoadTileDistr& b_lds_load_tile_distr,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        // B DRAM tile window for load
        auto b_copy_dram_window = CopyBDramWindow(b_dram_block_window_tmp, offset);

        // Create LDS windows for each buffer
        constexpr index_t num_buffers = BLdsTensorViewsTuple::size();
        auto b_lds_windows            = generate_tuple(
            [&](auto i) {
                return MakeBLdsWindows(b_lds_block_views_tuple[i], b_lds_load_tile_distr);
            },
            number<num_buffers>{});

        // Return: (dram_window, lds_windows_tuple)
        // lds_windows_tuple[i] = (copy_lds_window_i, lds_gemm_window_i)
        return make_tuple(std::move(b_copy_dram_window), std::move(b_lds_windows));
    }
};

} // namespace ck_tile
