// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/utility.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

/**
 * @brief This class provides tile (windowed) view and access to the device memory.
 *
 * @note This tile window does not support single issue you need to use tile_window_linear
 *       structure for this purpose
 *
 * @tparam BottomTensorView_        Class describing & holding device tensor memory.
 * @tparam WindowLengths_           Spatial sizes of windowed view on tensor.
 * @tparam StaticTileDistribution_  Thread distribution (mapping) into Tile dimensions
 * @tparam NumCoord                 TBD
 */
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename StaticPageIndexArray_,
          typename StaticValidArray_,
          index_t HsGatherDim = 0,
          index_t NumCoord    = 1,
          index_t YsGatherDim = 0>
struct tile_scatter_gather
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<StaticTileDistribution_>;
    using PageIdxArray     = remove_cvref_t<StaticPageIndexArray_>;
    using ValidArray       = remove_cvref_t<StaticValidArray_>;
    using WindowAdaptor    = typename TileDstr::PsYs2XsAdaptor;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    using DataType = remove_cvref_t<typename BottomTensorView::DataType>;

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::get_num_of_top_dimension();
    static constexpr index_t NDimBottomTensor     = BottomTensorDesc::get_num_of_dimension();

    static constexpr index_t NDimP = TileDstr::get_num_of_dimension_p();
    static constexpr index_t NDimY = TileDstr::get_num_of_dimension_y();

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static_assert(NumCoord == 1);

    // TODO: check WindowLengths and StaticTileDistribution are consistent

    static_assert(ck_tile::is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileDstr::is_static(), "wrong!");

    static_assert(NDimBottomTensor == WindowAdaptor::get_num_of_bottom_dimension(),
                  "wrong! inconsistent # of diemsnions");

    using AdaptorTopIndex   = array<index_t, NDimWindowAdaptorTop>;
    using BottomTensorIndex = array<index_t, NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    struct load_store_traits
    {
        private:
        static constexpr auto get_vector_dim_y_scalar_per_vector()
        {
            const auto [ys_vector_lengths, ys_vector_strides] =
                tile_scatter_gather::get_window_adaptor_ys_safe_vector_length_strides();

            index_t VectorDimY_      = 0;
            index_t ScalarPerVector_ = 1;

            for(index_t i = 0; i < NDimY; ++i)
            {
                if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector_)
                {
                    ScalarPerVector_ = ys_vector_lengths[i];
                    VectorDimY_      = i;
                }
            }

            return make_tuple(VectorDimY_, ScalarPerVector_);
        }

        public:
        static constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;
        static constexpr index_t VectorDimY = get_vector_dim_y_scalar_per_vector().template at<0>();
        static constexpr index_t ScalarPerVector =
            get_vector_dim_y_scalar_per_vector().template at<1>();

        // using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;
        // using vector_t      = typename vector_type_t::type;
        using vector_t = thread_buffer<DataType, ScalarPerVector / PackedSize>;

        private:
        static constexpr auto scalars_per_access_ = [] {
            constexpr auto scalars_per_access_arr = generate_array(
                [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, number<NDimY>{});

            /// TODO: add non-automatic storage argument support to macro TO_SEQUENCE()
            constexpr auto NDimY_ = NDimY;

            return TO_SEQUENCE(scalars_per_access_arr, NDimY_);
        }();

        static constexpr auto get_space_filling_curve()
        {
            constexpr auto tile_dstr = TileDstr{};

            constexpr auto thread_tensor_lengths_ys =
                to_sequence(tile_dstr.get_ys_to_d_descriptor().get_lengths());

            // FIXME: need logic to judge dim access order
            using DimAccessOrder = typename arithmetic_sequence_gen<0, NDimY, 1>::type;

            return space_filling_curve<decltype(thread_tensor_lengths_ys),
                                       DimAccessOrder,
                                       decltype(scalars_per_access_)>{};
        }

        public:
        using SFC_Ys = decltype(get_space_filling_curve());

        static constexpr index_t NumAccess = SFC_Ys::get_num_of_access();

        static_assert(0 < NumAccess, "Wrong! NumAccess should be larger than 0");
        static_assert(NumAccess % NumCoord == 0, "wrong! # of access is not divisible by NumCoord");
    };

    static constexpr index_t NumAccessPerCoord = load_store_traits::NumAccess / NumCoord;

    CK_TILE_DEVICE constexpr tile_scatter_gather() = default;

    CK_TILE_DEVICE constexpr tile_scatter_gather(const BottomTensorView& bottom_tensor_view,
                                                 const WindowLengths& window_lengths,
                                                 const BottomTensorIndex& window_origin,
                                                 const TileDstr& tile_distribution,
                                                 const PageIdxArray& page_idx,
                                                 const ValidArray& valids)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin},
          tile_dstr_{tile_distribution},
          page_idx_{page_idx},
          valids_{valids},
          pre_computed_coords_{}
    {
#if 0 // debug
      // TODO: this use more register for FA, but less register for GEMM
      // need investigation
      // only support warp-tile and block-tile
        static_assert(NDimP == 1 or NDimP == 2, "wrong!");

        WindowAdaptorCoord window_adaptor_thread_coord_tmp;

        if constexpr(NDimP == 1)
        {
            window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
                tile_distribution.get_ps_ys_to_xs_adaptor(), AdaptorTopIndex{get_lane_id(), 0});
        }
        else if constexpr(NDimP == 2)
        {
            window_adaptor_thread_coord_tmp =
                make_tensor_adaptor_coordinate(tile_distribution.get_ps_ys_to_xs_adaptor(),
                                               AdaptorTopIndex{get_warp_id(), get_lane_id(), 0});
        }
#else
        // TODO: this use less register for FA, but more register for GEMM
        // need investigation
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(get_partition_index(tile_distribution), array<index_t, NDimY>{0}));
#endif

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();
        bottom_tensor_thread_origin_idx_tmp(HsGatherDim) = 0;
        const auto bottom_tensor_thread_coord_tmp        = make_tensor_coordinate(
            bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = load_store_traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}), idx_diff_ys);

            move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            pre_computed_coords_(iCoord) =
                make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });
        if constexpr(BottomTensorView::buffer_view::get_address_space() ==
                     address_space_enum::global)
        {
            auto partition_index = get_partition_index(tile_distribution);

            auto use_lane_id_0                              = partition_index;
            use_lane_id_0[1]                                = 0;
            const auto window_adaptor_thread_coord_tmp_warp = make_tensor_adaptor_coordinate(
                tile_distribution.get_ps_ys_to_xs_adaptor(),
                container_concat(use_lane_id_0, array<index_t, NDimY>{0}));

            BottomTensorIndex bottom_tensor_thread_origin_idx_tmp_warp =
                window_origin + window_adaptor_thread_coord_tmp_warp.get_bottom_index();
            bottom_tensor_thread_origin_idx_tmp_warp(HsGatherDim) = 0;
            const auto bottom_tensor_thread_coord_tmp_warp =
                make_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                                       bottom_tensor_thread_origin_idx_tmp_warp);

            // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
            // future load/store() calls (might allocate more registers)
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp_warp;
                auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp_warp;

                constexpr auto idx_diff_ys =
                    SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

                constexpr auto idx_diff_ps_ys = container_concat(
                    generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                    idx_diff_ys);

                move_window_adaptor_and_bottom_tensor_thread_coordinate(
                    window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                pre_computed_warp_coords_(iCoord) =
                    make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
            });
        }
    }

    CK_TILE_DEVICE static constexpr index_t get_num_of_dimension() { return NDimBottomTensor; }

    CK_TILE_DEVICE static constexpr bool has_static_tile_distribution()
    {
        return TileDstr::is_static();
    }

    CK_TILE_DEVICE constexpr auto get_window_lengths() const { return window_lengths_; }

    CK_TILE_DEVICE constexpr auto get_tile_distribution() const { return tile_dstr_; }

    CK_TILE_DEVICE constexpr auto get_bottom_tensor_view() const { return bottom_tensor_view_; }

    CK_TILE_DEVICE constexpr auto get_window_origin() const { return window_origin_; }

    CK_TILE_DEVICE constexpr void
    set_bottom_tensor_view_data_ptr(typename BottomTensorView::DataType* data)
    {
        bottom_tensor_view_.buf_.p_data_ = data;
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    template <typename ATopIndex>
    CK_TILE_DEVICE void move_window_adaptor_and_bottom_tensor_thread_coordinate(
        WindowAdaptorCoord& window_adaptor_thread_coord,
        BottomTensorCoord& bottom_tensor_thread_coord,
        const ATopIndex& idx_diff_adaptor_top) const
    {
        array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                       window_adaptor_thread_coord,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                               bottom_tensor_thread_coord,
                               idx_diff_adaptor_bottom);
    }

    // return vector dimension among [y0, y1, ...]
    CK_TILE_DEVICE static constexpr auto get_window_adaptor_ys_safe_vector_length_strides()
    {
        // bottom tensor top dimension vector lengths and strides
        const auto [bottom_tensor_top_dim_vector_lengths, bottom_tensor_top_dim_vector_strides] =
            BottomTensorDesc::get_top_dimension_safe_vector_length_strides();

        // window vector lengths/strides
        const auto window_adaptor_bottom_dim_vector_lengths = bottom_tensor_top_dim_vector_lengths;
        const auto window_adaptor_bottom_dim_vector_strides = bottom_tensor_top_dim_vector_strides;

        // window adaptor [p0, p1, ..., y0, y1, ...]
        array<index_t, WindowAdaptor::get_num_of_hidden_dimension()> window_adaptor_vector_lengths{
            -1};
        array<index_t, WindowAdaptor::get_num_of_hidden_dimension()> window_adaptor_vector_strides{
            -1};

        constexpr auto window_adaptor_bottom_dims =
            WindowAdaptor::get_bottom_dimension_hidden_ids();

        set_container_subset(window_adaptor_vector_lengths,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_lengths);
        set_container_subset(window_adaptor_vector_strides,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_strides);

        const auto [window_adaptor_ps_ys_vector_lengths, window_adaptor_ps_ys_vector_strides] =
            WindowAdaptor{}.get_top_dimension_safe_vector_length_strides(
                window_adaptor_vector_lengths, window_adaptor_vector_strides);

        // [y0, y1, ...]
        constexpr auto y_dims = typename arithmetic_sequence_gen<TileDstr::get_num_of_dimension_p(),
                                                                 NDimWindowAdaptorTop,
                                                                 1>::type{};

        return make_tuple(get_container_subset(window_adaptor_ps_ys_vector_lengths, y_dims),
                          get_container_subset(window_adaptor_ps_ys_vector_strides, y_dims));
    }

    CK_TILE_DEVICE constexpr auto get_num_of_access() const { return load_store_traits::NumAccess; }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load(number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        constexpr auto tile_dstr = TileDstr{};
        auto dst_tensor          = make_static_distributed_tensor<DataType>(tile_dstr);
        load(dst_tensor, number<i_access_unsupport_>{}, bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    template <typename DistributedTensor,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE auto load(DistributedTensor& dst_tensor,
                             number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        using Traits   = load_store_traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = idx_ys_start[number<YsGatherDim>{}];
                const auto page_offset      = page_idx_[idx_gather];

                // read from bottom tensor
                const vector_t vec_value = [&]() {
                    if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                    {
                        return get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                            bottom_tensor_thread_coord,
                            page_offset,
                            bool_constant<oob_conditional_check>{});
                    }
                    else
                    {
                        return get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                            bottom_tensor_thread_coord,
                            page_offset,
                            valids_[idx_gather],
                            bool_constant<oob_conditional_check>{});
                    }
                }();
#if 1
                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    dst_tensor.get_thread_buffer().template at<d>() =
                        vec_value.template get_as<DataType>()[j / Traits::PackedSize];
                });
#else
                constexpr index_t d =
                    tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
                static_assert(d % Traits::ScalarPerVector == 0);

                dst_tensor.get_thread_buffer().template get_as<vector_t>()(
                    number<d / Traits::ScalarPerVector>{}) = bit_cast<vector_t>(vec_value);
#endif
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return i == YsGatherDim ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE auto async_load(LdsTileWindow_&& lds_tile,
                                   number<i_access_unsupport_>          = {},
                                   bool_constant<oob_conditional_check> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;
        using Traits        = load_store_traits;
        using vector_t      = typename Traits::vector_t;
        using SFC_Ys        = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // Precompute invariant values outside loops
        const auto window_origin       = lds_tile.get_window_origin();
        const auto& bottom_tensor_view = lds_tile.get_bottom_tensor_view();
        const auto& tensor_descriptor  = bottom_tensor_view.get_tensor_descriptor();
        auto smem_base_ptr             = bottom_tensor_view.get_buffer_view().p_data_;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            auto lds_window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto lds_bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // Use precomputed window origin
                auto lds_bottom_tensor_thread_idx =
                    window_origin + lds_window_adaptor_thread_coord.get_bottom_index();
                // Use precomputed tensor descriptor
                const auto lds_coord =
                    make_tensor_coordinate(tensor_descriptor, lds_bottom_tensor_thread_idx);
                // Calculate SMEM address using base pointer
                CK_TILE_LDS_ADDR LdsDataType* smem = smem_base_ptr + lds_coord.get_offset();

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = idx_ys_start[number<YsGatherDim>{}];
                const auto page_offset      = page_idx_[idx_gather];

                // merge page_offset into bottom_coord
                auto mixed_bottom_thread_coord = bottom_tensor_thread_coord;
                mixed_bottom_thread_coord.get_hidden_index()[number<0>{}] += page_offset;

                // read from bottom tensor
                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                    this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                        smem,
                        mixed_bottom_thread_coord,
                        number<0>{},
                        bool_constant<oob_conditional_check>{});
                else
                    this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                        smem,
                        mixed_bottom_thread_coord,
                        number<0>{},
                        valids_[idx_gather],
                        bool_constant<oob_conditional_check>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return i == YsGatherDim ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);
                    // lds_diff doesn't need to mask the difference of the gather-dim.
                    constexpr auto lds_idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        lds_window_adaptor_thread_coord,
                        lds_bottom_tensor_thread_coord,
                        lds_idx_diff_ps_ys);
                }
            });
        });
    }

    // TODO: currently async load only implemented in inline asm
    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false>
    CK_TILE_DEVICE auto async_load_raw(LdsTileWindow_&& lds_tile,
                                       number<i_access_unsupport_>          = {},
                                       bool_constant<oob_conditional_check> = {},
                                       bool_constant<pre_nop>               = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        // using LdsTensorView = typename LdsTileWindow::BottomTensorView;
        using LdsDataType = typename LdsTileWindow::DataType;
        // using LdsDescriptor = typename LdsTileWindow::BottomTensorDesc;

        // issues * warps * lanes
        static_assert(LdsTileWindow::get_num_of_dimension() == 3); // TODO: hard coded

        const index_t size_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{})) *
            sizeof(LdsDataType);

        const index_t size_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t size_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();
        m0_set_with_memory(
            amd_wave_read_first_lane(m0_init_value)); // This should be wave independent

        using Traits = load_store_traits;

        // using vector_type_t = typename Traits::vector_type_t;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        LdsDataType* smem = lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess  = number<iCoord * NumAccessPerCoord + iCoordAccess>{};
                constexpr auto pre_nop_ = [&]() {
                    if constexpr(pre_nop && iCoord == 0 && iCoordAccess == 0)
                        return bool_constant<true>{};
                    else
                        return bool_constant<false>{};
                }();

                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = idx_ys_start[number<YsGatherDim>{}];
                const auto page_offset      = page_idx_[idx_gather];

                // read from bottom tensor
                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                {
                    get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                        smem, bottom_tensor_thread_coord, page_offset, 0, pre_nop_);
                }
                else
                {
                    get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                        smem,
                        bottom_tensor_thread_coord,
                        page_offset,
                        valids_[idx_gather],
                        0,
                        pre_nop_);
                }

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return i == YsGatherDim ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                    m0_inc_with_memory(size_per_issue);
                }
            });
        });
    }

    // TODO: fix with swizzle
    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool static_move_ys         = false,
              typename = std::enable_if_t<std::is_class_v<remove_cvref_t<LdsTileWindow_>>>>
    CK_TILE_DEVICE void async_load_with_offset(index_t offset,
                                               LdsTileWindow_&& lds_tile,
                                               number<i_access_unsupport_>          = {},
                                               bool_constant<oob_conditional_check> = {},
                                               bool_constant<static_move_ys>        = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        using Traits = load_store_traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        // Precompute invariant values outside loops
        const auto window_origin       = lds_tile.get_window_origin();
        const auto& bottom_tensor_view = lds_tile.get_bottom_tensor_view();
        const auto& tensor_descriptor  = bottom_tensor_view.get_tensor_descriptor();
        auto lds_base_ptr              = bottom_tensor_view.get_buffer_view().p_data_;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            auto window_adaptor_warp_coord = pre_computed_warp_coords_[iCoord][I0];
            auto bottom_tensor_warp_coord  = pre_computed_warp_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                constexpr auto idx_ys_offset = [&]() {
                    constexpr auto idx_off_ys = SFC_Ys::get_step_between(number<0>{}, iAccess);
                    constexpr auto adapter_ys_offset = make_tensor_adaptor_coordinate(
                        StaticTileDistribution_{}.get_ps_ys_to_xs_adaptor(),
                        container_concat(array<index_t, NDimP>{0},
                                         to_array<index_t, idx_off_ys.size()>(idx_off_ys)));
                    return adapter_ys_offset.get_bottom_index();
                }();
                const auto lds_ys_offset = [&]() {
                    if constexpr(static_move_ys)
                    {
                        const auto coord_ys_offset =
                            make_tensor_coordinate(tensor_descriptor, idx_ys_offset);
                        return coord_ys_offset.get_offset();
                    }
                    else
                        return 0;
                }();

                // Use precomputed window origin & tensor descriptor
                auto lds_bottom_tensor_thread_idx =
                    window_origin + window_adaptor_warp_coord.get_bottom_index();
                const auto lds_coord =
                    make_tensor_coordinate(tensor_descriptor, lds_bottom_tensor_thread_idx);

                // Calculate SMEM address using base pointer
                CK_TILE_LDS_ADDR LdsDataType* smem = lds_base_ptr +
                                                     lds_coord.get_offset() / Traits::PackedSize +
                                                     lds_ys_offset / Traits::PackedSize;

                const auto dram_ys_offset = [&]() {
                    if constexpr(static_move_ys)
                    {
                        const auto coord_ys_offset = make_tensor_coordinate(
                            this->get_bottom_tensor_view().get_tensor_descriptor(), idx_ys_offset);
                        return coord_ys_offset.get_offset();
                    }
                    else
                        return 0;
                }();

                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = idx_ys_start[number<YsGatherDim>{}];
                const auto page_offset      = page_idx_[idx_gather];

                auto mixed_bottom_thread_coord = bottom_tensor_thread_coord;
                mixed_bottom_thread_coord.get_hidden_index()[number<0>{}] += page_offset;

                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                {
                    this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                        smem,
                        mixed_bottom_thread_coord,
                        offset + dram_ys_offset,
                        bool_constant<oob_conditional_check>{});
                }
                else
                {
                    this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                        smem,
                        mixed_bottom_thread_coord,
                        offset + dram_ys_offset,
                        valids_[idx_gather],
                        bool_constant<oob_conditional_check>{});
                }

                // Move thread coordinate if not last access
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return i == YsGatherDim ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    if constexpr(!static_move_ys)
                        move_window_adaptor_and_bottom_tensor_thread_coordinate(
                            window_adaptor_thread_coord,
                            bottom_tensor_thread_coord,
                            idx_diff_ps_ys);

                    if constexpr(!static_move_ys)
                        move_window_adaptor_and_bottom_tensor_thread_coordinate(
                            window_adaptor_warp_coord, bottom_tensor_warp_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void update(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                               number<i_access_unsupport_>          = {},
                               bool_constant<oob_conditional_check> = {}) const
    {
        using Traits = load_store_traits;

        // using vector_type_t = typename Traits::vector_type_t;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = idx_ys_start[number<0>{}];
                const auto page_offset      = page_idx_[idx_gather];

                // read from distributed tensor
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    vec_value.template get_as<DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                {
                    get_bottom_tensor_view().template update_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        page_offset,
                        vec_value,
                        bool_constant<oob_conditional_check>{});
                }
                else
                {
                    get_bottom_tensor_view().template update_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        page_offset,
                        valids_[idx_gather],
                        vec_value,
                        bool_constant<oob_conditional_check>{});
                }

                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return i == YsGatherDim ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void store(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                              number<i_access_unsupport_>          = {},
                              bool_constant<oob_conditional_check> = {}) const
    {
        using Traits = load_store_traits;

        // using vector_type_t = typename Traits::vector_type_t;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};
        // printf("off %d\n", page_idx_[I0]);
        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = idx_ys_start[number<0>{}];
                const auto page_offset      = page_idx_[idx_gather];

                // printf("idx_ys_start[0], idx_ys_start[1](%d, %d) \n",
                // idx_ys_start[number<0>{}]+0, idx_ys_start[number<1>{}]+0);

                // read from distributed tensor
                // vector_type_t vec;
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;
                    // printf("thread_idx_m: %d j: %d\n", idx_ys[number<0>{}] + 0, 0+j);
                    vec_value.template get_as<DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // const vector_t vec_value = vec.template get_as<vector_t>().template at<0>();

                // write into bottom tensor
                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                {
                    get_bottom_tensor_view().template set_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        page_offset,
                        vec_value,
                        bool_constant<oob_conditional_check>{});
                }
                else
                {
                    get_bottom_tensor_view().template set_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        page_offset,
                        valids_[idx_gather],
                        vec_value,
                        bool_constant<oob_conditional_check>{});
                }

                // printf("coord_offset:%d,   scatter_offset:%d \n",
                // bottom_tensor_thread_coord.get_offset(), offset); move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return i == YsGatherDim ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // move thread's botom tensor coordiante
    // [x0', x1', ... ] ==> [offset]
    // also move window-origin
    CK_TILE_DEVICE void move(const BottomTensorIndex& step)
    {
        window_origin_ += step;
        BottomTensorIndex step_new = step;
        step_new(HsGatherDim)      = 0;
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                                   pre_computed_coords_(iCoord)(I1),
                                   step_new);
        });
        if constexpr(BottomTensorView::buffer_view::get_address_space() ==
                     address_space_enum::global)
        {
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                                       pre_computed_warp_coords_(iCoord)(I1),
                                       step_new);
            });
        }
    }

    CK_TILE_DEVICE void update_page_idx(const PageIdxArray& new_idx) { page_idx_ = new_idx; }

    CK_TILE_DEVICE void update_valids(const ValidArray& new_valids)
    {
        if constexpr(std::is_same_v<ValidArray, std::nullptr_t> == false)
        {
            valids_ = new_valids;
        }
    }

    CK_TILE_DEVICE void update_page_idx_and_valids(const PageIdxArray& new_idx,
                                                   const ValidArray& new_valids)
    {
        update_page_idx(new_idx);
        update_valids(new_valids);
    }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& new_window_origin)
    {
        window_origin_ = new_window_origin;

#if 0 // debug
      // TODO: this use more register for FA, but less register for GEMM
      // need investigation
      // only support warp-tile and block-tile
        static_assert(NDimP == 1 or NDimP == 2, "wrong!");

        WindowAdaptorCoord window_adaptor_thread_coord_tmp;

        if constexpr(NDimP == 1)
        {
            window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
                tile_dstr_.get_ps_ys_to_xs_adaptor(), AdaptorTopIndex{get_lane_id(), 0});
        }
        else if constexpr(NDimP == 2)
        {
            window_adaptor_thread_coord_tmp =
                make_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                               AdaptorTopIndex{get_warp_id(), get_lane_id(), 0});
        }
#else
        // TODO: this use less register for FA, but more register for GEMM
        // need investigation
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_dstr_.get_ps_ys_to_xs_adaptor(),
            container_concat(get_partition_index(tile_dstr_), array<index_t, NDimY>{0}));
#endif

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin_ + window_adaptor_thread_coord_tmp.get_bottom_index();

        bottom_tensor_thread_origin_idx_tmp(HsGatherDim) = 0;
        const auto bottom_tensor_thread_coord_tmp        = make_tensor_coordinate(
            bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = load_store_traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}), idx_diff_ys);

            move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            pre_computed_coords_(iCoord) =
                make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });
    }

    CK_TILE_HOST_DEVICE void init_raw() { bottom_tensor_view_.init_raw(); }

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    //
    WindowLengths window_lengths_;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;

    // Tile tensor distribution, which contains:
    //   1. adaptor for window: [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register: [y0, y1, ...] ==> [d]
    TileDstr tile_dstr_;

    PageIdxArray page_idx_;
    ValidArray valids_;

    // this contains:
    //   per-thread coordinate for window adaptor
    //   per-thread coordinate for bottom tensor
    array<tuple<WindowAdaptorCoord, BottomTensorCoord>, NumCoord> pre_computed_coords_;
    std::conditional_t<BottomTensorView::buffer_view::get_address_space() ==
                           address_space_enum::global,
                       array<tuple<WindowAdaptorCoord, BottomTensorCoord>, NumCoord>,
                       std::byte>
        pre_computed_warp_coords_;
};

// TODO: use strategy
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename StaticPageIndexArray_,
          index_t HsGatherDim = 0,
          index_t NumCoord    = 1>
CK_TILE_DEVICE constexpr auto
make_tile_scatter_gather(const TensorView_& tensor_view,
                         const WindowLengths_& window_lengths,
                         const multi_index<TensorView_::get_num_of_dimension()>& origin,
                         const StaticTileDistribution_& tile_distribution,
                         const StaticPageIndexArray_& page_idx, // perbytes
                         number<HsGatherDim> = {},
                         number<NumCoord>    = {})
{
    return tile_scatter_gather<remove_cvref_t<TensorView_>,
                               remove_cvref_t<WindowLengths_>,
                               remove_cvref_t<StaticTileDistribution_>,
                               remove_cvref_t<StaticPageIndexArray_>,
                               std::nullptr_t,
                               HsGatherDim,
                               NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution, page_idx, nullptr};
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename StaticPageIndexArray,
          index_t HsGatherDim>
CK_TILE_DEVICE constexpr auto make_tile_scatter_gather(
    const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
    const multi_index<TensorView::get_num_of_dimension()>& origin,
    const StaticTileDistribution& tile_distribution,
    const StaticPageIndexArray& page_idx,
    number<HsGatherDim> = {})
{
    return make_tile_scatter_gather(tile_window.get_bottom_tensor_view(),
                                    tile_window.get_window_lengths(),
                                    origin,
                                    tile_distribution,
                                    page_idx,
                                    number<HsGatherDim>{});
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename StaticPageIndexArray,
          index_t HsGatherDim>
CK_TILE_DEVICE constexpr auto make_tile_scatter_gather(
    const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
    const StaticTileDistribution& tile_distribution,
    const StaticPageIndexArray& page_idx,
    number<HsGatherDim> = {})
{
    return make_tile_scatter_gather(tile_window.get_bottom_tensor_view(),
                                    tile_window.get_window_lengths(),
                                    tile_window.get_window_origin(),
                                    tile_distribution,
                                    page_idx,
                                    number<HsGatherDim>{});
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename StaticPageIndexArray_,
          typename StaticValidArray_,
          index_t HsGatherDim = 0,
          index_t NumCoord    = 1>
CK_TILE_DEVICE constexpr auto
make_tile_scatter_gather(const TensorView_& tensor_view,
                         const WindowLengths_& window_lengths,
                         const multi_index<TensorView_::get_num_of_dimension()>& origin,
                         const StaticTileDistribution_& tile_distribution,
                         const StaticPageIndexArray_& page_idx,
                         const StaticValidArray_& valids,
                         number<HsGatherDim> = {},
                         number<NumCoord>    = {})
{
    return tile_scatter_gather<remove_cvref_t<TensorView_>,
                               remove_cvref_t<WindowLengths_>,
                               remove_cvref_t<StaticTileDistribution_>,
                               remove_cvref_t<StaticPageIndexArray_>,
                               remove_cvref_t<StaticValidArray_>,
                               HsGatherDim,
                               NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution, page_idx, valids};
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename StaticPageIndexArray,
          typename StaticValidArray,
          index_t HsGatherDim>
CK_TILE_DEVICE constexpr auto make_tile_scatter_gather(
    const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
    const multi_index<TensorView::get_num_of_dimension()>& origin,
    const StaticTileDistribution& tile_distribution,
    const StaticPageIndexArray& page_idx,
    const StaticValidArray& valids,
    number<HsGatherDim> = {})
{
    return make_tile_scatter_gather(tile_window.get_bottom_tensor_view(),
                                    tile_window.get_window_lengths(),
                                    origin,
                                    tile_distribution,
                                    page_idx,
                                    valids,
                                    number<HsGatherDim>{});
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename StaticPageIndexArray,
          typename StaticValidArray,
          index_t HsGatherDim>
CK_TILE_DEVICE constexpr auto make_tile_scatter_gather(
    const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
    const StaticTileDistribution& tile_distribution,
    const StaticPageIndexArray& page_idx,
    const StaticValidArray& valids,
    number<HsGatherDim> = {})
{
    return make_tile_scatter_gather(tile_window.get_bottom_tensor_view(),
                                    tile_window.get_window_lengths(),
                                    tile_window.get_window_origin(),
                                    tile_distribution,
                                    page_idx,
                                    valids,
                                    number<HsGatherDim>{});
}

template <typename NewTensorView_,
          typename OldTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename StaticPageIndexArray_,
          typename StaticValidArray_,
          index_t HsGatherDim = 0,
          index_t NumCoord    = 1>
CK_TILE_DEVICE auto replace_bottom_tensor_view(const NewTensorView_& new_tensor_view,
                                               const tile_scatter_gather<OldTensorView_,
                                                                         WindowLengths_,
                                                                         StaticTileDistribution_,
                                                                         StaticPageIndexArray_,
                                                                         StaticValidArray_,
                                                                         HsGatherDim,
                                                                         NumCoord>& tile_window)
{
    return make_tile_scatter_gather(new_tensor_view,
                                    tile_window.window_lengths_,
                                    tile_window.window_origin_,
                                    tile_window.tile_dstr_,
                                    tile_window.page_idx_,
                                    tile_window.valids_);
}

} // namespace ck_tile
