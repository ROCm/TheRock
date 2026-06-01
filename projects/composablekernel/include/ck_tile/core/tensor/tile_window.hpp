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
#include "ck_tile/core/tensor/tensor_view.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/tile_window_base.hpp"
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
          index_t NumCoord>
struct tile_window_with_static_distribution
    : public tile_window_with_tile_dstr_base<
          tile_window_with_static_distribution<BottomTensorView_,
                                               WindowLengths_,
                                               StaticTileDistribution_,
                                               NumCoord>,
          BottomTensorView_,
          WindowLengths_,
          StaticTileDistribution_>
{
    using Base = tile_window_with_tile_dstr_base<
        tile_window_with_static_distribution<BottomTensorView_,
                                             WindowLengths_,
                                             StaticTileDistribution_,
                                             NumCoord>,
        BottomTensorView_,
        WindowLengths_,
        StaticTileDistribution_>;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static_assert(NumCoord == 1);

    static_assert(Base::Traits::NumAccess % NumCoord == 0,
                  "wrong! # of access is not divisible by NumCoord");
    static constexpr index_t NumAccessPerCoord = Base::Traits::NumAccess / NumCoord;

    CK_TILE_DEVICE constexpr tile_window_with_static_distribution() = default;

    CK_TILE_DEVICE constexpr tile_window_with_static_distribution(
        const typename Base::BottomTensorView& bottom_tensor_view,
        const typename Base::WindowLengths& window_lengths,
        const typename Base::BottomTensorIndex& window_origin,
        const typename Base::TileDstr& tile_distribution,
        decltype(get_partition_index(tile_distribution)) partition_index)
        : pre_computed_coords_{}
    {

        this->window_origin_      = window_origin;
        this->window_lengths_     = window_lengths;
        this->bottom_tensor_view_ = bottom_tensor_view;
        this->tile_dstr_          = tile_distribution;

        pre_computed_coords_ =
            prepare_coords(bottom_tensor_view, window_origin, tile_distribution, partition_index);
        if constexpr(Base::BottomTensorView::buffer_view::get_address_space() ==
                     address_space_enum::global)
        {
            auto use_lane_id_0 = partition_index;
            use_lane_id_0[1]   = 0;

            pre_computed_warp_coords_ =
                prepare_coords(bottom_tensor_view, window_origin, tile_distribution, use_lane_id_0);
        }
    }

    CK_TILE_DEVICE constexpr tile_window_with_static_distribution(
        const typename Base::BottomTensorView& bottom_tensor_view,
        const typename Base::WindowLengths& window_lengths,
        const typename Base::BottomTensorIndex& window_origin,
        const typename Base::TileDstr& tile_distribution)
        : tile_window_with_static_distribution(bottom_tensor_view,
                                               window_lengths,
                                               window_origin,
                                               tile_distribution,
                                               get_partition_index(tile_distribution))
    {
    }

    CK_TILE_DEVICE constexpr auto
    prepare_coords(const typename Base::BottomTensorView& bottom_tensor_view,
                   const typename Base::BottomTensorIndex& window_origin,
                   const typename Base::TileDstr& tile_distribution,
                   decltype(get_partition_index(tile_distribution)) partition_index) const
    {
        array<tuple<typename Base::WindowAdaptorCoord, typename Base::BottomTensorCoord>, NumCoord>
            coords;

        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(partition_index, multi_index<Base::NDimY>{0}));

        typename Base::BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            bottom_tensor_view.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = typename Base::Traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                idx_diff_ys);

            Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            coords(iCoord) = make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });

        return coords;
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load(number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        return load_with_offset(
            0, number<i_access_unsupport_>{}, bool_constant<oob_conditional_check>{});
    }

    template <index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              typename offset_t           = index_t>
    CK_TILE_DEVICE auto load_with_offset(offset_t offset,
                                         number<i_access_unsupport_>          = {},
                                         bool_constant<oob_conditional_check> = {}) const
    {
        constexpr auto tile_dstr = typename Base::TileDstr{};
        auto dst_tensor = make_static_distributed_tensor<typename Base::DataType>(tile_dstr);
        load_with_offset(offset,
                         dst_tensor,
                         number<i_access_unsupport_>{},
                         bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    /**
     * @brief Load tile with elementwise function
     *
     * @note Load tile with elementwise — during value loading, an
     *       elementwise function is executed for each A0, A1, … AN.
     *       The values A0, A1, … AN are read by the same thread. In this way, we
     *       reduce the amount of information loaded into the registers.
     *       The same thread, during vectorized reading, accesses the same set of
     *       data from A0, A1, A2, … AN.
     */
    template <typename TileWindow_,
              typename ElementWise_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE auto load(const TileWindow_& tile_window,
                             ElementWise_ elementwise,
                             number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        constexpr auto tile_dstr = typename Base::TileDstr{};
        auto dst_tensor = make_static_distributed_tensor<typename Base::DataType>(tile_dstr);
        load(dst_tensor,
             tile_window,
             elementwise,
             number<i_access_unsupport_>{},
             bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    template <typename DistributedTensor,
              typename TileWindow_,
              typename ElementWise_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE void load(DistributedTensor& dst_tensor,
                             const TileWindow_& tile_window,
                             ElementWise_ elementwise,
                             number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {

        using Traits   = typename Base::Traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr   = typename Base::TileDstr{};
        constexpr auto sizeOfTuple = TileWindow_::size();
        //  loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord =
                tile_window[number<0>{}].pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord =
                tile_window[number<0>{}].pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from bottom tensor
                const auto idx_vec_value = generate_tuple(
                    [&](auto jj) {
                        return tile_window[number<jj>{}]
                            .get_bottom_tensor_view()
                            .template get_vectorized_elements<vector_t>(
                                bottom_tensor_thread_coord,
                                0,
                                bool_constant<oob_conditional_check>{});
                    },
                    number<sizeOfTuple>{});

                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    ck_tile::apply(
                        [&](auto&&... t) {
                            elementwise(dst_tensor.get_thread_buffer().template at<d>(),
                                        t.template get_as<
                                            typename Base::DataType>()[j / Traits::PackedSize]...);
                        },
                        idx_vec_value);
                });
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <typename DistributedTensor,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE void load(DistributedTensor& dst_tensor,
                             number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        load_with_offset(
            0, dst_tensor, number<i_access_unsupport_>{}, bool_constant<oob_conditional_check>{});
    }

    template <typename offset_t>
    CK_TILE_DEVICE constexpr auto get_load_offset(offset_t = {}) const
    {
        constexpr auto bottom_tensor_idx_off = to_multi_index(offset_t{});
        const auto bottom_tensor_coord_off   = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_idx_off);
        return amd_wave_read_first_lane(bottom_tensor_coord_off.get_offset());
    }

    template <typename DataType,
              typename StaticTileDistribution,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              typename offset_t>
    CK_TILE_DEVICE void load_with_offset( //
        offset_t offset,
        static_distributed_tensor<DataType, StaticTileDistribution>& dst_tensor,
        number<i_access_unsupport_>          = {},
        bool_constant<oob_conditional_check> = {}) const
    {
        using Traits   = typename Base::Traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        const index_t linear_off = [&]() {
            if constexpr(std::is_integral_v<offset_t>)
                return offset;
            else if constexpr(is_constant_v<offset_t>)
                return offset_t::value;
            else
                return get_load_offset(offset_t{});
        }();
        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from bottom tensor
                const vector_t vec_value =
                    this->get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        linear_off,
                        bool_constant<oob_conditional_check>{});
                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    dst_tensor.get_thread_buffer().template at<d>() =
                        vec_value
                            .template get_as<typename Base::DataType>()[j / Traits::PackedSize];
                });
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <typename DstTile,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false>
    CK_TILE_DEVICE void load_raw(DstTile& dst_tensor,
                                 number<i_access_unsupport_>          = {},
                                 bool_constant<oob_conditional_check> = {},
                                 bool_constant<pre_nop>               = {}) const
    {
        using Traits   = typename Base::Traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;
        static constexpr index_t YElementSize =
            typename Base::TileDstr{}.get_ys_to_d_descriptor().get_element_space_size();
        static_assert(YElementSize % (Traits::PackedSize * Traits::ScalarPerVector) == 0);
        using vectorized_tbuf =
            array<vector_t, YElementSize / (Traits::PackedSize * Traits::ScalarPerVector)>;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        auto& dst_vec_tbuf = reinterpret_cast<vectorized_tbuf&>(dst_tensor.get_thread_buffer());

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

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr index_t d =
                    tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start) /
                    Traits::PackedSize;
                static_assert(d % Traits::ScalarPerVector == 0);

                this->get_bottom_tensor_view().template get_vectorized_elements_raw<vector_t>(
                    dst_vec_tbuf.template at<d / Traits::ScalarPerVector>(),
                    bottom_tensor_thread_coord,
                    0 /**/,
                    bool_constant<oob_conditional_check>{},
                    pre_nop_);
#if CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE || \
    CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE
                asm volatile(
                    ""); // this is starting from rocm-6.2, but same sympton, reuse this flag
#endif
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // TODO: currently async load only implemented in inline asm
    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false>
    CK_TILE_DEVICE void async_load_raw(LdsTileWindow_&& lds_tile,
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

        // Use VALU so the compiler can optimize redundant/repeated computations
        const index_t m0_init_value =
            size_per_buf + size_per_wave * get_warp_id(/*ReturnSgpr=*/bool_constant<false>{});
        m0_set_with_memory(
            amd_wave_read_first_lane(m0_init_value)); // This should be wave independent

        using Traits = typename Base::Traits;

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

                // read from bottom tensor
                this->get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                    smem, bottom_tensor_thread_coord, 0, pre_nop_);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                    m0_inc_with_memory(size_per_issue);
                }
            });
        });
    }

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
        using Traits        = typename Base::Traits;

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
                        container_concat(array<index_t, Base::NDimP>{0},
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

                this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                    smem,
                    bottom_tensor_thread_coord,
                    offset + dram_ys_offset,
                    bool_constant<oob_conditional_check>{});

                // Move thread coordinate if not last access
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys    = SFC_Ys::get_forward_step(iAccess);
                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    if constexpr(!static_move_ys)
                        Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                            window_adaptor_thread_coord,
                            bottom_tensor_thread_coord,
                            idx_diff_ps_ys);

                    if constexpr(!static_move_ys)
                        Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                            window_adaptor_warp_coord, bottom_tensor_warp_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <typename Policy, index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load_transpose(number<i_access_unsupport_>          = {},
                                       bool_constant<oob_conditional_check> = {}) const
    {
        return this->template load_transpose_with_offset<Policy>(
            0, number<i_access_unsupport_>{}, bool_constant<oob_conditional_check>{});
    }

    template <typename Policy, index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load_transpose_with_offset(index_t offset,
                                                   number<i_access_unsupport_>          = {},
                                                   bool_constant<oob_conditional_check> = {}) const
    {
        constexpr auto tile_dstr = typename Base::TileDstr{};
        auto dst_tensor = make_static_distributed_tensor<typename Base::DataType>(tile_dstr);
        this->template load_transpose_with_offset<Policy>(offset,
                                                          dst_tensor,
                                                          number<i_access_unsupport_>{},
                                                          bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    template <typename Policy,
              typename DistributedTensor,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE void load_transpose_with_offset(index_t offset,
                                                   DistributedTensor& dst_tensor,
                                                   number<i_access_unsupport_>          = {},
                                                   bool_constant<oob_conditional_check> = {}) const
    {
        using Traits   = typename Base::Traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        constexpr auto group_func = Policy::group_func;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from bottom tensor
                const vector_t vec_value =
                    this->get_bottom_tensor_view()
                        .template get_transpose_vectorized_elements<vector_t>(
                            bottom_tensor_thread_coord, offset);
                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto orig_idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr auto grouped_idx_ys = group_func(orig_idx_ys);

                    constexpr index_t linear_distributed_index =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(grouped_idx_ys);

                    dst_tensor.get_thread_buffer().template at<linear_distributed_index>() =
                        vec_value.template get_as<typename Base::DataType>()[j];
                });
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void store(const static_distributed_tensor<typename Base::DataType,
                                                              typename Base::TileDstr>& dstr_tensor,
                              number<i_access_unsupport_>          = {},
                              bool_constant<oob_conditional_check> = {}) const
    {
        using Traits = typename Base::Traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                // vector_type_t vec;
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    vec_value.template get_as<typename Base::DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // const vector_t vec_value = vec.template get_as<vector_t>().template at<0>();

                // write into bottom tensor
                this->get_bottom_tensor_view().template set_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    0,
                    vec_value,
                    bool_constant<oob_conditional_check>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1>
    CK_TILE_DEVICE void
    store_raw(const static_distributed_tensor<typename Base::DataType, typename Base::TileDstr>&
                  dstr_tensor,
              number<i_access_unsupport_> = {}) const
    {
        using Traits = typename Base::Traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr                    = typename Base::TileDstr{};
        static constexpr bool oob_conditional_check = true;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                vector_t vec_value;
                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});
                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;
                    vec_value.template get_as<typename Base::DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                this->get_bottom_tensor_view()
                    .template set_vectorized_elements_raw<vector_t, oob_conditional_check>(
                        bottom_tensor_thread_coord, 0, vec_value);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void
    update(const static_distributed_tensor<typename Base::DataType, typename Base::TileDstr>&
               dstr_tensor,
           number<i_access_unsupport_>          = {},
           bool_constant<oob_conditional_check> = {}) const
    {
        using Traits = typename Base::Traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    vec_value.template get_as<typename Base::DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                this->get_bottom_tensor_view().template update_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    0,
                    vec_value,
                    bool_constant<oob_conditional_check>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true, bool pre_nop>
    CK_TILE_DEVICE void
    update_raw(const static_distributed_tensor<typename Base::DataType, typename Base::TileDstr>&
                   dstr_tensor,
               number<i_access_unsupport_>          = {},
               bool_constant<oob_conditional_check> = {},
               bool_constant<pre_nop>               = {}) const
    {
        using Traits = typename Base::Traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    vec_value.template get_as<typename Base::DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                this->get_bottom_tensor_view().template update_vectorized_elements_raw<vector_t>(
                    bottom_tensor_thread_coord,
                    0,
                    vec_value,
                    bool_constant<oob_conditional_check>{},
                    bool_constant<pre_nop>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // Custom move behavior
    CK_TILE_DEVICE void move_extended(const typename Base::BottomTensorIndex& step)
    {
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            move_tensor_coordinate(this->bottom_tensor_view_.get_tensor_descriptor(),
                                   pre_computed_coords_(iCoord)(I1),
                                   step);
        });

        if constexpr(Base::BottomTensorView::buffer_view::get_address_space() ==
                     address_space_enum::global)
        {
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                move_tensor_coordinate(this->bottom_tensor_view_.get_tensor_descriptor(),
                                       pre_computed_warp_coords_(iCoord)(I1),
                                       step);
            });
        }
    }

    CK_TILE_DEVICE void set_window_origin_extended(const typename Base::BottomTensorIndex&)
    {
        // TODO: this use less register for FA, but more register for GEMM
        // need investigation
        const auto window_adaptor_thread_coord_tmp =
            make_tensor_adaptor_coordinate(this->tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                           container_concat(get_partition_index(this->tile_dstr_),
                                                            array<index_t, Base::NDimY>{0}));

        typename Base::BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            this->window_origin_ + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = typename Base::Traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                idx_diff_ys);

            Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            pre_computed_coords_(iCoord) =
                make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });
    }

    // this contains:
    //   per-thread coordinate for window adaptor
    //   per-thread coordinate for bottom tensor
    array<tuple<typename Base::WindowAdaptorCoord, typename Base::BottomTensorCoord>, NumCoord>
        pre_computed_coords_;
    // pre_computed_warp_coords_ exists only in the global memory tile_window
    std::conditional_t<
        Base::BottomTensorView::buffer_view::get_address_space() == address_space_enum::global,
        array<tuple<typename Base::WindowAdaptorCoord, typename Base::BottomTensorCoord>, NumCoord>,
        std::byte>
        pre_computed_warp_coords_;
};

// TODO: use strategy
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const multi_index<TensorView_::get_num_of_dimension()>& origin,
                 const StaticTileDistribution_& tile_distribution,
                 number<NumCoord> = {})
{
    return tile_window_with_static_distribution<remove_cvref_t<TensorView_>,
                                                remove_cvref_t<WindowLengths_>,
                                                remove_cvref_t<StaticTileDistribution_>,
                                                NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution};
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1,
          typename         = std::enable_if_t<is_tensor_view_v<TensorView_> &&
                                              is_tile_distribution_v<StaticTileDistribution_>>>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const multi_index<TensorView_::get_num_of_dimension()>& origin,
                 const StaticTileDistribution_& tile_distribution,
                 decltype(get_partition_index(tile_distribution)) partition_index,
                 number<NumCoord> = {})
{
    return tile_window_with_static_distribution<remove_cvref_t<TensorView_>,
                                                remove_cvref_t<WindowLengths_>,
                                                remove_cvref_t<StaticTileDistribution_>,
                                                NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution, partition_index};
}

// this version can't be called in a constexpr context
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1>
CK_TILE_DEVICE auto
make_tile_window_raw(const TensorView_& tensor_view,
                     const WindowLengths_& window_lengths,
                     const multi_index<TensorView_::get_num_of_dimension()>& origin,
                     const StaticTileDistribution_& tile_distribution,
                     number<NumCoord> = {})
{
    auto w = tile_window_with_static_distribution<remove_cvref_t<TensorView_>,
                                                  remove_cvref_t<WindowLengths_>,
                                                  remove_cvref_t<StaticTileDistribution_>,
                                                  NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution};
    w.init_raw();
    return w;
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
CK_TILE_DEVICE void move_tile_window(
    tile_window_with_static_distribution<TensorView_,
                                         WindowLengths_,
                                         StaticTileDistribution_,
                                         NumCoord>& window,
    const typename tile_window_with_static_distribution<TensorView_,
                                                        WindowLengths_,
                                                        StaticTileDistribution_,
                                                        NumCoord>::BottomTensorIndex& step)
{
    window.move(step);
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
CK_TILE_DEVICE void move_tile_window(
    tuple<tile_window_with_static_distribution<TensorView_,
                                               WindowLengths_,
                                               StaticTileDistribution_,
                                               NumCoord>>& window,
    const typename tile_window_with_static_distribution<TensorView_,
                                                        WindowLengths_,
                                                        StaticTileDistribution_,
                                                        NumCoord>::BottomTensorIndex& step)
{
    using T = tuple<tile_window_with_static_distribution<TensorView_,
                                                         WindowLengths_,
                                                         StaticTileDistribution_,
                                                         NumCoord>>;

    static constexpr auto N = T::size();
    static_for<0, N, 1>{}([&](auto Is) { window[number<Is>{}].move(step); });
}

template <typename TileWindowWithStaticDistributionType,
          typename StepType,
          typename std::enable_if_t<
              is_detected<is_tuple, TileWindowWithStaticDistributionType>::value>* = nullptr>
CK_TILE_DEVICE void move_tile_window(TileWindowWithStaticDistributionType& window, StepType& step)
{
    static constexpr auto N = TileWindowWithStaticDistributionType::size();
    static_for<0, N, 1>{}([&](auto Is) { window[number<Is>{}].move(step); });
}

/**
 * @brief This class provides description of tile windowed view on the device memory.
 *
 * @note This class does not provide any functions to read or modify device memory.
 *
 * @tparam BottomTensorView_    Class describing & holding device tensor memory.
 * @tparam WindowLengths_       Spatial sizes of windowed view on tensor.
 */
template <typename BottomTensorView_, typename WindowLengths_>
struct tile_window_with_static_lengths
    : public tile_window_base<tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>,
                              BottomTensorView_,
                              WindowLengths_>
{
    using Base =
        tile_window_base<tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>,
                         BottomTensorView_,
                         WindowLengths_>;

    CK_TILE_DEVICE constexpr tile_window_with_static_lengths() = default;

    CK_TILE_DEVICE constexpr tile_window_with_static_lengths(
        const typename Base::BottomTensorView& bottom_tensor_view,
        const typename Base::WindowLengths& window_lengths,
        const typename Base::BottomTensorIndex& window_origin)
    {
        this->window_origin_      = window_origin;
        this->window_lengths_     = window_lengths;
        this->bottom_tensor_view_ = bottom_tensor_view;
    }

    /**
     * @brief Print tile window elements for debugging.
     *
     * @tparam DataType Element data type (e.g., fp16_t, float, bf8_t)
     * @param start_i Starting row (inclusive)
     * @param end_i   Ending row (exclusive)
     * @param start_j Starting column (inclusive)
     * @param end_j   Ending column (exclusive)
     * @param label   Optional output label
     *
     * @note Tested on fp16. Custom types may need adjustments.
     * @example tile_window.template print_tile_window_range<fp16_t>(0, 4, 0, 8, "A");
     */
    template <typename DataType>
    CK_TILE_DEVICE void print_tile_window_range(index_t start_i,
                                                index_t end_i,
                                                index_t start_j,
                                                index_t end_j,
                                                const char* label = "") const
    {
        const auto& tensor_view  = this->get_bottom_tensor_view();
        const auto window_origin = this->get_window_origin();

        printf("%s Window Range [%d:%d, %d:%d] (origin: %d, %d):\n",
               label,
               start_i,
               end_i - 1,
               start_j,
               end_j - 1,
               window_origin[0],
               window_origin[1]);

        for(index_t i = start_i; i < end_i; i++)
        {
            for(index_t j = start_j; j < end_j; j++)
            {
                // Create coordinate for this element relative to window origin
                auto coord =
                    make_tensor_coordinate(tensor_view.get_tensor_descriptor(),
                                           make_tuple(window_origin[0] + i, window_origin[1] + j));

                // Get the element using thread buffer type directly
                using ThreadBuf = thread_buffer<DataType, 2>;
                auto buf        = tensor_view.template get_vectorized_elements<ThreadBuf>(coord, 0);
                auto value      = buf.at(number<0>{}); // Extract first element from thread buffer
                printf("  %s[%d,%d] = %f", label, i, j, type_convert<float>(value));
            }
            printf("\n");
        }
        printf("\n");
    }
};

template <typename TensorView_,
          typename WindowLengths_,
          typename = std::enable_if_t<is_tensor_view_v<TensorView_>>>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const multi_index<TensorView_::get_num_of_dimension()>& origin)
{
    static_assert(ck_tile::is_known_at_compile_time<WindowLengths_>::value,
                  "wrong! lengths should be static");

    return tile_window_with_static_lengths<remove_cvref_t<TensorView_>,
                                           remove_cvref_t<WindowLengths_>>{
        tensor_view, window_lengths, origin};
}

// duplicate tile window and replace its origin
template <typename TensorView, typename WindowLengths>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const multi_index<TensorView::get_num_of_dimension()>& origin)
{
    return tile_window_with_static_lengths<TensorView, WindowLengths>{
        tile_window.get_bottom_tensor_view(), tile_window.get_window_lengths(), origin};
}

template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const multi_index<TensorView::get_num_of_dimension()>& origin,
                 const StaticTileDistribution& tile_distribution)
{
    return make_tile_window(tile_window.get_bottom_tensor_view(),
                            tile_window.get_window_lengths(),
                            origin,
                            tile_distribution);
}

template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const StaticTileDistribution& tile_distribution)
{
    return make_tile_window(tile_window.get_bottom_tensor_view(),
                            tile_window.get_window_lengths(),
                            tile_window.get_window_origin(),
                            tile_distribution);
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename = std::enable_if_t<is_tile_distribution_v<StaticTileDistribution>>>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const StaticTileDistribution& tile_distribution,
                 decltype(get_partition_index(tile_distribution)) partition_index)
{
    return make_tile_window(tile_window.get_bottom_tensor_view(),
                            tile_window.get_window_lengths(),
                            tile_window.get_window_origin(),
                            tile_distribution,
                            partition_index);
}

template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto
make_tile_window_raw(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const StaticTileDistribution& tile_distribution)
{
    auto w = make_tile_window(tile_window, tile_distribution);
    w.init_raw();
    return w;
}

template <typename TensorView_, typename WindowLengths_>
CK_TILE_DEVICE void move_tile_window(
    tile_window_with_static_lengths<TensorView_, WindowLengths_>& window,
    const typename tile_window_with_static_lengths<TensorView_, WindowLengths_>::BottomTensorIndex&
        step)
{
    window.move(step);
}

template <typename NewTensorView_,
          typename OldTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1>
CK_TILE_DEVICE auto
replace_bottom_tensor_view(const NewTensorView_& new_tensor_view,
                           const tile_window_with_static_distribution<OldTensorView_,
                                                                      WindowLengths_,
                                                                      StaticTileDistribution_,
                                                                      NumCoord>& tile_window)
{
    return make_tile_window(new_tensor_view,
                            tile_window.get_window_lengths(),
                            tile_window.get_window_origin(),
                            tile_window.get_tile_distribution());
}

template <typename NewTensorView_, typename OldTensorView_, typename WindowLengths_>
CK_TILE_DEVICE auto replace_bottom_tensor_view(
    const NewTensorView_& new_tensor_view,
    const tile_window_with_static_lengths<OldTensorView_, WindowLengths_>& tile_window)
{
    return make_tile_window(
        new_tensor_view, tile_window.get_window_lengths(), tile_window.get_window_origin());
}

/**
 * @brief Type trait to determine if a type is a tile window with static distribution.
 *
 * Defaults to `false_type`. Specializations define when the trait evaluates to `true`.
 *
 * @tparam T The type to check.
 */
template <typename T>
struct is_tile_window_with_static_distribution : std::false_type
{
};

/**
 * @brief Specialization for `tile_window_with_static_distribution` to evaluate to `true_type`.
 *
 * @tparam BottomTensorView_ Bottom tensor view type of the tile window.
 * @tparam WindowLengths_ Static window lengths.
 * @tparam StaticTileDistribution_ Tile distribution policy.
 * @tparam NumCoord Number of coordinate dimensions.
 */
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
struct is_tile_window_with_static_distribution<
    tile_window_with_static_distribution<BottomTensorView_,
                                         WindowLengths_,
                                         StaticTileDistribution_,
                                         NumCoord>> : std::true_type
{
};

/**
 * @brief Helper variable template to check if a type is a tile window with static distribution.
 *
 * Equivalent to `is_tile_window_with_static_distribution<T>::value`.
 *
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_tile_window_with_static_distribution_v =
    is_tile_window_with_static_distribution<T>::value;

/**
 * @brief Type trait to determine if a type is a tile window with static lengths.
 *
 * Defaults to `false_type`. Specializations define when the trait evaluates to `true`.
 *
 * @tparam T The type to check.
 */
template <typename T>
struct is_tile_window_with_static_lengths : std::false_type
{
};

/**
 * @brief Specialization for `tile_window_with_static_lengths` to evaluate to `true_type`.
 *
 * @tparam BottomTensorView_ Bottom tensor view type of the tile window.
 * @tparam WindowLengths_ Static window lengths.
 */
template <typename BottomTensorView_, typename WindowLengths_>
struct is_tile_window_with_static_lengths<
    tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>> : std::true_type
{
};

/**
 * @brief Helper variable template to check if a type is a tile window with static lengths.
 *
 * Equivalent to `is_tile_window_with_static_lengths<T>::value`.
 *
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_tile_window_with_static_lengths_v =
    is_tile_window_with_static_lengths<T>::value;

} // namespace ck_tile
