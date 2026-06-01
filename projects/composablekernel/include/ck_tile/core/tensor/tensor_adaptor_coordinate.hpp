// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

template <index_t NDimHidden, typename BottomDimensionHiddenIds, typename TopDimensionHiddenIds>
struct tensor_adaptor_coordinate
{
    static constexpr index_t ndim_bottom_ = BottomDimensionHiddenIds::size();
    static constexpr index_t ndim_top_    = TopDimensionHiddenIds::size();

    using HiddenIndex = multi_index<NDimHidden>;
    using BottomIndex = multi_index<ndim_bottom_>;
    using TopIndex    = multi_index<ndim_top_>;

    public:
    CK_TILE_HOST_DEVICE constexpr tensor_adaptor_coordinate() = default;

    CK_TILE_HOST_DEVICE constexpr tensor_adaptor_coordinate(const HiddenIndex& idx_hidden)
        : idx_hidden_{idx_hidden}
    {
    }

    CK_TILE_HOST_DEVICE constexpr auto get_top_index() const
    {
        return get_container_subset(idx_hidden_, TopDimensionHiddenIds{});
    }

    CK_TILE_HOST_DEVICE constexpr auto get_bottom_index() const
    {
        return get_container_subset(idx_hidden_, BottomDimensionHiddenIds{});
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_hidden_index() const { return idx_hidden_; }

    CK_TILE_HOST_DEVICE constexpr auto& get_hidden_index() { return idx_hidden_; }

    //
    HiddenIndex idx_hidden_;
};

template <typename Adaptor, typename TopIndex>
CK_TILE_HOST_DEVICE constexpr auto make_tensor_adaptor_coordinate(const Adaptor& adaptor,
                                                                  const TopIndex& idx_top)
{
    static_assert(Adaptor::get_num_of_top_dimension() == TopIndex::size(),
                  "wrong! # of dimension inconsistent");

    constexpr index_t ntransform  = Adaptor::get_num_of_transform();
    constexpr index_t ndim_hidden = Adaptor::get_num_of_hidden_dimension();
    constexpr auto bottom_dim_ids = Adaptor::get_bottom_dimension_hidden_ids();
    constexpr auto top_dim_ids    = Adaptor::get_top_dimension_hidden_ids();

    multi_index<ndim_hidden> idx_hidden;

    // initialize visible index
    set_container_subset(idx_hidden, top_dim_ids, idx_top);

    // calculate hidden index
    static_for<ntransform, 0, -1>{}([&adaptor, &idx_hidden](auto itran_p1) {
        auto itran              = itran_p1 - number<1>{};
        const auto& tran        = adaptor.get_transforms().at(itran);
        constexpr auto dims_low = Adaptor::get_lower_dimension_hidden_idss().at(itran);
        constexpr auto dims_up  = Adaptor::get_upper_dimension_hidden_idss().at(itran);

        const auto idx_up = get_container_subset(idx_hidden, dims_up);

        multi_index<dims_low.size()> idx_low;

        tran.calculate_lower_index(idx_low, idx_up);

        set_container_subset(idx_hidden, dims_low, idx_low);
    });

    return tensor_adaptor_coordinate<ndim_hidden,
                                     remove_cvref_t<decltype(bottom_dim_ids)>,
                                     remove_cvref_t<decltype(top_dim_ids)>>{idx_hidden};
}

template <bool JudgeDoTransforms = true,
          typename Adaptor,
          typename AdaptorCoord,
          typename TopIndex,
          typename BottomIndex>
CK_TILE_HOST_DEVICE constexpr void move_tensor_adaptor_coordinate(const Adaptor& adaptor,
                                                                  AdaptorCoord& coord,
                                                                  const TopIndex& idx_diff_top,
                                                                  BottomIndex& idx_diff_bottom)
{
    constexpr index_t ndim_hidden = Adaptor::get_num_of_hidden_dimension();
    constexpr index_t ndim_top    = Adaptor::get_num_of_top_dimension();
    //  constexpr index_t ndim_bottom = Adaptor::get_num_of_bottom_dimension();
    constexpr index_t ntransform = Adaptor::get_num_of_transform();

    //  static_assert(TopIndex::size() == ndim_top && BottomIndex::size() == ndim_bottom, "");

    // judge whether calculation of lower diff is needed for each transform
    // use index_t for boolean type
    auto do_transforms = make_zero_multi_index<ntransform>();

    if constexpr(JudgeDoTransforms)
    {
        auto is_non_zero_diff = make_zero_multi_index<ndim_hidden>();

        // decide do_transform by checkout non-zero index diff components
        multi_index<ndim_top> non_zero_diff_pick_top;

        static_for<0, ndim_top, 1>{}(
            [&](auto i) { non_zero_diff_pick_top(i) = (idx_diff_top[i] != 0); });

        set_container_subset(
            is_non_zero_diff, Adaptor::get_top_dimension_hidden_ids(), non_zero_diff_pick_top);

        static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
            constexpr auto dims_low = Adaptor::get_lower_dimension_hidden_idss().at(itran);
            constexpr auto dims_up  = Adaptor::get_upper_dimension_hidden_idss().at(itran);

            const auto non_zero_diff_pick_up = get_container_subset(is_non_zero_diff, dims_up);

            multi_index<dims_low.size()> non_zero_diff_pick_low;

            // if any of upper index diff components is non-zero, then
            //   1) Need to do this transform
            //   2) all components of lower index diff will assume to be non-zero and need to be
            //   computed
            const bool idx_diff_up_has_non_zero = container_reduce(
                non_zero_diff_pick_up, [](auto a, auto b) constexpr { return a or b; }, false);

            do_transforms(itran) = idx_diff_up_has_non_zero;

            static_for<0, dims_low.size(), 1>{}(
                [&](auto i) { non_zero_diff_pick_low(i) = idx_diff_up_has_non_zero; });

            set_container_subset(is_non_zero_diff, dims_low, non_zero_diff_pick_low);
        });
    }
    else
    {
        static_for<ntransform - 1, -1, -1>{}([&](auto itran) { do_transforms(itran) = 1; });
    }

    // this is what needs to be calculated
    auto idx_diff_hidden = make_zero_multi_index<ndim_hidden>();

    // initialize top index diff
    set_container_subset(idx_diff_hidden, Adaptor::get_top_dimension_hidden_ids(), idx_diff_top);

    // this is what needs to be updated
    auto& idx_hidden = coord.get_hidden_index();

    // update top index
    auto idx_hidden_pick_top =
        get_container_subset(idx_hidden, Adaptor::get_top_dimension_hidden_ids());

    idx_hidden_pick_top += idx_diff_top;

    set_container_subset(idx_hidden, Adaptor::get_top_dimension_hidden_ids(), idx_hidden_pick_top);

    // update rest of hidden index
    static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
        if(do_transforms[itran])
        {
            const auto& tran        = adaptor.get_transforms().at(itran);
            constexpr auto dims_low = Adaptor::get_lower_dimension_hidden_idss().at(itran);
            constexpr auto dims_up  = Adaptor::get_upper_dimension_hidden_idss().at(itran);

            const auto idx_up_new  = get_container_subset(idx_hidden, dims_up);
            auto idx_low           = get_container_subset(idx_hidden, dims_low);
            const auto idx_diff_up = get_container_subset(idx_diff_hidden, dims_up);

            multi_index<dims_low.size()> idx_diff_low;

            tran.update_lower_index(idx_diff_low, idx_diff_up, idx_low, idx_up_new);

            set_container_subset(idx_diff_hidden, dims_low, idx_diff_low);
            set_container_subset(idx_hidden, dims_low, idx_low);
        }
    });

    // set bottom index diff
    idx_diff_bottom =
        get_container_subset(idx_diff_hidden, Adaptor::get_bottom_dimension_hidden_ids());
}

template <bool JudgeDoTransforms = true, typename Adaptor, typename AdaptorCoord, typename TopIndex>
CK_TILE_HOST_DEVICE constexpr void move_tensor_adaptor_coordinate(const Adaptor& adaptor,
                                                                  AdaptorCoord& coord,
                                                                  const TopIndex& idx_diff_top)
{
    constexpr index_t ndim_bottom = Adaptor::get_num_of_bottom_dimension();

    multi_index<ndim_bottom> tmp;

    move_tensor_adaptor_coordinate<JudgeDoTransforms>(adaptor, coord, idx_diff_top, tmp);
}

template <typename Adaptor, typename AdaptorCoord>
CK_TILE_HOST_DEVICE constexpr bool
adaptor_coordinate_is_valid_assuming_top_index_is_valid(const Adaptor& adaptor,
                                                        const AdaptorCoord& coord)
{
    bool valid = true;

    constexpr index_t ntransform = Adaptor::get_num_of_transform();

    const auto& idx_hidden = coord.get_hidden_index();

    static_for<ntransform - 1, -1, -1>{}([&adaptor, &idx_hidden, &valid](auto itran) {
        const auto tran = adaptor.get_transforms().at(itran);

        // check validity, only if current transformation does not always has a valid mapping
        if constexpr(!decltype(tran)::is_valid_upper_index_always_mapped_to_valid_lower_index())
        {
            const auto idx_up = get_container_subset(
                idx_hidden, Adaptor::get_upper_dimension_hidden_idss().at(itran));

            // Comment: using valid = valid && .. will result in weird control flow in ISA
            valid &= tran.is_valid_upper_index_mapped_to_valid_lower_index(idx_up);
        }
    });

    return valid;
}

template <typename Adaptor, typename AdpatorCoord>
CK_TILE_HOST_DEVICE constexpr bool adaptor_coordinate_is_valid(const Adaptor& adaptor,
                                                               const AdpatorCoord& coord)
{
    // check top index
    const auto& idx_top = coord.get_top_index();

    bool is_top_index_valid = true;

    static_for<0, Adaptor::get_num_of_dimension(), 1>{}(
        [&is_top_index_valid, &idx_top, &adaptor](auto i) {
            is_top_index_valid =
                is_top_index_valid && (idx_top[i] >= 0 && idx_top[i] < adaptor.get_length(i));
        });

    // check other hidden index
    return is_top_index_valid &&
           adaptor_coordinate_is_valid_assuming_top_index_is_valid(adaptor, coord);
}

namespace detail {
template <typename PREFIX = str_literal<>, typename SUFFIX = str_literal<>>
struct CK_PRINT_X_;

template <char... PREFIXChars, char... SUFFIXChars>
struct CK_PRINT_X_<str_literal<PREFIXChars...>, str_literal<SUFFIXChars...>>
{
    template <typename T>
    struct detail;
    template <index_t NDimHidden, typename BottomDimensionHiddenIds, typename TopDimensionHiddenIds>
    struct detail<
        tensor_adaptor_coordinate<NDimHidden, BottomDimensionHiddenIds, TopDimensionHiddenIds>>
    {
        using coord_t =
            tensor_adaptor_coordinate<NDimHidden, BottomDimensionHiddenIds, TopDimensionHiddenIds>;

        template <index_t I>
        CK_TILE_HOST_DEVICE static constexpr auto get_hidden_format_i()
        {
            constexpr bool is_bottom =
                sequence_any_of(BottomDimensionHiddenIds{}, [](auto b) { return b == I; });
            constexpr bool is_top =
                sequence_any_of(TopDimensionHiddenIds{}, [](auto t) { return t == I; });
            constexpr auto d = make_str_literal("%d");
            if constexpr(is_bottom && is_top)
                return make_str_literal("_^") + d;
            else if constexpr(is_bottom)
                return make_str_literal("_") + d;
            else if constexpr(is_top)
                return make_str_literal("^") + d;
            else
                return d;
        }
        template <index_t N = NDimHidden>
        CK_TILE_HOST_DEVICE static constexpr auto get_hidden_format()
        {
            constexpr auto sep = make_str_literal(" ");
            if constexpr(N == 0)
                return str_literal<>{};
            else
                return get_hidden_format<N - 1>() + sep + get_hidden_format_i<N - 1>();
        }
        CK_TILE_HOST_DEVICE static constexpr auto get_format()
        {
            constexpr auto d   = make_str_literal("%d");
            constexpr auto sep = make_str_literal(" ");
            constexpr auto bottom_fmt =
                d.template duplicate_n<BottomDimensionHiddenIds::size()>(sep);
            constexpr auto top_fmt    = d.template duplicate_n<TopDimensionHiddenIds::size()>(sep);
            constexpr auto hidden_fmt = get_hidden_format();
            return make_str_literal("[ __") + bottom_fmt + make_str_literal("__ | ^^") + top_fmt +
                   make_str_literal("^^ | ") + hidden_fmt + make_str_literal(" ]");
        }
        CK_TILE_HOST_DEVICE static constexpr index_t get_num_values()
        {
            return BottomDimensionHiddenIds::size() + TopDimensionHiddenIds::size() + NDimHidden;
        }

        CK_TILE_HOST_DEVICE static constexpr auto get_values(const coord_t& coord)
        {
            return container_concat(
                coord.get_bottom_index(), coord.get_top_index(), coord.get_hidden_index());
        }
    };

    CK_TILE_HOST_DEVICE static constexpr auto get_prefix()
    {
        constexpr auto fmt_tid = make_str_literal("tid %03d: ");
        if constexpr(sizeof...(PREFIXChars) == 0)
            return fmt_tid;
        else
            return fmt_tid + make_str_literal(" ") + str_literal<PREFIXChars...>{};
    }
    CK_TILE_HOST_DEVICE static constexpr auto get_suffix()
    {
        constexpr auto lf = make_str_literal("\n");
        if constexpr(sizeof...(SUFFIXChars) == 0)
            return lf;
        else
            return str_literal<SUFFIXChars...>{} + lf;
    }

    template <char... FMTChars, typename TArgs, index_t... Is, typename... Args>
    CK_TILE_HOST_DEVICE void impl(str_literal<FMTChars...>,
                                  const TArgs& targs,
                                  std::integer_sequence<index_t, Is...>,
                                  Args&&... args) const
    {
        constexpr auto fmt_wrap_v = get_prefix() + str_literal<FMTChars...>{} + get_suffix();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
        printf(fmt_wrap_v.data, get_thread_id(), args..., targs.at(number<Is>())...);
#pragma clang diagnostic pop
    }
    template <typename T, typename... Args>
    CK_TILE_HOST_DEVICE void operator()(T&& x, Args&&... args) const
    {
        using detail_t = detail<remove_cvref_t<T>>;
        impl(detail_t::get_format(),
             detail_t::get_values(std::forward<T>(x)),
             std::make_integer_sequence<index_t, (detail_t::get_num_values())>{},
             std::forward<Args>(args)...);
    }
};
} // namespace detail

template <index_t N, typename B, typename T>
CK_TILE_HOST_DEVICE void print(const tensor_adaptor_coordinate<N, B, T>& coord)
{
    detail::CK_PRINT_X_<>{}(coord);
}
} // namespace ck_tile
