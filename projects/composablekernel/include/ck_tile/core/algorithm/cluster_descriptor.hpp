// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename Lengths,
          typename ArrangeOrder = typename arithmetic_sequence_gen<0, Lengths::size(), 1>::type>
CK_TILE_HOST_DEVICE constexpr auto make_cluster_descriptor(
    const Lengths& lengths,
    ArrangeOrder order = typename arithmetic_sequence_gen<0, Lengths::size(), 1>::type{})
{
    constexpr index_t ndim_low = Lengths::size();

    const auto reordered_lengths = container_reorder_given_new2old(lengths, order);

    const auto low_lengths = generate_tuple(
        [&](auto idim_low) { return reordered_lengths[idim_low]; }, number<ndim_low>{});

    const auto transform = make_merge_transform(low_lengths);

    constexpr auto low_dim_old_top_ids = ArrangeOrder{};

    constexpr auto up_dim_new_top_ids = sequence<0>{};

    return make_single_stage_tensor_adaptor(
        make_tuple(transform), make_tuple(low_dim_old_top_ids), make_tuple(up_dim_new_top_ids));
}

} // namespace ck_tile
