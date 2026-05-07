// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

/// @file common_epilogue_ops.hpp
/// @brief  Reusable simple epilogue operations which might be used to compose more complex one.
///
///
/// @par Context Interface
///     Operations expect the context to provide:
///     - working_tile: Tile for intermediate computations
///     - out_tile: Output tile for final results
///     - aux_windows: Tuple of auxiliary tensor windows (e.g., D tensors)
///     - lds_write_window: Window for writing to LDS (if using LDS)
///     - lds_read_window: Window for reading from LDS (if using LDS)
namespace ck_tile {

/// @brief Scale working tile by scalar values
///
/// @par Context Requirements
///     working_tile: Tile to scale
struct ScaleScalarOp
{
    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename IAccess,
              typename Context,
              typename ScaleA,
              typename ScaleB>
    CK_TILE_DEVICE void operator()([[maybe_unused]] OutWindow& out_window,
                                   [[maybe_unused]] const AccTile& acc_tile,
                                   [[maybe_unused]] const AuxWindows& aux_windows,
                                   [[maybe_unused]] void* p_smem,
                                   [[maybe_unused]] IAccess iAccess,
                                   Context& context,
                                   const ScaleA& scale_a,
                                   const ScaleB& scale_b)
    {
        tile_elementwise_inout([&](auto& elem) { elem = elem * scale_a * scale_b; },
                               context.working_tile);
    }
};

/// @brief Cast working tile and store to LDS
///
/// @tparam DataType Target data type for casting
///
/// @par Context Requirements
///     working_tile: Tile to cast
///     lds_write_window: Window for writing to LDS
template <typename DataType>
struct CastAndStoreToLdsOp
{
    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename IAccess,
              typename Context>
    CK_TILE_DEVICE void operator()([[maybe_unused]] OutWindow& out_window,
                                   [[maybe_unused]] const AccTile& acc_tile,
                                   [[maybe_unused]] const AuxWindows& aux_windows,
                                   [[maybe_unused]] void* p_smem,
                                   [[maybe_unused]] IAccess iAccess,
                                   Context& context)
    {
        const auto casted_tile = cast_tile<DataType>(context.working_tile);
        store_tile(context.lds_write_window, casted_tile);
    }
};

/// @brief Load output tile from LDS with synchronization
///
/// @tparam TileEncodingPattern Pattern for tile distribution
///
/// @par Context Requirements
///     lds_read_window: Window for reading from LDS
///     out_tile: Destination for loaded tile
template <typename TileEncodingPattern>
struct LoadFromLdsOp
{
    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename IAccess,
              typename Context>
    CK_TILE_DEVICE void operator()([[maybe_unused]] OutWindow& out_window,
                                   [[maybe_unused]] const AccTile& acc_tile,
                                   [[maybe_unused]] const AuxWindows& aux_windows,
                                   [[maybe_unused]] void* p_smem,
                                   [[maybe_unused]] IAccess iAccess,
                                   Context& context)
    {
        constexpr auto tile_distribution = TileEncodingPattern::make_2d_static_tile_distribution();
        block_sync_lds();
        context.out_tile = load_tile(make_tile_window(context.lds_read_window, tile_distribution));
    }
};

/// @brief Apply elementwise operation with auxiliary tensors
///
/// @tparam Elementwise Elementwise functor type
/// @tparam NumAux Number of auxiliary tensors to load and apply
///
/// @par Context Requirements
///     out_tile: In/out tile for elementwise operation
///     aux_windows: Tuple of auxiliary tensor windows
template <typename Elementwise, index_t NumAux>
struct ElementwiseOp
{
    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename IAccess,
              typename Context>
    CK_TILE_DEVICE void operator()([[maybe_unused]] OutWindow& out_window,
                                   [[maybe_unused]] const AccTile& acc_tile,
                                   [[maybe_unused]] const AuxWindows& aux_windows,
                                   [[maybe_unused]] void* p_smem,
                                   [[maybe_unused]] IAccess iAccess,
                                   Context& context)
    {
        const auto aux_tiles = generate_tuple(
            [&](auto idx) { return load_tile(context.aux_windows[idx]); }, number<NumAux>{});

        const auto tiles = concat_tuple_of_reference(
            tie(context.out_tile, context.out_tile),
            generate_tie([&](auto idx) -> const auto& { return aux_tiles[idx]; },
                         number<NumAux>{}));

        tile_elementwise_inout_unpack(Elementwise{}, tiles);
    }
};

/// @brief Store output tile to global memory
///
/// @tparam MemOp Memory operation type (set or atomic_add)
///
/// @par Context Requirements
///     out_tile: Tile to store
template <memory_operation_enum MemOp>
struct StoreOp
{
    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename IAccess,
              typename Context>
    CK_TILE_DEVICE void operator()(OutWindow& out_window,
                                   [[maybe_unused]] const AccTile& acc_tile,
                                   [[maybe_unused]] const AuxWindows& aux_windows,
                                   [[maybe_unused]] void* p_smem,
                                   [[maybe_unused]] IAccess iAccess,
                                   Context& context)
    {
        if constexpr(MemOp == memory_operation_enum::set)
        {
            store_tile(out_window, context.out_tile);
        }
        else
        {
            update_tile(out_window, context.out_tile);
        }
    }
};

/// @brief Move output and auxiliary windows by step from space-filling curve
///
/// @tparam SFC Space filling curve type providing step computation
/// @tparam NumAux Number of auxiliary windows to move
///
/// @par Context Requirements
///     aux_windows: Tuple of windows to move
template <typename SFC, index_t NumAux>
struct MoveWindowsOp
{
    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename IAccess,
              typename Context>
    CK_TILE_DEVICE void operator()(OutWindow& out_window,
                                   [[maybe_unused]] const AccTile& acc_tile,
                                   [[maybe_unused]] const AuxWindows& aux_windows,
                                   [[maybe_unused]] void* p_smem,
                                   IAccess iAccess,
                                   Context& context)
    {
        constexpr index_t num_access = SFC::get_num_of_access();
        if constexpr(iAccess != num_access - 1)
        {
            constexpr auto step = SFC::get_forward_step(iAccess);

            move_tile_window(out_window, {step.at(number<0>{}), step.at(number<1>{})});

            static_for<0, NumAux, 1>{}([&](auto idx) {
                move_tile_window(context.aux_windows[idx],
                                 {step.at(number<0>{}), step.at(number<1>{})});
            });
        }
    }
};

} // namespace ck_tile
