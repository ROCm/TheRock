// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/epilogue/chainer/epilogue_chainer.hpp"
#include "ck_tile/ops/epilogue/chainer/common_epilogue_ops.hpp"
#include "ck_tile/ops/epilogue/chainer/cshuffle_epilogue_chainer_ops.hpp"

namespace ck_tile {

/// @brief Schedule type tags for epilogue selection
/// @par Purpose
///     Each tag corresponds to a pre-built schedule, these are used to select a schedule

/// Standard epilogue schedule: Slice → CastStore → Load → ApplyD → Store → Move
struct DefaultScheduleTag
{
};

/// RowCol quantization schedule: Slice → ScaleWindow → CastStore → Load → ApplyD → Store → Move
struct RowColQuantScheduleTag
{
};

/// Tensor quantization schedule: Slice → ScaleScalar → CastStore → Load → ApplyD → Store → Move
struct TensorQuantScheduleTag
{
};

/// @brief CShuffle epilogue scheduler providing pre-built schedules
///
/// @par Overview
///     CshuffleEpilogueSchedule acts as the scheduler component for EpilogueChainer.
///     It provides context creation and pre-built schedules. The scheduler
///     uses tags to select/create appropriate epilogue schedule.
///
/// @tparam Problem The epilogue problem configuration
/// @tparam ScheduleTag Tag selecting the epilogue schedule
template <typename Problem, typename ScheduleTag = DefaultScheduleTag>
struct CshuffleEpilogueSchedule
{
    using ProblemType = Problem;
    using BaseOp      = CShuffleEpilogueChainBaseOp<Problem>;

    static constexpr index_t NumAccess = BaseOp::SFC::get_num_of_access();

    /// @brief Create context for epilogue operations
    template <typename OutWindow, typename AccTile, typename AuxWindows>
    CK_TILE_DEVICE static auto create_context(OutWindow& out_window,
                                              const AccTile& acc_tile,
                                              const AuxWindows& aux_windows,
                                              void* p_smem)
    {
        return BaseOp{}(out_window, acc_tile, aux_windows, p_smem);
    }

    /// @brief Make schedule based on compile-time tag selection
    template <typename... Args>
    CK_TILE_DEVICE static auto make_schedule(Args&&... args)
    {
        if constexpr(std::is_same_v<ScheduleTag, DefaultScheduleTag>)
        {
            // Standard epilogue
            // Schedule: Slice -> CastAndStoreLds -> Load -> ApplyD -> Store -> MoveWindows
            static_assert(sizeof...(args) == 0, "DefaultSchedule expects no arguments");
            return make_graph<NumAccess>(
                make_node<CShuffleSliceOp<typename BaseOp::SFC,
                                          typename BaseOp::CWarpDstr,
                                          BaseOp::NumMXdlPerWavePerShuffle,
                                          BaseOp::NumNXdlPerWavePerShuffle,
                                          BaseOp::MPerIterationShuffle,
                                          BaseOp::NPerIterationShuffle>>(),
                make_node<CastAndStoreToLdsOp<typename BaseOp::ODataType>>(),
                make_node<LoadFromLdsOp<typename BaseOp::TileEncodingPattern>>(),
                make_node<ElementwiseOp<typename Problem::CDElementwise, Problem::NumDTensor>>(),
                make_node<StoreOp<Problem::MemoryOperation>>(),
                make_node<MoveWindowsOp<typename BaseOp::SFC, Problem::NumDTensor>>());
        }
        else if constexpr(std::is_same_v<ScheduleTag, RowColQuantScheduleTag>)
        {
            // RowCol quantization schedule with tensor windows
            // Schedule: Slice -> ScaleWindow -> CastAndStoreLds -> Load -> ApplyD -> Store ->
            // MoveWindows
            static_assert(sizeof...(args) == 2,
                          "RowColQuantSchedule requires exactly 2 scale tensor arguments");
            return make_graph<NumAccess>(
                make_node<CShuffleSliceOp<typename BaseOp::SFC,
                                          typename BaseOp::CWarpDstr,
                                          BaseOp::NumMXdlPerWavePerShuffle,
                                          BaseOp::NumNXdlPerWavePerShuffle,
                                          BaseOp::MPerIterationShuffle,
                                          BaseOp::NPerIterationShuffle>>(),
                make_node<CShuffleScaleWindowOp<typename BaseOp::SFC>>(std::forward<Args>(args)...),
                make_node<CastAndStoreToLdsOp<typename BaseOp::ODataType>>(),
                make_node<LoadFromLdsOp<typename BaseOp::TileEncodingPattern>>(),
                make_node<ElementwiseOp<typename Problem::CDElementwise, Problem::NumDTensor>>(),
                make_node<StoreOp<Problem::MemoryOperation>>(),
                make_node<MoveWindowsOp<typename BaseOp::SFC, Problem::NumDTensor>>());
        }
        else if constexpr(std::is_same_v<ScheduleTag, TensorQuantScheduleTag>)
        {
            // Tensor quantization schedule with scalar values
            // Schedule: Slice -> ScaleScalar -> CastAndStoreLds -> Load -> ApplyD -> Store ->
            // MoveWindows
            static_assert(sizeof...(args) == 2,
                          "TensorQuantSchedule requires exactly 2 scalar arguments");
            return make_graph<NumAccess>(
                make_node<CShuffleSliceOp<typename BaseOp::SFC,
                                          typename BaseOp::CWarpDstr,
                                          BaseOp::NumMXdlPerWavePerShuffle,
                                          BaseOp::NumNXdlPerWavePerShuffle,
                                          BaseOp::MPerIterationShuffle,
                                          BaseOp::NPerIterationShuffle>>(),
                make_node<ScaleScalarOp>(std::forward<Args>(args)...),
                make_node<CastAndStoreToLdsOp<typename BaseOp::ODataType>>(),
                make_node<LoadFromLdsOp<typename BaseOp::TileEncodingPattern>>(),
                make_node<ElementwiseOp<typename Problem::CDElementwise, Problem::NumDTensor>>(),
                make_node<StoreOp<Problem::MemoryOperation>>(),
                make_node<MoveWindowsOp<typename BaseOp::SFC, Problem::NumDTensor>>());
        }
        else
        {
            static_assert(false, "Unknown schedule tag");
        }
    }
};

} // namespace ck_tile
