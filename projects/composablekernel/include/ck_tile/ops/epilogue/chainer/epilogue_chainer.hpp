// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

/// @brief Epilogue Chainer - Modular epilogue processing facilitator
///
/// @par Overview
///     EpilogueChainer provides an interface for processing epilogue operations
///     through schedules. The chainer uses decomposed epilogue operations, these are
///     scheduled/sequenced by a Scheduler to form operation graphs.
///
/// @tparam Scheduler The schedule provider that defines epilogue operation graphs
template <typename Scheduler>
struct EpilogueChainer
{
    using Problem = typename Scheduler::ProblemType;
    using BaseOp  = typename Scheduler::BaseOp;

    using ODataType                       = typename BaseOp::ODataType;
    using DsDataType                      = typename BaseOp::DsDataType;
    using DsLayout                        = typename BaseOp::DsLayout;
    using AccDataType                     = typename BaseOp::AccDataType;
    static constexpr auto MemoryOperation = BaseOp::MemoryOperation;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return BaseOp::GetSmemSize(); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC()
    {
        return BaseOp::GetVectorSizeC();
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> idx)
    {
        return BaseOp::GetVectorSizeD(idx);
    }

    CK_TILE_DEVICE static constexpr auto MakeLdsDistributionEncode()
    {
        return BaseOp::MakeLdsDistributionEncode();
    }

    /// @brief Process epilogue through scheduler-defined operation graph
    ///
    /// @par Flow
    ///     1. Create shared context through scheduler
    ///     2. Generate operation schedule based on arguments
    ///     3. Run scheduled operations in sequence
    template <typename OutWindow, typename AccTile, typename AuxWindows, typename... Args>
    CK_TILE_DEVICE void operator()(OutWindow& out_window,
                                   const AccTile& acc_tile,
                                   const AuxWindows& aux_windows,
                                   void* p_smem,
                                   Args&&... args) const
    {
        // The context serves as a shared workspace that maintains intermediate results
        // and resources across multiple epilogue operations.
        auto context  = Scheduler::create_context(out_window, acc_tile, aux_windows, p_smem);
        auto schedule = Scheduler::make_schedule(std::forward<Args>(args)...);
        schedule(out_window, acc_tile, aux_windows, p_smem, context);
    }
};

/// @brief Epilogue operation wrapper with arguments
///
/// @par Purpose
///     EpilogueNode wraps individual epilogue operations with their required arguments,
///     allowing them to be composed into operation graphs. Arguments are captured at construction
///     time and automatically forwarded during processing.
///
/// @tparam EpilogueType Epilogue operation (e.g., SliceEpilogue, ScaleEpilogue)
/// @tparam Args Types of arguments required by the epilogue operation
template <typename EpilogueType, typename... Args>
struct EpilogueNode
{
    using Epilogue = EpilogueType;
    ck_tile::tuple<Args...> args;

    constexpr EpilogueNode(Args... a) : args(a...) {}

    /// @brief Process epilogue without iteration index
    template <typename OutWindow, typename AccTile, typename AuxWindows, typename Context>
    CK_TILE_DEVICE void operator()(OutWindow& out_window,
                                   const AccTile& acc_tile,
                                   const AuxWindows& aux_windows,
                                   void* p_smem,
                                   Context& context) const
    {
        ck_tile::apply(
            [&](auto&&... epilogue_args) {
                EpilogueType{}(out_window,
                               acc_tile,
                               aux_windows,
                               p_smem,
                               context,
                               std::forward<decltype(epilogue_args)>(epilogue_args)...);
            },
            args);
    }

    /// @brief Process epilogue with iteration index
    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename Context,
              index_t I>
    CK_TILE_DEVICE void operator()(OutWindow& out_window,
                                   const AccTile& acc_tile,
                                   const AuxWindows& aux_windows,
                                   void* p_smem,
                                   Context& context,
                                   number<I> iAccess) const
    {
        ck_tile::apply(
            [&](auto&&... epilogue_args) {
                EpilogueType{}(out_window,
                               acc_tile,
                               aux_windows,
                               p_smem,
                               iAccess,
                               context,
                               std::forward<decltype(epilogue_args)>(epilogue_args)...);
            },
            args);
    }
};

/// @brief Specialization for epilogue operation wrapper with no arguments
template <typename EpilogueType>
struct EpilogueNode<EpilogueType>
{
    using Epilogue = EpilogueType;
    ck_tile::tuple<> args;

    constexpr EpilogueNode() = default;

    template <typename OutWindow, typename AccTile, typename AuxWindows, typename Context>
    CK_TILE_DEVICE void operator()(OutWindow& out_window,
                                   const AccTile& acc_tile,
                                   const AuxWindows& aux_windows,
                                   void* p_smem,
                                   Context& context) const
    {
        EpilogueType{}(out_window, acc_tile, aux_windows, p_smem, context);
    }

    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename Context,
              index_t I>
    CK_TILE_DEVICE void operator()(OutWindow& out_window,
                                   const AccTile& acc_tile,
                                   const AuxWindows& aux_windows,
                                   void* p_smem,
                                   Context& context,
                                   number<I> iAccess) const
    {
        EpilogueType{}(out_window, acc_tile, aux_windows, p_smem, iAccess, context);
    }
};

/// @brief Operation graph that sequentially composes multiple epilogue operations
///
/// @tparam Steps Number of iterations
/// @tparam EpilogueTypes Types of epilogue nodes in the operation graph
template <index_t Steps, typename... EpilogueTypes>
struct EpilogueGraph
{
    ck_tile::tuple<EpilogueTypes...> epilogues;

    constexpr EpilogueGraph() = default;
    constexpr EpilogueGraph(EpilogueTypes... eps) : epilogues(eps...) {}

    /// @brief Process all epilogues for each iteration in sequence
    template <typename OutWindow, typename AccTile, typename AuxWindows, typename Context>
    CK_TILE_DEVICE void operator()(OutWindow& out_window,
                                   const AccTile& acc_tile,
                                   const AuxWindows& aux_windows,
                                   void* p_smem,
                                   Context& context) const
    {
        // For each iteration, process all epilogues in order
        static_for<0, Steps, 1>{}([&](auto iAccess) {
            static_for<0, sizeof...(EpilogueTypes), 1>{}([&](auto I) {
                epilogues.template get<I.value>()(
                    out_window, acc_tile, aux_windows, p_smem, context, iAccess);
            });
        });
    }
};

/// @brief Helper function for creating epilogue nodes
template <typename EpilogueType, typename... Args>
constexpr auto make_node(Args... args)
{
    return EpilogueNode<EpilogueType, Args...>{args...};
}

/// @brief Helper function for creating operation graphs
template <index_t Steps, typename... EpilogueTypes>
constexpr auto make_graph(EpilogueTypes... epilogues)
{
    return EpilogueGraph<Steps, EpilogueTypes...>{epilogues...};
}

} // namespace ck_tile
