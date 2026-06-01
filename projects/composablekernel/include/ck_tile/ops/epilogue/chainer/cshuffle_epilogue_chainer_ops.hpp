// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#include <optional>

namespace ck_tile {

//------------------------------------------------------------------------------
// CShuffle-specific epilogue operations
// These operations are specific to CShuffle epilogue due to its unique.
//------------------------------------------------------------------------------

/// @brief Slice accumulator tile for CShuffle epilogue
///
/// @par Purpose
///     Extracts a portion of the accumulator tile into the working tile
///     based on the current iteration index. This is CShuffle-specific.
///
/// @tparam SFC Space filling curve type
/// @tparam CWarpDstr Warp distribution type
/// @tparam NumMXdlPerWavePerShuffle XDL tiles in M per wave per shuffle
/// @tparam NumNXdlPerWavePerShuffle XDL tiles in N per wave per shuffle
/// @tparam MPerIterShuffle M elements per shuffle iteration
/// @tparam NPerIterShuffle N elements per shuffle iteration
template <typename SFC,
          typename CWarpDstr,
          index_t NumMXdlPerWavePerShuffle,
          index_t NumNXdlPerWavePerShuffle,
          index_t MPerIterShuffle,
          index_t NPerIterShuffle>
struct CShuffleSliceOp
{
    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename IAccess,
              typename Context>
    CK_TILE_DEVICE void operator()([[maybe_unused]] OutWindow& out_window,
                                   const AccTile& acc_tile,
                                   [[maybe_unused]] const AuxWindows& aux_windows,
                                   [[maybe_unused]] void* p_smem,
                                   IAccess iAccess,
                                   Context& context)
    {
        constexpr auto idx_start = SFC::get_index(iAccess);
        constexpr auto m_iter    = number<idx_start.at(number<0>{}) / MPerIterShuffle>{};
        constexpr auto n_iter    = number<idx_start.at(number<1>{}) / NPerIterShuffle>{};

        constexpr auto warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        context.working_tile.get_thread_buffer() = acc_tile.get_y_sliced_thread_data(
            merge_sequences(
                sequence<m_iter * NumMXdlPerWavePerShuffle, n_iter * NumNXdlPerWavePerShuffle>{},
                warp_y_index_zeros),
            merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                            warp_y_lengths));
    }
};

/// @brief Scale working tile using tensor windows (CShuffle-specific)
///
/// @par Purpose
///     Scales the working tile using row and column scale tensors.
///     CShuffle-specific because it creates scale windows from the
///     working tile's distribution and handles window movement.
///
/// @tparam SFC Space filling curve type
template <typename SFC>
struct CShuffleScaleWindowOp
{
    template <typename OutWindow,
              typename AccTile,
              typename AuxWindows,
              typename IAccess,
              typename Context,
              typename ScaleRowTensor,
              typename ScaleColTensor>
    CK_TILE_DEVICE void operator()([[maybe_unused]] OutWindow& out_window,
                                   [[maybe_unused]] const AccTile& acc_tile,
                                   [[maybe_unused]] const AuxWindows& aux_windows,
                                   [[maybe_unused]] void* p_smem,
                                   IAccess iAccess,
                                   Context& context,
                                   const ScaleRowTensor& scale_row_tensor,
                                   const ScaleColTensor& scale_col_tensor)
    {
        auto scale_row_window =
            make_tile_window(scale_row_tensor, context.working_tile.get_tile_distribution());
        auto scale_col_window =
            make_tile_window(scale_col_tensor, context.working_tile.get_tile_distribution());

        const auto scale_row_tile = load_tile(scale_row_window);
        const auto scale_col_tile = load_tile(scale_col_window);

        tile_elementwise_inout(element_wise::MultiDMultiply{},
                               context.working_tile,
                               context.working_tile,
                               scale_row_tile,
                               scale_col_tile);

        constexpr index_t num_access = SFC::get_num_of_access();
        if constexpr(iAccess != num_access - 1)
        {
            constexpr auto step = SFC::get_forward_step(number<iAccess>{});
            move_tile_window(scale_row_window, {step.at(number<0>{}), step.at(number<1>{})});
            move_tile_window(scale_col_window, {step.at(number<0>{}), step.at(number<1>{})});
        }
    }
};

//------------------------------------------------------------------------------
// CShuffle problem and base operation definitions
//------------------------------------------------------------------------------

/// @brief Problem configuration for CShuffle epilogue chainer operations
/// @note Mirrors CShuffleEpilogueProblem but uses AsDataType/BsDataType for tuple support
template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename DsLayout_,
          typename ELayout_,
          typename CDElementwise_,
          index_t kM_,
          index_t kN_,
          index_t MWave_,
          index_t NWave_,
          index_t MPerXdl_,
          index_t NPerXdl_,
          index_t KPerXdl_,
          bool isCTransposed_,
          memory_operation_enum MemoryOperation_,
          index_t kNumWaveGroups_      = 1,
          bool FixedVectorSize_        = false,
          index_t VectorSizeC_         = 1,
          bool TiledMMAPermuteN_       = false,
          index_t BlockedXDLN_PerWarp_ = 1>
struct CShuffleEpilogueChainProblem
{
    using AsDataType                                       = remove_cvref_t<AsDataType_>;
    using BsDataType                                       = remove_cvref_t<BsDataType_>;
    using AccDataType                                      = remove_cvref_t<AccDataType_>;
    using ODataType                                        = remove_cvref_t<ODataType_>;
    using DsDataType                                       = remove_cvref_t<DsDataType_>;
    using DsLayout                                         = remove_cvref_t<DsLayout_>;
    using ELayout                                          = remove_cvref_t<ELayout_>;
    using CDElementwise                                    = remove_cvref_t<CDElementwise_>;
    static constexpr index_t kBlockSize                    = MWave_ * NWave_ * get_warp_size();
    static constexpr index_t kMPerBlock                    = kM_;
    static constexpr index_t kNPerBlock                    = kN_;
    static constexpr index_t MWave                         = MWave_;
    static constexpr index_t NWave                         = NWave_;
    static constexpr index_t MPerXdl                       = MPerXdl_;
    static constexpr index_t NPerXdl                       = NPerXdl_;
    static constexpr index_t KPerXdl                       = KPerXdl_;
    static constexpr index_t isCTransposed                 = isCTransposed_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
    static constexpr bool FixedVectorSize                  = FixedVectorSize_;
    static constexpr index_t VectorSizeC                   = VectorSizeC_;
    static constexpr index_t BlockedXDLN_PerWarp           = BlockedXDLN_PerWarp_;
    static constexpr bool TiledMMAPermuteN                 = TiledMMAPermuteN_;
    static constexpr index_t kNumWaveGroups                = kNumWaveGroups_;
    static constexpr index_t NumDTensor                    = DsDataType::size();

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
};

template <typename Problem_, typename Policy_ = void>
struct CShuffleEpilogueChainBaseOp
{
    using Problem     = remove_cvref_t<Problem_>;
    using AsDataType  = remove_cvref_t<typename Problem::AsDataType>;
    using BsDataType  = remove_cvref_t<typename Problem::BsDataType>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;
    using DsDataType  = remove_cvref_t<typename Problem::DsDataType>;
    using DsLayout    = remove_cvref_t<typename Problem::DsLayout>;

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

    using ATypeToUse =
        std::conditional_t<std::is_same_v<ADataType, pk_int4_t>, BDataType, ADataType>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
    using ELayout       = remove_cvref_t<typename Problem::ELayout>;
    using CDElementwise = remove_cvref_t<typename Problem::CDElementwise>;
    static constexpr memory_operation_enum MemoryOperation = Problem::MemoryOperation;
    static constexpr index_t kBlockSize                    = Problem::kBlockSize;
    static constexpr index_t kMPerBlock                    = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock                    = Problem::kNPerBlock;
    static constexpr index_t MWave                         = Problem::MWave;
    static constexpr index_t NWave                         = Problem::NWave;
    static constexpr index_t MPerXdl                       = Problem::MPerXdl;
    static constexpr index_t NPerXdl                       = Problem::NPerXdl;
    static constexpr index_t KPerXdl                       = Problem::KPerXdl;
    static constexpr index_t isCTransposed                 = Problem::isCTransposed;
    static constexpr bool FixedVectorSize                  = Problem::FixedVectorSize;
    static constexpr bool TiledMMAPermuteN                 = Problem::TiledMMAPermuteN;
    static constexpr index_t BlockedXDLN_PerWarp           = Problem::BlockedXDLN_PerWarp;
    static constexpr index_t VectorSizeC                   = Problem::VectorSizeC;
    static constexpr index_t MPerIteration                 = MPerXdl * MWave;
    static constexpr index_t NPerIteration                 = NPerXdl * NWave;
    static constexpr index_t NumDTensor                    = Problem::NumDTensor;

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");

    /**
     * @brief Get the vector store size for C tensor.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC()
    {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeC;
        }
        constexpr index_t max_vector_size = 16;
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    /**
     * @brief Get the vector store size for Di tensor.
     *
     * @return The vector store size for Di tensor.
     */
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> index)
    {
        constexpr index_t max_vector_size = 16;
        using DiDataType = remove_cvref_t<std::tuple_element_t<index.value, DsDataType>>;
        using DiLayout   = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
        if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else
        {
            static_assert(false, "Unsupported DLayout!");
        }
    }
    /**
     * @brief Shuffle tile configuration parameters
     *
     * @details These parameters control the number of XDL tiles processed per wave in each shuffle
     * iteration:
     * - NumMXdlPerWavePerShuffle: Number of XDL tiles in M dimension processed per wave
     * - NumNXdlPerWavePerShuffle: Number of XDL tiles in N dimension processed per wave
     */
    static constexpr auto shuffle_tile_tuple = [] {
        constexpr index_t elem_per_thread = MPerXdl * NPerXdl / get_warp_size();
        if constexpr(elem_per_thread >= GetVectorSizeC())
        {
            return std::make_tuple(1, 1);
        }
        else
        {
            constexpr index_t num_xdl_shuffles = GetVectorSizeC() / elem_per_thread;
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                static_assert((kMPerBlock % (MPerXdl * MWave) == 0) &&
                                  (kMPerBlock % num_xdl_shuffles == 0),
                              "kMPerBlock must be divisible by MPerXdl*MWave and "
                              "num_xdl_shuffles for CShuffleEpilogueStageBase");
                return std::make_tuple(min(num_xdl_shuffles, kMPerBlock / (MPerXdl * MWave)), 1);
            }
            else
            {
                static_assert((kNPerBlock % (NPerXdl * NWave) == 0) &&
                                  (kNPerBlock % num_xdl_shuffles == 0),
                              "kNPerBlock must be divisible by NPerXdl*NWave and "
                              "num_xdl_shuffles for CShuffleEpilogueStageBase");
                return std::make_tuple(1, min(num_xdl_shuffles, kNPerBlock / (NPerXdl * NWave)));
            }
        }
    }();
    static constexpr index_t NumMXdlPerWavePerShuffle = std::get<0>(shuffle_tile_tuple);
    static constexpr index_t NumNXdlPerWavePerShuffle =
        max(BlockedXDLN_PerWarp, std::get<1>(shuffle_tile_tuple));

    static constexpr auto MNPerIterationShuffle = [] {
        constexpr index_t m_val = MPerXdl * MWave * NumMXdlPerWavePerShuffle;
        constexpr index_t n_val = NPerXdl * NWave * NumNXdlPerWavePerShuffle;
        if constexpr(kMPerBlock % m_val != 0 || kNPerBlock % n_val != 0)
            return std::make_tuple(MPerXdl * MWave, NPerXdl * NWave);
        else
            return std::make_tuple(m_val, n_val);
    }();
    static constexpr index_t MPerIterationShuffle = std::get<0>(MNPerIterationShuffle);
    static constexpr index_t NPerIterationShuffle = std::get<1>(MNPerIterationShuffle);

    using WG = WarpGemmDispatcher<ATypeToUse,
                                  BTypeToUse,
                                  AccDataType,
                                  MPerXdl,
                                  NPerXdl,
                                  KPerXdl,
                                  isCTransposed>;

    using CWarpDstr         = typename WG::CWarpDstr;
    using CWarpTensor       = typename WG::CWarpTensor;
    using CWarpDstrEncoding = typename WG::CWarpDstrEncoding;
    using SFC               = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                                  sequence<0, 1>,
                                                  sequence<MPerIterationShuffle, NPerIterationShuffle>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<NPerIterationShuffle>{}, number<1>{}));
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<1>{}, number<MPerIterationShuffle>{}));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    CK_TILE_DEVICE static constexpr auto MakeLdsDistributionEncode()
    {
        constexpr auto block_outer_dstr_encoding = [] {
            if constexpr(BlockedXDLN_PerWarp == 1)
            {
                return tile_distribution_encoding<sequence<>,
                                                  tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                                                        sequence<NumNXdlPerWavePerShuffle, NWave>>,
                                                  tuple<sequence<1, 2>>,
                                                  tuple<sequence<1, 1>>,
                                                  sequence<1, 2>,
                                                  sequence<0, 0>>{};
            }
            else
            {
                constexpr int RakedXDLN_PerWarp = NumNXdlPerWavePerShuffle / BlockedXDLN_PerWarp;
                // BlockedLayout
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                          sequence<RakedXDLN_PerWarp, NWave, BlockedXDLN_PerWarp>>,
                    tuple<sequence<1, 2>>,
                    tuple<sequence<1, 1>>,
                    sequence<1, 2, 2>,
                    sequence<0, 0, 2>>{};
            }
        }();
        constexpr auto block_dstr_encoding = detail::make_embed_tile_distribution_encoding(
            block_outer_dstr_encoding, typename CWarpDstr::DstrEncode{});

        return block_dstr_encoding;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
    }

    using TileEncodingPattern =
        tile_distribution_encoding_pattern_2d<kBlockSize,
                                              MPerIterationShuffle,
                                              NPerIterationShuffle,
                                              GetVectorSizeC(),
                                              tile_distribution_pattern::thread_raked,
                                              Problem::kNumWaveGroups>;

    /// @brief Context structure for CShuffle epilogue operations
    ///
    /// @par Purpose
    ///     The context serves as a shared workspace that maintains intermediate results
    ///     and resources across multiple epilogue operations. It eliminates the need for
    ///     operations to recreate shared data structures and enables data flow
    ///     through the operation graph.
    ///
    /// @par Standardized Interface
    ///     Uses standardized member names so common operations can work with this context:
    ///     - working_tile: Intermediate tile for shuffle operations
    ///     - out_tile: Output tile for final results
    ///     - aux_windows: Auxiliary tensor windows (D tensors)
    ///     - lds_write_window: Window for writing to LDS
    ///     - lds_read_window: Window for reading from LDS
    template <typename WorkingTileType,
              typename LdsBlockType,
              typename LdsWriteWindowType,
              typename LdsReadWindowType,
              typename AuxWindowsType,
              typename OutTileType>
    struct CShuffleContext
    {
        WorkingTileType working_tile;        // Working tile for shuffle operations
        LdsBlockType lds_block;              // LDS block view
        LdsWriteWindowType lds_write_window; // Window for writing to LDS
        LdsReadWindowType lds_read_window;   // Window for reading from LDS
        AuxWindowsType aux_windows;          // Auxiliary tensor windows (D tensors)
        OutTileType out_tile;                // Output tile
    };

    template <typename OutDramWindow, typename AccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto operator()([[maybe_unused]] OutDramWindow& out_window,
                                   [[maybe_unused]] const AccTile& acc_tile,
                                   const DsDramWindows& ds_windows,
                                   void* p_smem)
    {
        static_assert(
            std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
            "Currently, the CShuffleEpilogueStageBase only supports the Row Major Output layout");

        constexpr auto working_tile_distr =
            make_static_tile_distribution(MakeLdsDistributionEncode());
        auto working_tile = make_static_distributed_tensor<AccDataType>(working_tile_distr);

        constexpr auto lds_block_desc = MakeLdsBlockDescriptor<Problem>();
        auto lds_block = make_tensor_view<address_space_enum::lds>(static_cast<ODataType*>(p_smem),
                                                                   lds_block_desc);

        auto lds_write_window = make_tile_window(
            lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0},
            working_tile_distr);

        auto lds_read_window = make_tile_window(
            lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0});

        constexpr auto dram_tile_distribution =
            TileEncodingPattern::make_2d_static_tile_distribution();
        auto aux_windows = generate_tuple(
            [&](auto idx) { return make_tile_window(ds_windows[idx], dram_tile_distribution); },
            number<NumDTensor>{});

        auto out_tile = load_tile(make_tile_window(lds_read_window, dram_tile_distribution));

        using ContextType = CShuffleContext<decltype(working_tile),
                                            decltype(lds_block),
                                            decltype(lds_write_window),
                                            decltype(lds_read_window),
                                            decltype(aux_windows),
                                            decltype(out_tile)>;

        ContextType context;
        context.working_tile     = working_tile;
        context.lds_block        = lds_block;
        context.lds_write_window = lds_write_window;
        context.lds_read_window  = lds_read_window;
        context.aux_windows      = aux_windows;
        context.out_tile         = out_tile;

        return context;
    }
};

} // namespace ck_tile
