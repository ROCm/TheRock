// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

enum class GroupedConvDirection
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHT
};

/// @brief The Grouped Conv kernel host arguments.
///
/// @par Overview
///      This structure is passed to Grouped Convolution Kernels when creating kernel
///      arguments object. It contain all necessary information required to
///      build proper kernel argument and launch kernel on GPU.
template <typename InPtr, typename WeiPtr, typename OutPtr, typename CDElementwise>
struct GroupedConvHostArgs : public conv::ConvParam
{
    CK_TILE_HOST GroupedConvHostArgs() = delete;
    CK_TILE_HOST GroupedConvHostArgs(ConvParam conv_param,
                                     InPtr in_ptr_,
                                     WeiPtr wei_ptr_,
                                     const std::vector<const void*> ds_ptr_,
                                     OutPtr out_ptr_,
                                     index_t k_batch_,
                                     CDElementwise elfunc_ = CDElementwise{})
        : conv::ConvParam(conv_param),
          in_ptr(in_ptr_),
          wei_ptr(wei_ptr_),
          ds_ptr(ds_ptr_),
          out_ptr(out_ptr_),
          k_batch(k_batch_),
          elfunc(elfunc_)
    {
    }

    InPtr in_ptr;
    WeiPtr wei_ptr;
    const std::vector<const void*> ds_ptr;
    OutPtr out_ptr;
    index_t k_batch;
    const CDElementwise elfunc;
};

using PassThrough = ck_tile::element_wise::PassThrough;

template <typename CDElementwise = PassThrough>
using GroupedConvFwdHostArgs = GroupedConvHostArgs<const void*, const void*, void*, CDElementwise>;
using GroupedConvBwdWeightHostArgs =
    GroupedConvHostArgs<const void*, void*, const void*, PassThrough>;
using GroupedConvBwdDataHostArgs =
    GroupedConvHostArgs<void*, const void*, const void*, PassThrough>;

template <index_t NDimSpatial_,
          ConvolutionSpecialization ConvSpecialization_,
          typename InLayout_,
          typename WeiLayout_,
          typename DsLayout_,
          typename OutLayout_,
          index_t VectorSizeA_      = 1,
          index_t VectorSizeB_      = 1,
          index_t VectorSizeC_      = 1,
          index_t NumGroupsToMerge_ = 1,
          bool EnableSplitImage_    = false,
          bool ExplicitGemm_        = false>
struct GroupedConvTraits
{
    private:
    static constexpr auto generate_implicit_gemm_layout()
    {
        return generate_tuple([](auto) { return ck_tile::tensor_layout::gemm::RowMajor{}; },
                              number<DsLayout_::size()>{});
    }

    public:
    // Fixed values for Implicit GEMM
    struct FixedGemmParams
    {
        static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
        static constexpr ck_tile::index_t TilePartitionerM01      = 4;
        static constexpr bool kPadM                               = true;
        static constexpr bool kPadN                               = true;
        static constexpr bool kPadK                               = true;
        static constexpr bool TransposeC                          = false;
        static constexpr bool FixedVectorSize                     = true;
        static constexpr bool UseStructuredSparsity               = false;
        static constexpr bool Persistent                          = false;
        using ELayout = ck_tile::tensor_layout::gemm::RowMajor;
    };
    // Compile time parameters
    static constexpr index_t NumGroupsToMerge                     = NumGroupsToMerge_;
    static constexpr bool EnableSplitImage                        = EnableSplitImage_;
    static constexpr bool ExplicitGemm                            = ExplicitGemm_;
    static constexpr index_t NDimSpatial                          = NDimSpatial_;
    static constexpr ConvolutionSpecialization ConvSpecialization = ConvSpecialization_;
    using InLayout                                                = InLayout_;
    using WeiLayout                                               = WeiLayout_;
    using DsLayout                                                = DsLayout_;
    using OutLayout                                               = OutLayout_;

    // Forward Gemm Layouts
    using AsLayoutFwd = ck_tile::tensor_layout::gemm::RowMajor;
    using BsLayoutFwd = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayoutFwd  = ck_tile::tensor_layout::gemm::RowMajor;
    // Backward Data Gemm Layouts
    using AsLayoutBwdData = ck_tile::tensor_layout::gemm::RowMajor;
    using BsLayoutBwdData = ck_tile::tensor_layout::gemm::RowMajor;
    using CLayoutBwdData  = ck_tile::tensor_layout::gemm::RowMajor;
    // Backward Weight Gemm Layouts
    using AsLayoutBwdWeight = ck_tile::tensor_layout::gemm::ColumnMajor;
    using BsLayoutBwdWeight = ck_tile::tensor_layout::gemm::RowMajor;
    using CLayoutBwdWeight  = ck_tile::tensor_layout::gemm::RowMajor;

    template <GroupedConvDirection Direction>
    struct GemmLayouts
    {
        static_assert(false, "Unsupported direction.");
    };

    template <>
    struct GemmLayouts<GroupedConvDirection::FORWARD>
    {
        using AsLayout = AsLayoutFwd;
        using BsLayout = BsLayoutFwd;
        using CLayout  = CLayoutFwd;
    };

    template <>
    struct GemmLayouts<GroupedConvDirection::BACKWARD_DATA>
    {
        using AsLayout = AsLayoutBwdData;
        using BsLayout = BsLayoutBwdData;
        using CLayout  = CLayoutBwdData;
    };

    template <>
    struct GemmLayouts<GroupedConvDirection::BACKWARD_WEIGHT>
    {
        using AsLayout = AsLayoutBwdWeight;
        using BsLayout = BsLayoutBwdWeight;
        using CLayout  = CLayoutBwdWeight;
    };

    template <ck_tile::index_t NumWaveGroups = 1>
    using GroupedConvImplicitGemmTraitsFwd =
        TileGemmTraits<true, true, true, AsLayoutFwd, BsLayoutFwd, CLayoutFwd, NumWaveGroups>;
    template <ck_tile::index_t NumWaveGroups = 1>
    using GroupedConvImplicitGemmTraitsBwdData = TileGemmTraits<true,
                                                                true,
                                                                true,
                                                                AsLayoutBwdData,
                                                                BsLayoutBwdData,
                                                                CLayoutBwdData,
                                                                NumWaveGroups>;
    template <ck_tile::index_t NumWaveGroups = 1>
    using GroupedConvImplicitGemmTraitsBwdWeight  = TileGemmTraits<true,
                                                                   true,
                                                                   true,
                                                                   AsLayoutBwdWeight,
                                                                   BsLayoutBwdWeight,
                                                                   CLayoutBwdWeight,
                                                                   NumWaveGroups>;
    static constexpr ck_tile::index_t VectorSizeA = VectorSizeA_;
    static constexpr ck_tile::index_t VectorSizeB = VectorSizeB_;
    static constexpr ck_tile::index_t VectorSizeC = VectorSizeC_;
    static constexpr ck_tile::index_t NumDTensor  = DsLayout::size();
    using ImplicitGemmDsLayout                    = decltype(generate_implicit_gemm_layout());
};

/// @brief Helper struct for split-image piece information
///
/// @par Overview
///      Stores metadata for a single spatial piece in split-image convolution.
///      Used to track block ranges and spatial coordinates for each piece.
struct SplitImagePieceInfo
{
    ck_tile::index_t block_start, block_end;    ///< GPU block range for this piece
    ck_tile::index_t d_start, h_start, w_start; ///< Spatial start coordinates (output space)
    ck_tile::index_t d_size, h_size, w_size;    ///< Spatial dimensions of this piece
};

/// @brief Calculate piece information for split-image convolution
///
/// @par Overview
///      Computes spatial coordinates, dimensions, and GPU block range for a single
///      piece in split-image convolution. Handles edge pieces that may have different
///      sizes due to non-uniform division.
///
/// @tparam TilePartitioner Type providing MPerBlock and NPerBlock constants
///
/// @param piece_idx Index of the piece to calculate (0-based)
/// @param num_d_pieces Number of pieces in D dimension
/// @param num_h_pieces Number of pieces in H dimension
/// @param num_w_pieces Number of pieces in W dimension
/// @param base_piece_d Base size of each D piece (may differ for last piece)
/// @param base_piece_h Base size of each H piece (may differ for last piece)
/// @param base_piece_w Base size of each W piece (may differ for last piece)
/// @param total_d Total D dimension size (output space)
/// @param total_h Total H dimension size (output space)
/// @param total_w Total W dimension size (output space)
/// @param N Batch size
/// @param K Output channels
/// @param total_blocks Accumulated block count from previous pieces
///
/// @return SplitImagePieceInfo containing all metadata for this piece
template <typename TilePartitioner>
CK_TILE_HOST SplitImagePieceInfo calculate_spatial_piece(ck_tile::index_t piece_idx,
                                                         ck_tile::index_t num_d_pieces,
                                                         ck_tile::index_t num_h_pieces,
                                                         ck_tile::index_t num_w_pieces,
                                                         ck_tile::index_t base_piece_d,
                                                         ck_tile::index_t base_piece_h,
                                                         ck_tile::index_t base_piece_w,
                                                         ck_tile::index_t total_d,
                                                         ck_tile::index_t total_h,
                                                         ck_tile::index_t total_w,
                                                         ck_tile::index_t N,
                                                         ck_tile::index_t K,
                                                         ck_tile::index_t total_blocks)
{
    // Unflatten piece index into 3D coordinates (W-major, then H, then D)
    const ck_tile::index_t w_idx = piece_idx % num_w_pieces;
    const ck_tile::index_t h_idx = (piece_idx / num_w_pieces) % num_h_pieces;
    const ck_tile::index_t d_idx = piece_idx / (num_w_pieces * num_h_pieces);

    // Calculate spatial start positions
    const ck_tile::index_t w_start = w_idx * base_piece_w;
    const ck_tile::index_t h_start = h_idx * base_piece_h;
    const ck_tile::index_t d_start = d_idx * base_piece_d;

    // Calculate piece sizes (last piece may be larger to cover remainder)
    const ck_tile::index_t w_size =
        (w_idx == num_w_pieces - 1) ? (total_w - w_start) : base_piece_w;
    const ck_tile::index_t h_size =
        (h_idx == num_h_pieces - 1) ? (total_h - h_start) : base_piece_h;
    const ck_tile::index_t d_size =
        (d_idx == num_d_pieces - 1) ? (total_d - d_start) : base_piece_d;

    // Calculate GEMM dimensions for this piece
    const ck_tile::index_t piece_gemm_m = N * d_size * h_size * w_size;
    const ck_tile::index_t piece_gemm_n = K;

    // Calculate GPU grid size for this piece
    const ck_tile::index_t piece_grid =
        ((piece_gemm_m + TilePartitioner::MPerBlock - 1) / TilePartitioner::MPerBlock) *
        ((piece_gemm_n + TilePartitioner::NPerBlock - 1) / TilePartitioner::NPerBlock);

    return {
        total_blocks, total_blocks + piece_grid, d_start, h_start, w_start, d_size, h_size, w_size};
}

} // namespace ck_tile
