// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

template <typename Problem, typename DataType, index_t YPerTile, index_t XPerTile>
CK_TILE_HOST_DEVICE static constexpr auto GetABQGlobalVectorLoadSize()
{
    using I1                 = number<1>;
    constexpr index_t NWarps = Problem::BlockGemmShape::BlockWarps::at(I1{});

    constexpr index_t BlockSize = Problem::kBlockSize;

    // Data is replicated across warps along NWarps, so we divide BlockSize by NWarps
    constexpr index_t elements_per_thread = (YPerTile * XPerTile) / (BlockSize / NWarps);
    constexpr index_t PackedSize = ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;

    // Define vector load candidates in descending order of priority
    constexpr std::array<index_t, 5> candidates{
        PackedSize * 32 / sizeof(DataType),
        PackedSize * 16 / sizeof(DataType),
        PackedSize * 8 / sizeof(DataType),
        PackedSize * 4 / sizeof(DataType),
        PackedSize * 2 / sizeof(DataType),
    };

    for(const auto vec_size : candidates)
    {
        if(vec_size <= 0 || XPerTile % vec_size != 0 || elements_per_thread % vec_size != 0)
            continue;
        bool is_valid = (vec_size > 0) && (XPerTile % vec_size == 0) &&
                        (elements_per_thread % vec_size == 0) && vec_size != candidates[4];
        if(is_valid)
        {
            return vec_size;
        }
    }
    return PackedSize; // Absolute fallback
}

// AQ holds groupquant scale data for A. Data is loaded from DRAM and partitioned across
// threads. Post mfma scales are shuffled across threads in the warp and applied to
// accum registers.
template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t KPerBlockAQ,
          index_t VecSize,
          bool PreshuffleQuant>
struct tile_distribution_encoding_pattern_aq : public tile_distribution_encoding_pattern
{
    static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t num_warps = BlockSize / get_warp_size();

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr index_t MIterPerWarp = BlockGemmShape::kM / (MWarps * WarpGemm::kM);

    static_assert(num_warps == MWarps * NWarps * KWarps);

    // KWarps > 1 isn't supported
    static_assert(KWarps == 1);

    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution()
    {
        if constexpr(PreshuffleQuant)
        {
            // # of elements per thread
            static_assert(XPerTile >= warp_size && XPerTile % warp_size == 0);
            constexpr index_t X1 = warp_size;
            constexpr index_t X0 = XPerTile / warp_size;

            constexpr index_t Y1 = MWarps;
            constexpr index_t Y0 = YPerTile / Y1;
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<NWarps>,
                                           tuple<sequence<Y0, Y1>, sequence<X0, X1>>,
                                           tuple<sequence<1, 0>, sequence<2>>,
                                           tuple<sequence<1, 0>, sequence<1>>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{});
        }
        else
        {
            // # of elements per thread
            constexpr index_t X = XPerTile;

            constexpr index_t YR = 1;
            constexpr index_t Y0 = MIterPerWarp ? MIterPerWarp : 1;
            constexpr index_t Y1 = MWarps;
            constexpr index_t Y2 = WarpGemm::kM;
            static_assert(Y2 >= WarpGemm::kM,
                          "Scales for all rows must be available within the warp.");
            static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover the blocktile along Y.");
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<NWarps, YR>,
                                           tuple<sequence<Y0, Y1, Y2>, sequence<X>>,
                                           tuple<sequence<1, 0>, sequence<0, 1>>,
                                           tuple<sequence<1, 0>, sequence<1, 2>>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{});
        }
    }
    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution_transposed()
    {

        constexpr index_t Y0 = YPerTile;
        constexpr index_t X0 = 1;
        constexpr index_t X1 = MIterPerWarp ? MIterPerWarp : 1;
        constexpr index_t X2 = MWarps;
        constexpr index_t X3 = WarpGemm::kM;

        static_assert(X3 >= WarpGemm::kM, "Scales for all rows must be available within the warp.");
        static_assert(X0 * X1 * X2 * X3 == XPerTile,
                      "X0, X1, X2, X3 must cover the blocktile along X.");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>,
                                       tuple<sequence<Y0>, sequence<X0, X1, X2, X3>>,
                                       tuple<sequence<2, 0>, sequence<2, 2>>,
                                       tuple<sequence<2, 0>, sequence<0, 3>>,
                                       sequence<2, 1>,
                                       sequence<1, 0>>{});
    }
};

template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize>
struct tile_distribution_encoding_pattern_aq_transposed_c
    : public tile_distribution_encoding_pattern
{
    // TODO: make pattern where below condition does not need to hold - GGemmMultiDSplitk!
    static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t num_warps = BlockSize / get_warp_size();

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr index_t MIterPerWarp = BlockGemmShape::kM / (MWarps * WarpGemm::kM);

    static_assert(num_warps == MWarps * NWarps * KWarps);

    // KWarps > 1 isn't supported
    static_assert(KWarps == 1);

    // # of elements per thread
    static constexpr index_t X  = XPerTile;
    static constexpr index_t XR = 2;

    // Number of iters per warp
    // MIters are indexed using (Y0, Y1)
    static constexpr index_t Y0 = MIterPerWarp;

    // # of warps in Y dim
    static constexpr index_t Y1 = MWarps;

    static constexpr index_t Y2 = WarpGemm::kM;

    static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover the blocktile along Y.");

    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps, XR>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X>>,
                                       tuple<sequence<1, 0>, sequence<0, 1>>,
                                       tuple<sequence<1, 0>, sequence<1, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{});
    }
};

// TODO:: might need to update
template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t KPerTile,
          index_t NPerTile,
          index_t NPerQ,
          index_t KPerQ,
          typename BQLayout    = tensor_layout::gemm::ColumnMajor,
          bool PreshuffleQuant = false>
struct tile_distribution_encoding_pattern_bq : public tile_distribution_encoding_pattern
{
    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t num_warps = BlockSize / get_warp_size();

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr index_t NIterPerWarp = BlockGemmShape::kN / (NWarps * WarpGemm::kN);

    static_assert(num_warps == MWarps * NWarps * KWarps);
    static_assert(KWarps == 1);

    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution()
    {
        // Preshuffle only supported for ColumnMajor currently
        static_assert(!(PreshuffleQuant && std::is_same_v<BQLayout, tensor_layout::gemm::RowMajor>),
                      "PreshuffleQuant only supported for ColumnMajor BQLayout");

        if constexpr(PreshuffleQuant)
        {
            // =============================================================================
            // PRE-SHUFFLED BQ SCALE TILE DISTRIBUTION
            // =============================================================================
            // For pre-shuffled quantization, the BQ scale tensor has been reorganized
            // (pre-shuffled) to optimize memory access patterns during dequantization.
            //
            // Tile Dimensions:
            //   - K-axis (Y in encoding): Corresponds to the K-dimension iteration
            //   - N-axis (X in encoding): Flattened scale index combining N and K groups
            //
            // The encoding distributes work across threads such that each thread loads
            // the correct pre-shuffled scale for its corresponding B-matrix elements.
            // =============================================================================
            if constexpr(NPerQ <= WarpGemm::kN)
            {
                // =========================================================================
                // CASE 1: Fine-grained Quantization (NPerQ <= WarpGemm::kN)
                // =========================================================================
                // Multiple quantization scales exist within a single warp's N-dimension.
                // Each warp processes multiple scales: WarpGemm::kN / NPerQ scales per warp.
                //
                // Example: NPerQ=8, WarpGemm::kN=16, KPerQ=128, BlockGemmShape::kK=256
                //          → 2 scales per warp in N, 2 K-groups per block
                constexpr auto N1 = BlockGemmShape::kK /
                                    KPerQ; // Number of K-dimension quantization groups per block,
                                           // Each K-group of KPerQ elements shares the same scale.
                constexpr auto N0 =
                    WarpGemm::kN / NPerQ;   // Number of scales per warp in N-dimension, Since NPerQ
                                            // <= WarpGemm::kN, each warp handles multiple scales.
                constexpr auto N2  = 1;     // Elements per thread
                constexpr auto NR1 = NPerQ; // Elements sharing the same scale in N-dimension
                constexpr auto NR0 =
                    warp_size /
                    (N0 * N1 * N2 * NR1);   // Interleave factor to ensure full warp utilization
                constexpr auto K1 = NWarps; // Number of warps distributed along this dimension
                constexpr auto K0 = KPerTile / K1; // Iterations per warp to cover the K-tile
                constexpr auto KR = 1;             // No replication in K-dimension

                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<MWarps, NR0, NR1, KR>,
                                               tuple<sequence<K0, K1>, sequence<N0, N1, N2>>,
                                               tuple<sequence<0, 1>, sequence<0, 2, 0, 2, 0>>,
                                               tuple<sequence<0, 1>, sequence<1, 0, 2, 1, 3>>,
                                               sequence<1, 2>,
                                               sequence<0, 2>>{});
            }
            else if constexpr(NPerQ < WarpGemm::kN * NWarps)
            {
                // =========================================================================
                // CASE 2: Medium-grained Quantization (WarpGemm::kN < NPerQ < WarpGemm::kN *
                // NWarps)
                // =========================================================================
                // Each warp handles exactly one quantization scale in N-dimension.
                // Some warps share the same scale (KR > 1 creates warp grouping).
                //
                // Example: NPerQ=32, WarpGemm::kN=16, NWarps=4
                //          → KR=2 (2 warps share same scale), K1=2 (2 unique scale groups)

                constexpr auto KR  = NPerQ / WarpGemm::kN; // Number of warps sharing the same scale
                constexpr auto K1  = NWarps / KR; // Number of distinct warp groups (unique scales)
                constexpr auto K0  = KPerTile / K1; // Iterations to cover K-tile per warp group
                constexpr auto N1  = BlockGemmShape::kK / KPerQ; // K-dimension quantization groups
                constexpr auto N0  = 1; // Scales per warp in N-dim (1 since NPerQ >= WarpGemm::kN)
                constexpr auto N2  = 1; // Elements per thread
                constexpr auto NR1 = NPerQ; // Scale broadcast factor (full NPerQ)
                constexpr auto NR0 =
                    warp_size / (N0 * N1 * N2 * NR1); // Remaining interleave factor

                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<MWarps, NR0, NR1, KR>,
                                               tuple<sequence<K0, K1>, sequence<N0, N1, N2>>,
                                               tuple<sequence<0, 1, 0>, sequence<0, 2, 0, 2>>,
                                               tuple<sequence<0, 1, 3>, sequence<1, 0, 2, 1>>,
                                               sequence<1, 2>,
                                               sequence<0, 2>>{});
            }
            else
            {
                // =========================================================================
                // CASE 3: Coarse-grained Quantization (NPerQ >= WarpGemm::kN * NWarps)
                // =========================================================================
                // The quantization group spans ALL warps in N-dimension.
                // All warps share the same scale value for their N-tiles.
                //
                // Example: NPerQ=128, WarpGemm::kN=16, NWarps=4
                //          → 128 >= 16*4=64, so all 4 warps use the same scale
                constexpr auto N1  = BlockGemmShape::kK / KPerQ; // K-dimension quantization groups
                constexpr auto N0  = 1;  // Minimal (1) since scale is shared across N
                constexpr auto N2  = 1;  // Elements per thread
                constexpr auto NR1 = 32; // Fixed broadcast size
                constexpr auto NR0 =
                    warp_size / (N0 * N1 * N2 * NR1); // Remaining interleave factor
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<MWarps, NWarps, NR0, NR1>,
                                               tuple<sequence<KPerTile>, sequence<N0, N1, N2>>,
                                               tuple<sequence<0, 0>, sequence<0, 2, 0, 2>>,
                                               tuple<sequence<0, 1>, sequence<2, 0, 3, 1>>,
                                               sequence<1, 2>,
                                               sequence<0, 2>>{});
            }
        }
        else
        {
            /// @brief Creates a 2D tile distribution for BQ (B-matrix quantization scales)
            ///
            /// This function determines the optimal thread distribution pattern for loading and
            /// applying quantization scales to the B matrix based on the quantization group size
            /// (NPerQ) relative to warp dimensions.
            ///
            /// Three distinct distribution patterns are handled:
            ///
            /// 1. Fine-grained quantization (NPerQ < WarpGemm::kN):
            ///    - Multiple quantization groups exist within a single warp's N-dimension
            ///    - Each warp processes multiple scales (WarpGemm::kN / NPerQ scales per warp)
            ///    - Distribution includes explicit replication factor (XR = NPerQ) for scale
            ///    broadcast
            ///    - Example: NPerQ=8, WarpGemm::kN=16, NWarps=4 → 2 scales per warp
            ///
            /// 2. Medium-grained quantization (WarpGemm::kN <= NPerQ <= WarpGemm::kN * NWarps):
            ///    - Each warp handles exactly one quantization scale
            ///    - Scales are distributed across warps with replication factor XR = NPerQ /
            ///    WarpGemm::kN
            ///    - Example: NPerQ=64, WarpGemm::kN=16, NWarps=4 → 1 scale per warp, XR=4
            ///
            /// 3. Coarse-grained quantization (NPerQ > WarpGemm::kN * NWarps):
            ///    - Quantization group spans multiple warps
            ///    - All warps share the same scale value
            ///    - Example: NPerQ=128, WarpGemm::kN=16, NWarps=4 → all warps use same scale
            ///
            /// @return A static tile distribution encoding for the BQ scale tensor
            if constexpr(NPerQ < WarpGemm::kN)
            {
                // Case 1: Fine-grained - multiple quantization scales within a single warp
                // N dimension needs to be partitioned the same way regardless of layout
                constexpr index_t NR = 1;                    // No N replication needed
                constexpr index_t N0 = NIterPerWarp;         // Iterations per warp in N-dim
                constexpr index_t N1 = NWarps;               // Number of warps in N-dim
                constexpr index_t N2 = WarpGemm::kN / NPerQ; // Number of scales per warp

                static_assert(N0 * N1 * N2 == NPerTile,
                              "N0, N1, N2 must cover the blocktile along N dimension.");

                if constexpr(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>)
                {
                    // ColumnMajor: [(N0, N1, N2), K] - N on Y-axis, partition Y
                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<MWarps, NR, NPerQ>,
                                                   tuple<sequence<N0, N1, N2>, sequence<KPerTile>>,
                                                   tuple<sequence<0, 1>, sequence<0, 1, 0>>,
                                                   tuple<sequence<0, 1>, sequence<1, 2, 2>>,
                                                   sequence<1, 2>,
                                                   sequence<0, 0>>{});
                }
                else
                {
                    // RowMajor: [K, (N0, N1, N2)] - N on X-axis, partition X
                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<MWarps, NR, NPerQ>,
                                                   tuple<sequence<KPerTile>, sequence<N0, N1, N2>>,
                                                   tuple<sequence<0, 2>, sequence<0, 2, 0>>,
                                                   tuple<sequence<0, 1>, sequence<1, 2, 2>>,
                                                   sequence<2, 1>,
                                                   sequence<0, 0>>{});
                }
            }
            else if constexpr(NPerQ <= WarpGemm::kN * NWarps)
            {
                // Case 2: Medium-grained - one quantization scale per warp
                constexpr auto NR = NPerQ / WarpGemm::kN; // Scale replication factor
                constexpr auto N1 = NWarps / NR;          // Warps per unique scale
                constexpr auto N0 = NPerTile / N1;        // Iterations to cover N dimension

                if constexpr(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>)
                {
                    // ColumnMajor: [(N0, N1), K] - N on Y-axis
                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<MWarps, NR, get_warp_size()>,
                                                   tuple<sequence<N0, N1>, sequence<KPerTile>>,
                                                   tuple<sequence<0, 1, 0>, sequence<0>>,
                                                   tuple<sequence<0, 1, 1>, sequence<2>>,
                                                   sequence<1, 2>,
                                                   sequence<0, 0>>{});
                }
                else
                {
                    // RowMajor: [K, (N0, N1)] - N on X-axis
                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<MWarps, NR, get_warp_size()>,
                                                   tuple<sequence<KPerTile>, sequence<N0, N1>>,
                                                   tuple<sequence<0, 2, 0>, sequence<0>>,
                                                   tuple<sequence<0, 1, 1>, sequence<2>>,
                                                   sequence<2, 1>,
                                                   sequence<0, 0>>{});
                }
            }
            else // NPerQ > WarpGemm::kN * NWarps
            {
                // Case 3: Coarse-grained - quantization group spans all warps
                // All warps in N-dimension share the same quantization scale
                if constexpr(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>)
                {
                    // ColumnMajor: [N, K]
                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<MWarps, NWarps, get_warp_size()>,
                                                   tuple<sequence<NPerTile>, sequence<KPerTile>>,
                                                   tuple<sequence<0, 0>, sequence<0>>,
                                                   tuple<sequence<0, 1>, sequence<2>>,
                                                   sequence<1, 2>,
                                                   sequence<0, 0>>{});
                }
                else
                {
                    // RowMajor: [K, N]
                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<MWarps, NWarps, get_warp_size()>,
                                                   tuple<sequence<KPerTile>, sequence<NPerTile>>,
                                                   tuple<sequence<0, 0>, sequence<0>>,
                                                   tuple<sequence<0, 1>, sequence<2>>,
                                                   sequence<2, 1>,
                                                   sequence<0, 0>>{});
                }
            }
        }
    }
};

template <typename GroupSizes>
struct QuantGroupShape
{
    static constexpr index_t kM = GroupSizes::at(number<0>{});
    static constexpr index_t kN = GroupSizes::at(number<1>{});
    static constexpr index_t kK = GroupSizes::at(number<2>{});

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_', "quant_group_shape", concat('x', kM, kN, kK));
    }
};

} // namespace ck_tile
