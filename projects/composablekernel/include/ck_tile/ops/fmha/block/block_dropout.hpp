// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

// BlockDropoutBwd and BlockDropout (fwd) support two warp gemm tile sizes: 32x32 (MFMA only) and
// 16x16 (MFMA and WMMA). Even if fwd and bwd use different tile sizes, generated random
// numbers will be the same, they are also the same for MFMA (on CDNA), WMMA (on RDNA), or host
// (for verification, see ck_tile/host/reference/reference_batched_dropout_randval.hpp).
//
// The (row, col) coordinate of the current 32x32 tile in the P matrix determines a subsequence of
// random numbers (ph_subsequence).
// The (batch, head, 0..63) coordinate determines an offset in the subsequence (ph_head_offset and
// ph_offset).
// This means that subsequences are non-overlapping, reproducible and independent of mask or window.
//
// There are 3 modes (all produce the same results):
//  * For 32x32 MFMA tile each of 64 lanes generates 4 * 32 bits or 16 bytes, so one warp generates
//  the entire 32x32 tile (64 * 16 = 32 * 32).
//  * For 16x16 MFMA tile one warp generates 1/4 of the 32x32 tile ((16 * 16) / (64 * 16) = 1/4), 4
//  warps generate the same 64 * 16 random bytes and each uses its own quarter. If kMPerBlock >
//  MWarp * WG::kM one warp can generate two 16x16 tiles (MIterPerWarp = 2) so fewer instructions
//  are needed for generating a 32x32 tile.
//  * For 16x16 WMMA tile one warp generates 1/2 of the 32x32 tile ((16 * 16) / (32 * 16) = 1/2), 2
//  warps generate the same 64 * 16 random bytes and each uses its own half. If kMPerBlock > MWarp *
//  WG::kM one warp can generate two 16x16 tiles.

namespace detail {
// The number of Philox 4x32 results required to fill 32x32 tile of 8-bit values
constexpr index_t philox_per_tile = 64;
} // namespace detail

struct NullBlockDropout
{
    template <typename BlockGemm, bool IsFwd = true, typename RandValDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeRandvalDramWindow(RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
                          index_t seqlen_qk_start)
    {
        (void)randval_dram_block_window_tmp;
        (void)seqlen_qk_start;

        return make_null_tile_window(make_tuple(number<0>{}, number<0>{}));
    }
};

struct BlockDropout
{
    CK_TILE_HOST_DEVICE BlockDropout(index_t i_batch,
                                     index_t i_head,
                                     index_t nheads,
                                     unsigned long long seed,
                                     unsigned long long offset,
                                     float rp_undrop_,
                                     uint8_t p_undrop_in_uint8_t_,
                                     bool is_store_randval_)
        : ph_seed(amd_wave_read_first_lane(seed)),
          ph_head_offset(amd_wave_read_first_lane(offset + (i_batch * nheads + i_head) *
                                                               detail::philox_per_tile)),
          rp_undrop(rp_undrop_),
          p_undrop_in_uint8_t(p_undrop_in_uint8_t_),
          is_store_randval(is_store_randval_)
    {
    }

    template <typename BlockGemm, bool IsFwd = true, typename RandValDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeRandvalDramWindow(RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
                          index_t seqlen_qk_start)
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                       = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr bool IsWG32          = WG::kM == 32;
        constexpr index_t MWarp        = config.template at<1>();
        constexpr index_t NWarp        = config.template at<2>();
        using BlockGemmShape           = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock   = BlockGemmShape::kM;
        constexpr index_t MIterPerWarp = (!IsWG32 && kMPerBlock > MWarp * WG::kM) ? 2 : 1;
        constexpr index_t kMPerStep    = MIterPerWarp * MWarp * WG::kM;
        constexpr index_t kNPerStep    = NWarp * WG::kN;

        const auto block_origin  = randval_dram_block_window_tmp.get_window_origin();
        auto randval_dram_window = [&]() {
            if constexpr(IsFwd)
            {
                return make_tile_window(
                    randval_dram_block_window_tmp.get_bottom_tensor_view(),
                    ck_tile::make_tuple(number<kMPerStep>{}, number<kNPerStep>{}),
                    {block_origin.at(number<0>{}), seqlen_qk_start}); // M/N
            }
            else
            {
                return make_tile_window(
                    randval_dram_block_window_tmp.get_bottom_tensor_view(),
                    ck_tile::make_tuple(number<kMPerStep>{}, number<kNPerStep>{}),
                    {seqlen_qk_start, block_origin.at(number<1>{})}); // M/N
            }
        }();

        return randval_dram_window;
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValLdsBlockDescriptor()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                       = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr bool IsWG32          = WG::kM == 32;
        constexpr index_t MWarp        = config.template at<1>();
        constexpr index_t NWarp        = config.template at<2>();
        using BlockGemmShape           = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock   = BlockGemmShape::kM;
        constexpr index_t MIterPerWarp = (!IsWG32 && kMPerBlock > MWarp * WG::kM) ? 2 : 1;
        constexpr index_t kMPerStep    = MIterPerWarp * MWarp * WG::kM;
        constexpr index_t kNPerStep    = NWarp * WG::kN;
        constexpr index_t kN1          = 8;
        constexpr index_t kN0          = kNPerStep / kN1;

        constexpr auto randval_lds_block_desc_0 = make_naive_tensor_descriptor(
            ck_tile::make_tuple(number<kN0>{}, number<kMPerStep>{}, number<kN1>{}),
            ck_tile::make_tuple(number<(kMPerStep + 1) * kN1>{}, number<kN1>{}, number<1>{}),
            number<kN1>{},
            number<1>{});

        constexpr auto randval_lds_block_desc = transform_tensor_descriptor(
            randval_lds_block_desc_0,
            ck_tile::make_tuple(
                make_pass_through_transform(number<kMPerStep>{}),
                make_merge_transform(ck_tile::make_tuple(number<kN0>{}, number<kN1>{}))),
            ck_tile::make_tuple(sequence<1>{}, sequence<0, 2>{}),
            ck_tile::make_tuple(sequence<0>{}, sequence<1>{}));

        return randval_lds_block_desc;
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValTileDistribution()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                       = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr bool IsWG32          = WG::kM == 32;
        constexpr index_t MWarp        = config.template at<1>();
        constexpr index_t NWarp        = config.template at<2>();
        using BlockGemmShape           = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock   = BlockGemmShape::kM;
        constexpr index_t MIterPerWarp = (!IsWG32 && kMPerBlock > MWarp * WG::kM) ? 2 : 1;
        constexpr index_t NIterPerWarp = 1;

        // The tile distribution is different from the one in MakeRandValLdsShuffleTileDistribution,
        // because it can combine 2 (MIterPerWarp) 16x16 subtiles for generating them at once
        constexpr auto randval_block_outer_part_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MWarp, MIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<1, 0>>{};

        // Use Bwd WarpGemm to ensure that Fwd's random values ​​are consistent with Bwd.
        constexpr auto randval_block_inner_part_dstr_encoding =
            typename WarpGemmDispatcher<typename WG::ADataType,
                                        typename WG::BDataType,
                                        typename WG::CDataType,
                                        WG::kM,
                                        WG::kN,
                                        WG::kK,
                                        false,
                                        IsWG32>::CWarpDstrEncoding{};

        constexpr auto randval_block_part_dstr_encode =
            detail::make_embed_tile_distribution_encoding(randval_block_outer_part_dstr_encoding,
                                                          randval_block_inner_part_dstr_encoding);

        return make_static_tile_distribution(randval_block_part_dstr_encode);
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValLdsShuffleTileDistribution()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                       = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr bool IsWG32          = WG::kM == 32;
        constexpr index_t MWarp        = config.template at<1>();
        constexpr index_t NWarp        = config.template at<2>();
        using BlockGemmShape           = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock   = BlockGemmShape::kM;
        constexpr index_t MIterPerWarp = (!IsWG32 && kMPerBlock > MWarp * WG::kM) ? 2 : 1;
        constexpr index_t NIterPerWarp = 1;

        constexpr auto randval_block_outer_part_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto randval_block_part_dstr_encode =
            detail::make_embed_tile_distribution_encoding(randval_block_outer_part_dstr_encoding,
                                                          typename WG::CWarpDstrEncoding{});

        return make_static_tile_distribution(randval_block_part_dstr_encode);
    }

    template <typename BlockGemm,
              typename PComputeDataType,
              typename RandValOutputDataType,
              typename PComputeWindow,
              typename RandValDramWindow>
    CK_TILE_HOST_DEVICE void Run(void* randval_ptr,
                                 const index_t start_n0_idx,
                                 PComputeWindow& p_compute,
                                 RandValDramWindow& randval_dram_window) const
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                       = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr bool IsWG32          = WG::kM == 32;
        constexpr index_t MWarp        = config.template at<1>();
        constexpr index_t NWarp        = config.template at<2>();
        using BlockGemmShape           = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock   = BlockGemmShape::kM;
        constexpr index_t kNPerBlock   = BlockGemmShape::kN;
        constexpr index_t MIterPerWarp = (!IsWG32 && kMPerBlock > MWarp * WG::kM) ? 2 : 1;
        constexpr index_t kMPerStep    = MIterPerWarp * MWarp * WG::kM;
        constexpr index_t kNPerStep    = NWarp * WG::kN;

        // randval tile in LDS
        auto randval_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<uint8_t*>(randval_ptr), MakeRandValLdsBlockDescriptor<BlockGemm>());

        auto randval_lds_window = make_tile_window(
            randval_lds, MakeRandValLdsBlockDescriptor<BlockGemm>().get_lengths(), {0, 0});

        // register distribute
        auto randval_dist_generated =
            make_static_distributed_tensor<uint8_t>(MakeRandValTileDistribution<BlockGemm>());

        const auto randval_lds_read_window =
            make_tile_window(randval_lds_window.get_bottom_tensor_view(),
                             randval_lds_window.get_window_lengths(),
                             randval_lds_window.get_window_origin(),
                             MakeRandValLdsShuffleTileDistribution<BlockGemm>());

        const index_t start_m0_idx = randval_dram_window.get_window_origin().at(number<0>{});
        const index_t iMWarp       = get_warp_id() / NWarp;
        const index_t iNWarp       = get_warp_id() % NWarp;

        auto generate_randval = [&](auto i_m0, auto i_n0) {
            // Generate random numbers
            uint8_t random_uint8_t[randval_dist_generated.kThreadElementSpaceSize];
            const index_t wg_m0 = (start_m0_idx / WG::kM) + (i_m0 * MWarp + iMWarp) * MIterPerWarp;
            const index_t wg_n0 = (start_n0_idx / WG::kN) + (i_n0 * NWarp + iNWarp);
            if constexpr(IsWG32)
            {
                // Generate the whole 32x32 tile at once (each tile consists of random numbers taken
                // from a separate subsequence of Philox)
                const unsigned long long ph_subsequence =
                    bit_cast<unsigned long long>(make_uint2(wg_m0, wg_n0));
                const index_t ph_offset = get_lane_id();
                const ck_tile::philox ph(ph_seed, ph_head_offset + ph_offset);
                static_assert(randval_dist_generated.kThreadElementSpaceSize == 16);
                ph.get_random_16x8(random_uint8_t, ph_subsequence);
            }
            else
            {
                // Generate one or two 16x16 subtiles of the 32x32 tile (depending on whether
                // MIterPerWarp is equal to 1 or 2)
                const unsigned long long ph_subsequence =
                    bit_cast<unsigned long long>(make_uint2(wg_m0 / 2, wg_n0 / 2));
                const index_t subtile_m0 = wg_m0 % 2;
                if constexpr(get_warp_size() == 32)
                {
                    const index_t ph_offset = (get_lane_id() & 15) +
                                              (((get_lane_id() >> 4) & 1) << 5) +
                                              ((wg_n0 % 2) << 4);
                    const ck_tile::philox ph(ph_seed, ph_head_offset + ph_offset);
                    if constexpr(MIterPerWarp == 1)
                    {
                        static_assert(randval_dist_generated.kThreadElementSpaceSize == 8);
                        ph.get_random_8x8(
                            random_uint8_t, ph_subsequence, subtile_m0 * 2 + 0, subtile_m0 * 2 + 1);
                    }
                    else
                    {
                        static_assert(randval_dist_generated.kThreadElementSpaceSize == 16);
                        ph.get_random_16x8(random_uint8_t, ph_subsequence);
                    }
                }
                else
                {
                    const index_t subtile_n0 = (get_lane_id() >> 4) & 1;
                    const index_t ph_offset  = (get_lane_id() & 47) + ((wg_n0 % 2) << 4);
                    const ck_tile::philox ph(ph_seed, ph_head_offset + ph_offset);
                    if constexpr(MIterPerWarp == 1)
                    {
                        static_assert(randval_dist_generated.kThreadElementSpaceSize == 4);
                        ph.get_random_4x8(
                            random_uint8_t, ph_subsequence, subtile_m0 * 2 + subtile_n0);
                    }
                    else
                    {
                        static_assert(randval_dist_generated.kThreadElementSpaceSize == 8);
                        ph.get_random_8x8(
                            random_uint8_t, ph_subsequence, 0 * 2 + subtile_n0, 1 * 2 + subtile_n0);
                    }
                }
            }

            constexpr auto randval_dist_generated_spans =
                decltype(randval_dist_generated)::get_distributed_spans();
            int i_random_idx = 0;
            sweep_tile_span(randval_dist_generated_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(randval_dist_generated_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx          = ck_tile::make_tuple(idx0, idx1);
                    randval_dist_generated(i_j_idx) = random_uint8_t[i_random_idx++];
                });
            });
            // Transpose randval using LDS
            store_tile(randval_lds_window, randval_dist_generated);
            block_sync_lds();
            const auto randval = load_tile(randval_lds_read_window);
            block_sync_lds();
            return randval;
        };

        static_for<0, kMPerBlock / kMPerStep, 1>{}([&](auto i_m0) {
            static_for<0, kNPerBlock / kNPerStep, 1>{}([&](auto i_n0) {
                const auto randval = generate_randval(i_m0, i_n0);
                if(is_store_randval)
                {
                    const auto randval_store = cast_tile<RandValOutputDataType>(randval);
                    store_tile(randval_dram_window, randval_store);
                }
                move_tile_window(randval_dram_window, {0, kNPerStep});
                // Drop values of P based on the generated probabilities
                constexpr auto randval_spans = decltype(randval)::get_distributed_spans();
                sweep_tile_span(randval_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(randval_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto p_idx0 =
                            tile_distributed_index<i_m0 * MIterPerWarp +
                                                   idx0.impl_.template at<0>()>{};
                        constexpr auto p_idx1 =
                            tile_distributed_index<i_n0,
                                                   idx1.impl_.template at<1>(),
                                                   idx1.impl_.template at<2>()>{};
                        constexpr auto p_idx = ck_tile::make_tuple(p_idx0, p_idx1);
                        constexpr auto r_idx = ck_tile::make_tuple(idx0, idx1);
                        p_compute(p_idx)     = randval[r_idx] <= p_undrop_in_uint8_t
                                                   ? p_compute[p_idx] * rp_undrop
                                                   : PComputeDataType(0);
                    });
                });
            });
            move_tile_window(randval_dram_window, {kMPerStep, -kNPerBlock});
        });
        move_tile_window(randval_dram_window, {-kMPerBlock, kNPerBlock});
    }

    const unsigned long long ph_seed;
    const unsigned long long ph_head_offset;
    const float rp_undrop;
    const uint8_t p_undrop_in_uint8_t;
    const bool is_store_randval;
};

// TODO: IsWG32_ is not needed as template parameter and can be removed. IsDropout_ == false can be
// replaced with NullBlockDropout. This requires changes in xformers and other libs.
template <bool IsDropout_, bool IsWG32_, bool IsStoreRandval_>
struct BlockDropoutBwd;

template <bool IsWG32_, bool IsStoreRandval_>
struct BlockDropoutBwd<false, IsWG32_, IsStoreRandval_>
{
    static constexpr bool IsDropout      = false;
    static constexpr bool IsStoreRandval = IsStoreRandval_;

    template <typename BlockGemm, bool IsFwd = false, typename RandValDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeRandvalDramWindow(RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
                          index_t seqlen_qk_start)
    {
        (void)randval_dram_block_window_tmp;
        (void)seqlen_qk_start;

        return make_null_tile_window(make_tuple(number<0>{}, number<0>{}));
    }
};

template <bool IsWG32_, bool IsStoreRandval_>
struct BlockDropoutBwd<true, IsWG32_, IsStoreRandval_>
{
    static constexpr bool IsDropout      = true;
    static constexpr bool IsStoreRandval = IsStoreRandval_;

    CK_TILE_HOST_DEVICE BlockDropoutBwd(index_t i_batch,
                                        index_t i_head,
                                        index_t nheads,
                                        unsigned long long seed,
                                        unsigned long long offset,
                                        float rp_undrop_,
                                        uint8_t p_undrop_in_uint8_t_)
        : ph_seed(amd_wave_read_first_lane(seed)),
          ph_head_offset(amd_wave_read_first_lane(offset + (i_batch * nheads + i_head) *
                                                               detail::philox_per_tile)),
          rp_undrop(rp_undrop_),
          p_undrop_in_uint8_t(p_undrop_in_uint8_t_)
    {
    }

    template <typename BlockGemm, bool IsFwd = false, typename RandValDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeRandvalDramWindow(RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
                          index_t seqlen_qk_start)
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                       = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr bool IsWG32          = WG::kM == 32;
        constexpr index_t MWarp        = config.template at<1>();
        constexpr index_t NWarp        = config.template at<2>();
        using BlockGemmShape           = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock   = BlockGemmShape::kM;
        constexpr index_t MIterPerWarp = (!IsWG32 && kMPerBlock > MWarp * WG::kM) ? 2 : 1;
        constexpr index_t kMPerStep    = MIterPerWarp * MWarp * WG::kM;
        constexpr index_t kNPerStep    = NWarp * WG::kN;

        const auto block_origin  = randval_dram_block_window_tmp.get_window_origin();
        auto randval_dram_window = [&]() {
            if constexpr(IsFwd)
            {
                return make_tile_window(
                    randval_dram_block_window_tmp.get_bottom_tensor_view(),
                    ck_tile::make_tuple(number<kMPerStep>{}, number<kNPerStep>{}),
                    {block_origin.at(number<0>{}), seqlen_qk_start}); // M/N
            }
            else
            {
                return make_tile_window(
                    randval_dram_block_window_tmp.get_bottom_tensor_view(),
                    ck_tile::make_tuple(number<kMPerStep>{}, number<kNPerStep>{}),
                    {seqlen_qk_start, block_origin.at(number<1>{})}); // M/N
            }
        }();

        return randval_dram_window;
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValTileDistribution()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                       = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr bool IsWG32          = WG::kM == 32;
        constexpr index_t MWarp        = config.template at<1>();
        constexpr index_t NWarp        = config.template at<2>();
        using BlockGemmShape           = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock   = BlockGemmShape::kM;
        constexpr index_t MIterPerWarp = (!IsWG32 && kMPerBlock > MWarp * WG::kM) ? 2 : 1;
        constexpr index_t NIterPerWarp = 1;

        constexpr auto randval_block_outer_part_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MWarp, MIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<1, 0>>{};

        constexpr auto randval_block_inner_part_dstr_encoding =
            typename WarpGemmDispatcher<typename WG::ADataType,
                                        typename WG::BDataType,
                                        typename WG::CDataType,
                                        WG::kM,
                                        WG::kN,
                                        WG::kK,
                                        false,
                                        IsWG32>::CWarpDstrEncoding{};
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(randval_block_inner_part_dstr_encoding)>,
                           typename WG::CWarpDstrEncoding>);

        constexpr auto randval_block_part_dstr_encode =
            detail::make_embed_tile_distribution_encoding(randval_block_outer_part_dstr_encoding,
                                                          randval_block_inner_part_dstr_encoding);

        return make_static_tile_distribution(randval_block_part_dstr_encode);
    }

    template <typename BlockGemm,
              typename RandValOutputDataType,
              typename PComputeWindow,
              typename RandValDramWindow>
    CK_TILE_HOST_DEVICE void Run(const index_t start_m0_idx,
                                 const index_t start_n0_idx,
                                 PComputeWindow& p_compute,
                                 RandValDramWindow& randval_dram_window) const
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                       = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr bool IsWG32          = WG::kM == 32;
        constexpr index_t MWarp        = config.template at<1>();
        constexpr index_t NWarp        = config.template at<2>();
        using BlockGemmShape           = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock   = BlockGemmShape::kM;
        constexpr index_t kNPerBlock   = BlockGemmShape::kN;
        constexpr index_t MIterPerWarp = (!IsWG32 && kMPerBlock > MWarp * WG::kM) ? 2 : 1;
        constexpr index_t kMPerStep    = MIterPerWarp * MWarp * WG::kM;
        constexpr index_t kNPerStep    = NWarp * WG::kN;

        // register distribute
        auto randval_dist_generated =
            make_static_distributed_tensor<uint8_t>(MakeRandValTileDistribution<BlockGemm>());

        const index_t iMWarp = get_warp_id() / NWarp;
        const index_t iNWarp = get_warp_id() % NWarp;

        auto generate_randval = [&](auto i_m0, auto i_n0) {
            // Generate random numbers
            uint8_t random_uint8_t[randval_dist_generated.kThreadElementSpaceSize];
            const index_t wg_m0 = (start_m0_idx / WG::kM) + (i_m0 * MWarp + iMWarp) * MIterPerWarp;
            const index_t wg_n0 = (start_n0_idx / WG::kN) + (i_n0 * NWarp + iNWarp);
            if constexpr(IsWG32)
            {
                // Generate the whole 32x32 tile at once (each tile consists of random numbers
                // taken from a separate subsequence of Philox)
                const unsigned long long ph_subsequence =
                    bit_cast<unsigned long long>(make_uint2(wg_m0, wg_n0));
                const index_t ph_offset = get_lane_id();
                const ck_tile::philox ph(ph_seed, ph_head_offset + ph_offset);
                static_assert(randval_dist_generated.kThreadElementSpaceSize == 16);
                ph.get_random_16x8(random_uint8_t, ph_subsequence);
            }
            else
            {
                // Generate one or two 16x16 subtiles of the 32x32 tile (depending on whether
                // MIterPerWarp is equal to 1 or 2)
                const unsigned long long ph_subsequence =
                    bit_cast<unsigned long long>(make_uint2(wg_m0 / 2, wg_n0 / 2));
                const index_t subtile_m0 = wg_m0 % 2;
                if constexpr(get_warp_size() == 32)
                {
                    const index_t ph_offset = (get_lane_id() & 15) +
                                              (((get_lane_id() >> 4) & 1) << 5) +
                                              ((wg_n0 % 2) << 4);
                    const ck_tile::philox ph(ph_seed, ph_head_offset + ph_offset);
                    if constexpr(MIterPerWarp == 1)
                    {
                        static_assert(randval_dist_generated.kThreadElementSpaceSize == 8);
                        ph.get_random_8x8(
                            random_uint8_t, ph_subsequence, subtile_m0 * 2 + 0, subtile_m0 * 2 + 1);
                    }
                    else
                    {
                        static_assert(randval_dist_generated.kThreadElementSpaceSize == 16);
                        ph.get_random_16x8(random_uint8_t, ph_subsequence);
                    }
                }
                else
                {
                    const index_t subtile_n0 = (get_lane_id() >> 4) & 1;
                    const index_t ph_offset  = (get_lane_id() & 47) + ((wg_n0 % 2) << 4);
                    const ck_tile::philox ph(ph_seed, ph_head_offset + ph_offset);
                    if constexpr(MIterPerWarp == 1)
                    {
                        static_assert(randval_dist_generated.kThreadElementSpaceSize == 4);
                        ph.get_random_4x8(
                            random_uint8_t, ph_subsequence, subtile_m0 * 2 + subtile_n0);
                    }
                    else
                    {
                        static_assert(randval_dist_generated.kThreadElementSpaceSize == 8);
                        ph.get_random_8x8(
                            random_uint8_t, ph_subsequence, 0 * 2 + subtile_n0, 1 * 2 + subtile_n0);
                    }
                }
            }

            constexpr auto randval_dist_generated_spans =
                decltype(randval_dist_generated)::get_distributed_spans();
            int i_random_idx = 0;
            sweep_tile_span(randval_dist_generated_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(randval_dist_generated_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx          = ck_tile::make_tuple(idx0, idx1);
                    randval_dist_generated(i_j_idx) = random_uint8_t[i_random_idx++];
                });
            });
            return randval_dist_generated;
        };

        static_for<0, kNPerBlock / kNPerStep, 1>{}([&](auto i_n0) {
            static_for<0, kMPerBlock / kMPerStep, 1>{}([&](auto i_m0) {
                const auto randval = generate_randval(i_m0, i_n0);
                // Drop values of P based on the generated probabilities, negative sign is used to
                // distinguish such values ​​later in bwd pipeline.
                constexpr auto randval_spans = decltype(randval)::get_distributed_spans();
                sweep_tile_span(randval_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(randval_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto r_idx = ck_tile::make_tuple(idx0, idx1);
                        constexpr auto p_idx0 =
                            tile_distributed_index<i_m0 * MIterPerWarp +
                                                       idx0.impl_.template at<0>(),
                                                   idx0.impl_.template at<1>(),
                                                   idx0.impl_.template at<2>()>{};
                        constexpr auto p_idx1 = tile_distributed_index<i_n0>{};
                        constexpr auto p_idx  = ck_tile::make_tuple(p_idx0, p_idx1);
                        p_compute(p_idx)      = randval[r_idx] <= p_undrop_in_uint8_t
                                                    ? p_compute[p_idx]
                                                    : -p_compute[p_idx];
                    });
                });
                // save to Global
                if constexpr(IsStoreRandval)
                {
                    const auto randval_store = cast_tile<RandValOutputDataType>(randval);
                    store_tile(randval_dram_window, randval_store);
                    move_tile_window(randval_dram_window, {kMPerStep, 0});
                }
            });
            if constexpr(IsStoreRandval)
            {
                move_tile_window(randval_dram_window, {-kMPerBlock, kNPerStep});
            }
        });
        if constexpr(IsStoreRandval)
        {
            move_tile_window(randval_dram_window, {kMPerBlock, -kNPerBlock});
        }
    }

    const unsigned long long ph_seed;
    const unsigned long long ph_head_offset;
    const float rp_undrop;
    const uint8_t p_undrop_in_uint8_t;
};

} // namespace ck_tile
