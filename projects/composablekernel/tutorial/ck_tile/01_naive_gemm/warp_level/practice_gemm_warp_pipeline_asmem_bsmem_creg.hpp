// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = PracticeGemmWarpPolicy>
struct PracticeGemmWarpPipelineASmemBSmemCreg
{

    using ADataType     = remove_cvref_t<typename Problem::ADataType>;
    using BDataType     = remove_cvref_t<typename Problem::BDataType>;
    using CDataType     = remove_cvref_t<typename Problem::CDataType>;
    using WaveGemmShape = remove_cvref_t<typename Problem::Shape>;

    using WarpGemm = remove_cvref_t<
        decltype(Policy::template GetWarpGemmMWarpNWarp<Problem>().template get<0>())>;
    static constexpr index_t MWarp =
        Policy::template GetWarpGemmMWarpNWarp<Problem>().template get<1>();
    static constexpr index_t NWarp =
        Policy::template GetWarpGemmMWarpNWarp<Problem>().template get<2>();

    using AWarpDstr = typename WarpGemm::AWarpDstr;
    using BWarpDstr = typename WarpGemm::BWarpDstr;
    using CWarpDstr = typename WarpGemm::CWarpDstr;

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;

    static constexpr auto a_warp_y_lengths =
        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto b_warp_y_lengths =
        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    static constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    static constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    static constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    // C += A * B
    template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   [[maybe_unused]] const ABlockWindowTmp& a_block_window_tmp,
                                   [[maybe_unused]] const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(std::is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          std::is_same_v<BDataType, typename BBlockWindowTmp::DataType> &&
                          std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<1>{}];

        static_assert(MPerBlock == WaveGemmShape::BlockTile_M &&
                          NPerBlock == WaveGemmShape::BlockTile_N &&
                          KPerBlock == WaveGemmShape::BlockTile_K,
                      "wrong!");

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

#if !defined(ENABLE_PREFETCH)
        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iMWarp = get_warp_id() / NWarp;
        const index_t iNWarp = get_warp_id() % NWarp;

        // Construct A-warp-window
        auto a_warp_window_tmp = make_tile_window(
            a_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WarpGemm::kM>{}, number<WarpGemm::kK>{}),
            {a_block_window_tmp.get_window_origin().at(number<0>{}) + iMWarp * WarpGemm::kM,
             a_block_window_tmp.get_window_origin().at(number<1>{})},
            make_static_tile_distribution(typename WarpGemm::AWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;
                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // Construct B-warp-window
        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WarpGemm::kN>{}, number<WarpGemm::kK>{}),
            {b_block_window_tmp.get_window_origin().at(number<0>{}) + iNWarp * WarpGemm::kN,
             b_block_window_tmp.get_window_origin().at(number<1>{})},
            make_static_tile_distribution(typename WarpGemm::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // Read A warp tensor from A block tensor
                AWarpTensor a_warp_tensor;

                a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // Read B warp tensor from B block tensor
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                    // Read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    // Warp GEMM
                    WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // Write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    // C = A * B
    template <typename ABlockWindowTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE auto operator()([[maybe_unused]] const ABlockWindowTmp& a_block_window_tmp,
                                   [[maybe_unused]] const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(std::is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          std::is_same_v<BDataType, typename BBlockWindowTmp::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<1>{}];

        static_assert(MPerBlock == WaveGemmShape::BlockTile_M &&
                          NPerBlock == WaveGemmShape::BlockTile_N &&
                          KPerBlock == WaveGemmShape::BlockTile_K,
                      "wrong!");

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);

        static_assert(std::is_same_v<CDataType, typename WarpGemm::CDataType>, "wrong!");

        // Construct C-Block-Tensor
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);

        return c_block_tensor;
    }
};

} // namespace ck_tile
