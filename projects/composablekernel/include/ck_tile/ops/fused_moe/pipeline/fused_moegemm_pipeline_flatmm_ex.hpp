// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_policy.hpp"

namespace ck_tile {

/*
This pipeline deal with a gemm(actually 2 gemm) with one very small(token), one very big(weight)
we need to design the pipeline such that all waves along gemm-N dim (gemm-m only 1 wave)

    <----- gemm-N ------>
    +----+----+----+----+
    | w0 | w1 | w2 | w3 | gemm-m
    +----+----+----+----+
*/
template <typename Problem_, typename Policy_ = FusedMoeGemmPipelineFlatmmPolicy>
struct FusedMoeGemmPipeline_FlatmmEx
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using BlockShape = typename Problem::BlockShape; // this is FusedMoeGemmShape

    using ADataType            = typename Problem::ADataType;
    using GDataType            = typename Problem::GDataType;
    using DDataType            = typename Problem::DDataType;
    using AccDataType          = typename Problem::AccDataType;
    using ODataType            = typename Problem::ODataType;
    using AScaleDataType       = typename Problem::AScaleDataType;
    using GScaleDataType       = typename Problem::GScaleDataType;
    using DScaleDataType       = typename Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Problem::TopkWeightDataType;
    using IndexDataType        = typename Problem::IndexDataType;
    using YDataType            = typename Problem::YDataType;

    using Traits = typename Problem::Traits;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>();
    static constexpr index_t kAlignmentG = Policy::template GetAlignment_G<Problem>();
    static constexpr index_t kAlignmentD = Policy::template GetAlignment_D<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignment_O<Problem>();

    static constexpr index_t SLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::SLD_A);
    static constexpr index_t GLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_A);
    static constexpr index_t GLD_B = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_B);
    static constexpr index_t GST_O = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GST_O);

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            return 2;
        }
    }();

    static constexpr const char* name = "fused_moe_flatmm";

    // TODO: there are multiple buffers
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize_A()
    {
        return Policy::template GetSmemSize_A<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetACoord()
    {
        constexpr auto a_dist = Policy::template MakeGlobalTileDistribution_A<Problem>();
        const auto a_coord    = a_dist.calculate_index();
        return a_coord;
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetOCoord()
    {
        constexpr auto o_dist = Policy::template MakeOGlobalTileDistribution<Problem>();
        const auto o_coord    = o_dist.calculate_index();
        return o_coord;
    }

    template <typename AWindow, typename GWindow, typename DWindow, typename OWindow>
    CK_TILE_DEVICE auto operator()(const AWindow& a_window_,
                                   const GWindow& g_window_,
                                   const DWindow& d_window_,
                                   OWindow& o_window_,
                                   TopkWeightDataType /*topk_weight*/,
                                   CK_TILE_LDS_ADDR void* smem,
                                   index_t hidden_size,
                                   index_t intermediate_size)
    {
        _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wc++20-extensions\"");
        constexpr auto NEG1  = number<-1>{};
        constexpr auto I0    = number<0>{};
        constexpr auto I1    = number<1>{};
        constexpr auto TRUE  = bool_constant<true>{};
        constexpr auto FALSE = bool_constant<false>{};

        CK_TILE_LDS_ADDR ADataType* smem_0 = reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem);
        CK_TILE_LDS_ADDR ADataType* smem_1 = reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(
            reinterpret_cast<CK_TILE_LDS_ADDR char*>(smem) +
            Policy::template GetSmemSize_A<Problem>());

        auto g_view = g_window_.get_bottom_tensor_view();

        auto u_view = [&]() {
            if constexpr(IsGateOnly)
            {
                return g_view;
            }
            else
            {
                index_t nr_0 = intermediate_size / BlockShape::Block_Nr0;
                index_t kr_0 = hidden_size / BlockShape::Block_Kr0;

                const GDataType* g_ptr =
                    g_window_.get_bottom_tensor_view().get_buffer_view().p_data_;
                const GDataType* u_ptr = g_ptr + (nr_0 / 2) * kr_0 * number<BlockShape::Block_W0>{};

                const auto u_view_ = make_naive_tensor_view<address_space_enum::global>(
                    u_ptr,
                    make_tuple(nr_0, kr_0, number<BlockShape::Block_W0>{}),
                    make_tuple(kr_0 * BlockShape::Block_W0, number<BlockShape::Block_W0>{}, 1),
                    number<kAlignmentG>{},
                    number<1>{});
                const auto u_view_1_ =
                    pad_tensor_view(u_view_,
                                    make_tuple(number<BlockShape::Block_Nr0>{},
                                               number<BlockShape::Block_Kr0>{},
                                               number<BlockShape::Block_W0>{}),
                                    sequence<PadIntermediateSize, PadHiddenSize, 0>{});
                return u_view_1_;
            }
        }();

        auto a_win = make_tile_window_linear(
            a_window_, Policy::template MakeGlobalTileDistribution_A<Problem>());
        auto g_win =
            make_tile_window_linear(g_window_,
                                    Policy::template MakeGlobalTileDistribution_G<Problem>(),
                                    sequence<0, 1, 1>{});
        auto d_win =
            make_tile_window_linear(d_window_,
                                    Policy::template MakeGlobalTileDistribution_D<Problem>(),
                                    sequence<0, 1, 1>{});
        auto o_win = make_tile_window_linear(
            o_window_, Policy::template MakeGlobalTileDistribution_O<Problem>());

        using g_thread_type = decltype(load_tile(g_win));
        using d_thread_type = decltype(load_tile(d_win));

        using WarpGemm0  = decltype(Policy::template GetWarpGemm0<Problem>());
        using WarpGemm1  = decltype(Policy::template GetWarpGemm1<Problem>());
        auto warp_gemm_0 = WarpGemm0{};
        auto warp_gemm_1 = WarpGemm1{};

        // issues_warps_lanes
        auto a_sst_win0 =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 smem_0, Policy::template MakeLdsStoreDesc_A<Problem>()),
                             Policy::template MakeLdsStoreDesc_A<Problem>().get_lengths(),
                             {0, 0, 0});

        auto a_sst_win1 =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 smem_1, Policy::template MakeLdsStoreDesc_A<Problem>()),
                             Policy::template MakeLdsStoreDesc_A<Problem>().get_lengths(),
                             {0, 0, 0});
        // m*k
        auto a_sld_win0 = [&]() {
            using WG                        = WarpGemm0;
            constexpr auto a_outer_dstr_enc = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<BlockShape::Repeat_M0, BlockShape::WarpPerBlock_M0>,
                      sequence<BlockShape::Repeat_K0>>,
                tuple<sequence<1>>,
                tuple<sequence<1>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                a_outer_dstr_enc, typename WG::AWarpDstrEncoding{});
            return make_tile_window_linear(
                make_tensor_view<address_space_enum::lds>(
                    smem_0, Policy::template MakeLdsLoadDesc_A<Problem>()),
                Policy::template MakeLdsLoadDesc_A<Problem>().get_lengths(),
                {0, 0},
                make_static_tile_distribution(a_block_dstr_encode));
        }();

        // m*k
        auto a_sld_win1 = [&]() {
            using WG                        = WarpGemm0;
            constexpr auto a_outer_dstr_enc = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<BlockShape::Repeat_M0, BlockShape::WarpPerBlock_M0>,
                      sequence<BlockShape::Repeat_K0>>,
                tuple<sequence<1>>,
                tuple<sequence<1>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                a_outer_dstr_enc, typename WG::AWarpDstrEncoding{});
            return make_tile_window_linear(
                make_tensor_view<address_space_enum::lds>(
                    smem_1, Policy::template MakeLdsLoadDesc_A<Problem>()),
                Policy::template MakeLdsLoadDesc_A<Problem>().get_lengths(),
                {0, 0},
                make_static_tile_distribution(a_block_dstr_encode));
        }();

        auto bridge_sst_win = [&]() {
            return make_tile_window(
                make_tensor_view<address_space_enum::lds>(
                    reinterpret_cast<YDataType*>(smem),
                    Policy::template MakeBridgeLdsStoreDesc<Problem>()),
                Policy::template MakeBridgeLdsStoreDesc<Problem>().get_lengths(),
                {0, 0});
        }();

        auto bridge_sld_win = [&]() {
            return make_tile_window_linear(
                make_tensor_view<address_space_enum::lds>(
                    reinterpret_cast<YDataType*>(smem),
                    Policy::template MakeBridgeLdsLoadDesc<Problem>()),
                Policy::template MakeBridgeLdsLoadDesc<Problem>().get_lengths(),
                {0, 0},
                Policy::template MakeYTileDistribution<Problem>());
        }();

        // also OK with C array, 2 register buffer
        statically_indexed_array<g_thread_type, 2> gs;

        constexpr auto issues_a = number<a_win.get_num_of_access()>{};
        constexpr auto issues_g = number<g_win.get_num_of_access()>{};
        // constexpr auto issues_d = number<d_win.get_num_of_access()>{};
        // constexpr auto issues_o = number<o_win.get_num_of_access()>{};
        constexpr auto issues_gemm0 =
            number<BlockShape::Repeat_M0 * BlockShape::Repeat_N0 * BlockShape::Repeat_K0 *
                   warp_gemm_0.get_num_of_access()>{};
        constexpr auto issues_gemm1 =
            number<BlockShape::Repeat_M1 * BlockShape::Repeat_N1 * BlockShape::Repeat_K1 *
                   warp_gemm_1.get_num_of_access()>{};
        // constexpr auto issues_sld_a = number<a_sld_win0.get_num_of_access()>{};

        const index_t num_blocks_k0 =
            (hidden_size + BlockShape::Block_K0 - 1) / BlockShape::Block_K0;
        const index_t num_blocks_n1 =
            (hidden_size + BlockShape::Block_N1 - 1) / BlockShape::Block_N1;

        using a_thread_type = decltype(load_tile(a_sld_win0));
        statically_indexed_array<a_thread_type, 2> as;

        auto gld_a = [&]<typename PreNop = bool_constant<false>>(
                         auto& a_store_, auto i_access, PreNop = {}) {
            async_load_tile_raw(a_store_, a_win, i_access, PreNop{});
        };
        auto move_a = [&]() {
            move_tile_window(a_win, {number<0>{}, number<BlockShape::Block_K0>{}});
        };
        auto sld_a = [&](auto& a_, auto& win_, auto i_access) {
            load_tile_raw(a_, win_, i_access);
        };

        auto gld_g =
            [&]<typename PreNop = bool_constant<false>>(auto& g_, auto i_access, PreNop = {}) {
                if constexpr(IsGateOnly)
                {
                    // TODO: hack!
                    if constexpr(i_access.value == 0)
                    {
                        g_win.bottom_tensor_view_ = g_view;
                    }
                    else if constexpr(i_access.value == issues_g / 2)
                    {
                        g_win.bottom_tensor_view_ = u_view;
                    }
                }
                load_tile_raw(g_, g_win, i_access, FALSE, PreNop{});
            };
        auto move_g = [&]() {
            move_tile_window(g_win, {number<0>{}, number<BlockShape::Block_Kr0>{}, number<0>{}});
        };
        statically_indexed_array<d_thread_type, 2> ds;

        auto gld_d =
            [&]<typename PreNop = bool_constant<false>>(auto& d_, auto i_access, PreNop = {}) {
                load_tile_raw(d_, d_win, i_access, FALSE, PreNop{});
            };
        auto move_d = [&]() {
            // d move along gemm-n
            move_tile_window(d_win, {number<BlockShape::Block_N1>{}, number<0>{}});
        };

        auto atomic_add_o =
            [&]<typename PreNop = bool_constant<false>>(auto& o_, auto i_access, PreNop = {}) {
                update_tile_raw(o_win, o_, i_access, TRUE, PreNop{});
            };

        auto acc_0  = Policy::template MakeCBlockTile_Gemm0<Problem>();
        auto acc_1s = generate_tuple(
            [&](auto) { return Policy::template MakeCBlockTile_Gemm1<Problem>(); }, number<2>{});

        // clang-format off
        auto gemm_0 = [&]<typename PostNop = bool_constant<false>>
        (auto& t_c, auto& t_a, auto& t_b, auto i_access, PostNop = {}) {
            using WarpGemm = remove_cvref_t<decltype(warp_gemm_0)>;

            constexpr auto repeat_sub = WarpGemm::get_num_of_access();
            constexpr auto repeat_m = BlockShape::Repeat_M0;
            // constexpr auto repeat_n = BlockShape::Repeat_N0;
            constexpr auto repeat_k = BlockShape::Repeat_K0;
            // loop order n->m->k
            constexpr auto i_sub = i_access % repeat_sub;
            constexpr auto i_k = (i_access / repeat_sub) % repeat_k;
            constexpr auto i_m = (i_access / (repeat_sub * repeat_k )) % repeat_m;
            constexpr auto i_n = (i_access / (repeat_sub * repeat_k )) / repeat_m;

            using AWarpTensor = typename WarpGemm::AWarpTensor;
            using BWarpTensor = typename WarpGemm::BWarpTensor;
            using CWarpTensor = typename WarpGemm::CWarpTensor;
            using AWarpDstr = typename WarpGemm::AWarpDstr;
            using BWarpDstr = typename WarpGemm::BWarpDstr;
            using CWarpDstr = typename WarpGemm::CWarpDstr;

            constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
            constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

            AWarpTensor w_a;
            w_a.get_thread_buffer() = t_a.get_y_sliced_thread_data(
                    merge_sequences(sequence<i_m, i_k>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

            BWarpTensor w_b;
            w_b.get_thread_buffer() = t_b.get_y_sliced_thread_data(
                merge_sequences(sequence<i_n, i_k>{}, b_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

            CWarpTensor w_c;
            w_c.get_thread_buffer() = t_c.get_y_sliced_thread_data(
                        merge_sequences(sequence<i_m, i_n>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            warp_gemm_0(w_c, w_a, w_b, number<i_sub>{}, PostNop{});

            t_c.set_y_sliced_thread_data(
                        merge_sequences(sequence<i_m, i_n>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        w_c.get_thread_buffer());
        };
        // clang-format on

        // clang-format off
        auto gemm_1 = [&]<typename PostNop = bool_constant<false>>
        (auto& t_c, auto& t_a, auto& t_b, auto i_access, PostNop = {}) {
            using WarpGemm = remove_cvref_t<decltype(warp_gemm_1)>;

            constexpr auto repeat_sub = WarpGemm::get_num_of_access();
            constexpr auto repeat_m = BlockShape::Repeat_M0;
            // constexpr auto repeat_n = BlockShape::Repeat_N0;
            constexpr auto repeat_k = BlockShape::Repeat_K0;
            // loop order n->m->k
            constexpr auto i_sub = i_access % repeat_sub;
            constexpr auto i_k = (i_access / repeat_sub) % repeat_k;
            constexpr auto i_m = (i_access / (repeat_sub * repeat_k )) % repeat_m;
            constexpr auto i_n = (i_access / (repeat_sub * repeat_k )) / repeat_m;

            using AWarpTensor = typename WarpGemm::AWarpTensor;
            using BWarpTensor = typename WarpGemm::BWarpTensor;
            using CWarpTensor = typename WarpGemm::CWarpTensor;
            using AWarpDstr = typename WarpGemm::AWarpDstr;
            using BWarpDstr = typename WarpGemm::BWarpDstr;
            using CWarpDstr = typename WarpGemm::CWarpDstr;

            constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
            constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

            AWarpTensor w_a;
            w_a.get_thread_buffer() = t_a.get_y_sliced_thread_data(
                    merge_sequences(sequence<i_m, i_k>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

            BWarpTensor w_b;
            w_b.get_thread_buffer() = t_b.get_y_sliced_thread_data(
                merge_sequences(sequence<i_n, i_k>{}, b_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

            CWarpTensor w_c;
            w_c.get_thread_buffer() = t_c.get_y_sliced_thread_data(
                        merge_sequences(sequence<i_m, i_n>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            warp_gemm_1(w_c, w_a, w_b, number<i_sub>{}, PostNop{});

            t_c.set_y_sliced_thread_data(
                        merge_sequences(sequence<i_m, i_n>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        w_c.get_thread_buffer());
        };
        // clang-format on
        _Pragma("clang diagnostic pop");

        // this gemm pipeline is designed with assumption that issues of buffer-load/ds_read can
        // be hide under mfma. In other words, issues of mfma is >= memory this is true if we
        // pre-shuffle B matrix, and A matrix is relatively small we prefer use multiple mfma
        // paired with 1 buffer-load B matrix, to get max throughput of buffer_load. and by
        // preshuffle, we always pack to dwordx4 load, and this will already extend to multiple
        // mfma but that is already consumed inside warpgemm-impl. So indeed how many extra
        // mfma(that can reuse the B matrix) only affected by M repeat.
        auto pipeline_gemm0 = [&]() {
            constexpr index_t total_loops = issues_gemm0;
            constexpr auto sr             = Policy::template GetSequencer_0<Problem>();
            static_assert(sr.size() == total_loops);

            constexpr auto c_sld_a_0 = MAKE_SC();
            constexpr auto c_gld_a_0 = MAKE_SC();
            constexpr auto c_gld_b_0 = MAKE_SC();
            // compute buffer 1
            static_for<0, total_loops, 1>{}([&](auto i_issue) {
                gemm_0(acc_0, as[I0], gs[I0], i_issue);
                constexpr index_t slot = sr.at(i_issue);

                if constexpr(slot & SLD_A)
                    sld_a(as[I1], a_sld_win1, number<NEXT_SCI(c_sld_a_0, i_issue)>{});
                if constexpr(slot & GLD_A)
                    gld_a(a_sst_win0, number<NEXT_SCI(c_gld_a_0, i_issue)>{});
                if constexpr(slot & GLD_B)
                    gld_g(gs[I0], number<NEXT_SCI(c_gld_b_0, i_issue)>{});
            });
            move_g();
            move_a();
            block_sync_load_raw(issues_a + issues_g);
            lds_load_fence();

            constexpr auto c_sld_a_1 = MAKE_SC();
            constexpr auto c_gld_a_1 = MAKE_SC();
            constexpr auto c_gld_b_1 = MAKE_SC();

            // compute buffer 1
            static_for<0, total_loops, 1>{}([&](auto i_issue) {
                gemm_0(acc_0, as[I1], gs[I1], i_issue);
                constexpr index_t slot = sr.at(i_issue);

                if constexpr(slot & SLD_A)
                    sld_a(as[I0], a_sld_win0, number<NEXT_SCI(c_sld_a_1, i_issue)>{});
                if constexpr(slot & GLD_A)
                    gld_a(a_sst_win1, number<NEXT_SCI(c_gld_a_1, i_issue)>{});
                if constexpr(slot & GLD_B)
                    gld_g(gs[I1], number<NEXT_SCI(c_gld_b_1, i_issue)>{});
            });
            move_g();
            move_a();
            block_sync_load_raw(issues_a + issues_g);
            lds_load_fence();
        };

        auto pipeline_gemm0_tail = [&]() {
            constexpr index_t total_loops = issues_gemm0;
            constexpr auto sr             = Policy::template GetSequencer_0<Problem>();
            static_assert(sr.size() == total_loops);

            constexpr auto c_gld_b_0 = MAKE_SC();

            // compute buffer 0
            static_for<0, total_loops, 1>{}([&](auto i_issue) {
                gemm_0(acc_0, as[I0], gs[I0], i_issue);
                constexpr index_t slot = sr.at(i_issue);

                if constexpr(slot & GLD_B)
                    gld_g(gs[I1], number<NEXT_SCI(c_gld_b_0, i_issue)>{});
            });

            block_sync_load_raw(issues_g);
            sld_a(as[I1], a_sld_win1, NEG1);

            // compute buffer 1
            static_for<0, total_loops, 1>{}([&](auto i_issue) {
                constexpr auto last_nop = [&]() {
                    if constexpr(i_issue == (total_loops - 1))
                        return TRUE;
                    else
                        return FALSE;
                }();
                gemm_0(acc_0, as[I1], gs[I1], i_issue, last_nop); // last gemm has nop
            });
        };

        auto y = Policy::template MakeYBlockTile<Problem>();

        auto pipeline_bridge = [&]() {
            // cast to Y data
            auto y_pre = cast_tile<YDataType>(acc_0);
            store_tile(bridge_sst_win, y_pre);
            clear_tile(acc_1s(I0));
            // wave_barrier();
            load_tile(y, bridge_sld_win);
            clear_tile(acc_1s(I1));
        };

        // note, gemm-1 start from idx-1 to N-2 (0, 1, 2....N-1)
        auto pipeline_gemm1 = [&]() {
            constexpr index_t total_loops = issues_gemm1;
            constexpr auto sr             = Policy::template GetSequencer_1<Problem>();
            static_assert(sr.size() == total_loops);

            constexpr auto c_gld_b_0 = MAKE_SC();
            constexpr auto c_gst_o_0 = MAKE_SC();
            constexpr auto c_gld_b_1 = MAKE_SC();
            constexpr auto c_gst_o_1 = MAKE_SC();

            // compute buffer 0
            static_for<0, total_loops, 1>{}([&](auto i_issue) {
                gemm_1(acc_1s[I1], y, ds[I1], i_issue);
                constexpr index_t slot = sr.at(i_issue);
                if constexpr(slot & GLD_B)
                    gld_d(ds[I0], number<NEXT_SCI(c_gld_b_0, i_issue)>{});

                if constexpr(slot & GST_O)
                {
                    auto out = cast_tile<ODataType>(acc_1s[I0]);
                    atomic_add_o(out, number<NEXT_SCI(c_gst_o_0, i_issue)>{});
                }
            });
            move_d();
            // move_o();

            // compute buffer 1
            static_for<0, total_loops, 1>{}([&](auto i_issue) {
                gemm_1(acc_1s[I0], y, ds[I0], i_issue);
                constexpr index_t slot = sr.at(i_issue);
                if constexpr(slot & GLD_B)
                    gld_d(ds[I1], number<NEXT_SCI(c_gld_b_1, i_issue)>{});

                if constexpr(slot & GST_O)
                {
                    auto out = cast_tile<ODataType>(acc_1s[I1]);
                    atomic_add_o(out, number<NEXT_SCI(c_gst_o_1, i_issue)>{});
                }
            });
            move_d();
        };

        auto pipeline_gemm1_head = [&]() {
            constexpr index_t total_loops = issues_gemm1;
            constexpr auto sr             = Policy::template GetSequencer_1<Problem>();
            static_assert(sr.size() == total_loops);

            constexpr auto c_gld_b_0 = MAKE_SC();

            // compute buffer 0
            static_for<0, total_loops, 1>{}([&](auto i_issue) {
                gemm_1(acc_1s[I0], y, ds[I0], i_issue);
                constexpr index_t slot = sr.at(i_issue);
                if constexpr(slot & GLD_B)
                    gld_d(ds[I1], number<NEXT_SCI(c_gld_b_0, i_issue)>{});
            });
            move_d();
        };
        auto pipeline_gemm1_tail = [&]() {
            constexpr index_t total_loops = issues_gemm1;
            constexpr auto sr             = Policy::template GetSequencer_1<Problem>();
            static_assert(sr.size() == total_loops);

            constexpr auto c_gst_o_0 = MAKE_SC();

            // compute buffer 1
            static_for<0, total_loops, 1>{}([&](auto i_issue) {
                gemm_1(acc_1s[I1], y, ds[I1], i_issue);

                constexpr index_t slot = sr.at(i_issue);
                if constexpr(slot & GST_O)
                {
                    auto out = cast_tile<ODataType>(acc_1s[I0]);
                    atomic_add_o(out, number<NEXT_SCI(c_gst_o_0, i_issue)>{});
                }
            });
            {
                auto out = cast_tile<ODataType>(acc_1s[I1]);
                atomic_add_o(out, NEG1);
            }
        };

        // start of pipeline
        // clang-format off
        gld_a(a_sst_win0, NEG1, TRUE);
        gld_g(gs[I0], NEG1, TRUE);
        move_a();
        move_g();
        clear_tile(acc_0);

        // preload for next round
        gld_a(a_sst_win1, NEG1); 
        gld_g(gs[I1], NEG1);

        // make sure a,g loaded
        block_sync_load_raw(issues_a + issues_g);
        lds_load_fence();

        // we manually unroll double buffer inside hot loop
        const index_t iters_0 = (num_blocks_k0 - 2) / 2;
        index_t i_0 = 0; // (void)i_0; (void)iters_0; (void)pipeline_gemm0;
        while(i_0++ < iters_0)
        {
            pipeline_gemm0();
        }
        pipeline_gemm0_tail();

        pipeline_bridge();

        const index_t iters_1 = (num_blocks_n1 - 2) / 2;
        index_t i_1 = 0; // (void) i_1; (void)iters_1; (void)pipeline_gemm1;
        pipeline_gemm1_head();
        while(i_1++ < iters_1)
        {
            pipeline_gemm1();
        }
        pipeline_gemm1_tail();
        // clang-format on
    }
};

} // namespace ck_tile
