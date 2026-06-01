// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/flatmm/block/flatmm_uk_config.hpp"

namespace ck_tile {

// "S"tream update output along "N"
// A in smem, B load from global
// require 4 wave, occupancy=1c
struct FlatmmSn_32x128x512_1x4x1_16x16x32_Base
{
    static constexpr index_t Block_M = 32;
    static constexpr index_t Block_N = 128;
    static constexpr index_t Block_K = 512;

    static constexpr index_t WarpPerBlock_M = 1;
    static constexpr index_t WarpPerBlock_N = 4;
    static constexpr index_t WarpPerBlock_K = 1;

    static constexpr index_t Warp_M = 16;
    static constexpr index_t Warp_N = 16;
    static constexpr index_t Warp_K = 32;

    static constexpr index_t BlockSize = 256;

    // static constexpr index_t KPack = 2; // this is used to gurantee every threads can do dwordx4

    // TODO: note Nr/Kr/W need consider KPack
    static constexpr index_t Block_W  = Warp_N * Warp_K;  // 512 element
    static constexpr index_t Block_Nr = Block_N / Warp_N; // 32 element, 4 per wave
    static constexpr index_t Block_Kr = Block_K / Warp_K; // 4

    static constexpr index_t Repeat_M = Block_M / (Warp_M * WarpPerBlock_M); // 2
    static constexpr index_t Repeat_N = Block_N / (Warp_N * WarpPerBlock_N); // 2
    static constexpr index_t Repeat_K = Block_K / (Warp_K * WarpPerBlock_K); // 16

    static CK_TILE_DEVICE constexpr auto MakeCBlockDist()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<Repeat_M, WarpPerBlock_M>, sequence<Repeat_N, WarpPerBlock_N>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<2, 1>, // !! note here is different
            sequence<0, 0>>{};

        using WG = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution<>;

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        return c_block_dstr;
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        //                    y     y     p     p      p      y
        // reg before shfl  M0(2)*N0(2)*Nl(4)*Nw(4)*Mw(16)*Nv(4)
        // but order is N0*M0*Nv
        // in LDS we need store as
        //          M0(2)* N0(2) *  Nl(4) * Nw(4) * (Mw(16)*Nv(4) + 4)
        //             y    y       wave-id  lid/16  lid%16   v
        constexpr index_t nbufs = 2;
        return 2 * 2 * 4 * 4 * (16 * 4 + 4) * sizeof(bf16_t) * nbufs;
    }
};

struct FlatmmSn_32x128x512_1x4x1_16x16x32_BF16 : public FlatmmSn_32x128x512_1x4x1_16x16x32_Base
{
    using BDataType = bf16_t;
    using ODataType = bf16_t;

    // TODO: need paired with tile_window_linear!
    // TODO: need call init_raw() before call this function!
    // template <typename AWindow, typename BWindow, typename OWindow, typename ScaleTensor>
    template <typename BRes,
              typename BCoords,
              typename ORes,
              typename OCoords,
              typename OFlags,
              typename ScaleTensor>
    CK_TILE_DEVICE auto
    operator()(const BRes& res_b,
               const BCoords& cached_coords_b,
               const ORes& res_o,
               const OCoords& cached_coords_o,
               const OFlags& o_flags, // this should be in sgpr
               CK_TILE_LDS_ADDR void* smem,
               index_t n, // loop along n dim
               const ScaleTensor& scale_,
               index_t tile_offset_b, // stride b is fixed to blockKr * blockW, but still can adjust
               index_t tile_offset_o)
    {
        static_assert(BCoords::size() == 8); // 8
        static_assert(OCoords::size() == 8);

        const index_t tile_stride_b_bytes = tile_offset_b * sizeof(BDataType);
        const index_t tile_stride_o_bytes = tile_offset_o * sizeof(ODataType);

        static_assert(ScaleTensor::size() == 2);
        float s0 = scale_[number<0>{}];
        float s1 = scale_[number<1>{}];

        index_t loop_cnt = n / Block_N;

        register float v_c0 asm("v64");
        register float v_c1 asm("v65");
        register float v_c2 asm("v66");
        register float v_c3 asm("v67");
        register float v_c4 asm("v68");
        register float v_c5 asm("v69");
        register float v_c6 asm("v70");
        register float v_c7 asm("v71");
        register float v_c8 asm("v72");
        register float v_c9 asm("v73");
        register float v_c10 asm("v74");
        register float v_c11 asm("v75");
        register float v_c12 asm("v76");
        register float v_c13 asm("v77");
        register float v_c14 asm("v78");
        register float v_c15 asm("v79");
        register float v_c16 asm("v80");
        register float v_c17 asm("v81");
        register float v_c18 asm("v82");
        register float v_c19 asm("v83");
        register float v_c20 asm("v84");
        register float v_c21 asm("v85");
        register float v_c22 asm("v86");
        register float v_c23 asm("v87");
        register float v_c24 asm("v88");
        register float v_c25 asm("v89");
        register float v_c26 asm("v90");
        register float v_c27 asm("v91");
        register float v_c28 asm("v92");
        register float v_c29 asm("v93");
        register float v_c30 asm("v94");
        register float v_c31 asm("v95");
        int32_t nan_hi = 0x7fff0000;
        int32_t nan_lo = 0x00007fff;

        // in smem, the layout is  M0(2)*K0(128)*M1(16)*K1(4)
        // every threads need 8xK in contiguous register
        // ... and every wave need the same data
        int lane_id  = threadIdx.x % 64;
        int sld_y_os = (lane_id % 16) * 4 + (lane_id / 16) * 128;
        sld_y_os *= 2;

        //                    y     y     p     p      p      y
        // reg before shfl  M0(2)*N0(2)*Nl(4)*Nw(4)*Mw(16)*Nv(4)
        // but order is N0*M0*Nv
        // in LDS we need store as
        //          M0(2)* N0(2) *  Nl(4) * Nw(4) * (Mw(16)*Nv(4) + 4)
        //             y    y       wave-id  lid/16  lid%16   v
        // sst(v3) = (v0/16*34 + v0%16 * 2 + wid*136) * 4
        int sfl_sst = (threadIdx.x % 16 * 4) + (threadIdx.x / 16) * (64 + 4);
        sfl_sst *= 2;

        // from LDS we need load as
        //          M0(2)*    N0(2) *  Nl(4) * Nw(4) * (Mw(16)         *  Nv(4) + 4)
        //        ( 2 issue)    (rem 32-lane)        (4 wave*4issue)   2lane*1ussue(pk2)
        // sld(v4) = v0/2 *34*4  + v0 % 2 *4 + wid*2 *4
        int sfl_sld = (lane_id % 2) * 2 + (lane_id / 2) * (64 + 4) + (threadIdx.x / 64) * 4;
        sfl_sld *= 2;

        // B nr->kr
        // clang-format off
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winline-asm"
        asm volatile(
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_BF16
#include "uk/flatmm_sn_uk_gfx9_32x128x512_1x4x1_16x16x16.inc"
            :[smem_]"+r"(smem),
            [s_loop_cnt]"+s"(loop_cnt),
                [c0]"+v" (v_c0),
                [c1]"+v" (v_c1),
                [c2]"+v" (v_c2),
                [c3]"+v" (v_c3),
                [c4]"+v" (v_c4),
                [c5]"+v" (v_c5),
                [c6]"+v" (v_c6),
                [c7]"+v" (v_c7),
                [c8]"+v" (v_c8),
                [c9]"+v" (v_c9),
                [c10]"+v"(v_c10),
                [c11]"+v"(v_c11),
                [c12]"+v"(v_c12),
                [c13]"+v"(v_c13),
                [c14]"+v"(v_c14),
                [c15]"+v"(v_c15),
                [c16]"+v"(v_c16),
                [c17]"+v"(v_c17),
                [c18]"+v"(v_c18),
                [c19]"+v"(v_c19),
                [c20]"+v"(v_c20),
                [c21]"+v"(v_c21),
                [c22]"+v"(v_c22),
                [c23]"+v"(v_c23),
                [c24]"+v"(v_c24),
                [c25]"+v"(v_c25),
                [c26]"+v"(v_c26),
                [c27]"+v"(v_c27),
                [c28]"+v"(v_c28),
                [c29]"+v"(v_c29),
                [c30]"+v"(v_c30),
                [c31]"+v"(v_c31)
            :
            [sld_a_base]"n"(0),
            [shfl_base]"n"(0),
            [v_sld_y_os]"v"(sld_y_os),
            [v_sfl_sld]"v"(sfl_sld),
            [v_sfl_sst]"v"(sfl_sst),
            [s_res_o0]"s"(res_o[0]),
                [s_res_o1]"s"(res_o[1]),
                //[s_res_o2]"s"(res_o[2]),
                //[s_res_o3]"s"(res_o[3]),
                [s_res_b0]"s"(res_b[0]),
                [s_res_b1]"s"(res_b[1]),
                [s_res_b2]"s"(res_b[2]),
                [s_res_b3]"s"(res_b[3]),
                [v_os_o0]"v"(static_cast<index_t>(cached_coords_o[number<0>{}] * sizeof(ODataType))),
                [v_os_o1]"v"(static_cast<index_t>(cached_coords_o[number<1>{}] * sizeof(ODataType))),
                [v_os_o2]"v"(static_cast<index_t>(cached_coords_o[number<2>{}] * sizeof(ODataType))),
                [v_os_o3]"v"(static_cast<index_t>(cached_coords_o[number<3>{}] * sizeof(ODataType))),
                [v_os_o4]"v"(static_cast<index_t>(cached_coords_o[number<4>{}] * sizeof(ODataType))),
                [v_os_o5]"v"(static_cast<index_t>(cached_coords_o[number<5>{}] * sizeof(ODataType))),
                [v_os_o6]"v"(static_cast<index_t>(cached_coords_o[number<6>{}] * sizeof(ODataType))),
                [v_os_o7]"v"(static_cast<index_t>(cached_coords_o[number<7>{}] * sizeof(ODataType))),
                [v_os_b0]"v"(static_cast<index_t>(cached_coords_b[number<0>{}] * sizeof(BDataType))),
                [v_os_b1]"v"(static_cast<index_t>(cached_coords_b[number<1>{}] * sizeof(BDataType))),
                [v_os_b2]"v"(static_cast<index_t>(cached_coords_b[number<2>{}] * sizeof(BDataType))),
                [v_os_b3]"v"(static_cast<index_t>(cached_coords_b[number<3>{}] * sizeof(BDataType))),
                [v_os_b4]"v"(static_cast<index_t>(cached_coords_b[number<4>{}] * sizeof(BDataType))),
                [v_os_b5]"v"(static_cast<index_t>(cached_coords_b[number<5>{}] * sizeof(BDataType))),
                [v_os_b6]"v"(static_cast<index_t>(cached_coords_b[number<6>{}] * sizeof(BDataType))),
                [v_os_b7]"v"(static_cast<index_t>(cached_coords_b[number<7>{}] * sizeof(BDataType))),

                [s_tile_os_o]"s"(tile_stride_o_bytes),
                [s_tile_os_b]"s"(tile_stride_b_bytes),
                [scale_0]"v"(s0),
                [scale_1]"v"(s1),
                [v_nan_lo]"v"(nan_lo),
                [v_nan_hi]"v"(nan_hi),
                [s_execflag_0]"s"(o_flags[number<0>{}]),
                [s_execflag_1]"s"(o_flags[number<1>{}]),
                [s_execflag_2]"s"(o_flags[number<2>{}]),
                [s_execflag_3]"s"(o_flags[number<3>{}]),
                [s_execflag_4]"s"(o_flags[number<4>{}]),
                [s_execflag_5]"s"(o_flags[number<5>{}]),
                [s_execflag_6]"s"(o_flags[number<6>{}]),
                [s_execflag_7]"s"(o_flags[number<7>{}])
            :
          "memory", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
          "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19",
          "a20", "a21", "a22", "a23", "a24", "a25", "a26", "a27", "a28", "a29",
          "a30", "a31", "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39",
          "a40", "a41", "a42", "a43", "a44", "a45", "a46", "a47", "a48", "a49",
          "a50", "a51", "a52", "a53", "a54", "a55", "a56", "a57", "a58", "a59",
          "a60", "a61", "a62", "a63", "a64", "a65", "a66", "a67", "a68", "a69",
          "a70", "a71", "a72", "a73", "a74", "a75", "a76", "a77", "a78", "a79",
          "a80", "a81", "a82", "a83", "a84", "a85", "a86", "a87", "a88", "a89",
          "a90", "a91", "a92", "a93", "a94", "a95", "a96", "a97", "a98", "a99",
          "a100", "a101", "a102", "a103", "a104", "a105", "a106", "a107",
          "a108", "a109", "a110", "a111", "a112", "a113", "a114", "a115",
          "a116", "a117", "a118", "a119", "a120", "a121", "a122", "a123",
          "a124", "a125", "a126", "a127", "a128", "a129", "a130", "a131",
          "a132", "a133", "a134", "a135", "a136", "a137", "a138", "a139",
          "a140", "a141", "a142", "a143", "a144", "a145", "a146", "a147",
          "a148", "a149", "a150", "a151", "a152", "a153", "a154", "a155",
          "a156", "a157", "a158", "a159", "a160", "a161", "a162", "a163",
          "a164", "a165", "a166", "a167", "a168", "a169", "a170", "a171",
          "a172", "a173", "a174", "a175", "a176", "a177", "a178", "a179",
          "a180", "a181", "a182", "a183", "a184", "a185", "a186", "a187",
          "a188", "a189", "a190", "a191", "a192", "a193", "a194", "a195",
          "a196", "a197", "a198", "a199", "a200", "a201", "a202", "a203",
          "a204", "a205", "a206", "a207", "a208", "a209", "a210", "a211",
          "a212", "a213", "a214", "a215", "a216", "a217", "a218", "a219",
          "a220", "a221", "a222", "a223", "a224", "a225", "a226", "a227",
          "a228", "a229", "a230", "a231", "a232", "a233", "a234", "a235",
          "a236", "a237", "a238", "a239", "a240", "a241", "a242", "a243",
          "a244", "a245", "a246", "a247", "a248", "a249", "a250", "a251",
          "a252", "a253", "a254", "a255", 
          "s8", "s9", "s12", "s13", "s14", "s15", "s38", "s39", "s52", "s86",
           "s36", "s37",
          "v50", "v54", "v55",
          "v64","v65","v66","v67","v68","v69","v70","v71",
          "v72","v73","v74","v75","v76","v77","v78","v79",
          "v80","v81","v82","v83","v84","v85","v86","v87",
          "v88","v89","v90","v91","v92","v93","v94","v95",
          "v128", "v129", "v130", "v131",
          "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139",
          "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147",
          "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155",
          "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163",
          "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171",
          "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179",
          "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187",
          "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195",
          "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203",
          "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211",
          "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219",
          "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227",
          "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235",
          "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243",
          "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251",
          "v252", "v253", "v254", "v255"
        );
#pragma clang diagnostic pop
        // clang-format on
    }
};

struct FlatmmSn_32x128x512_1x4x1_16x16x32_FP16 : public FlatmmSn_32x128x512_1x4x1_16x16x32_Base
{
    using BDataType = bf16_t;
    using ODataType = bf16_t;

    // TODO: need paired with tile_window_linear!
    // TODO: need call init_raw() before call this function!
    // template <typename AWindow, typename BWindow, typename OWindow, typename ScaleTensor>
    template <typename BRes,
              typename BCoords,
              typename ORes,
              typename OCoords,
              typename OFlags,
              typename ScaleTensor>
    CK_TILE_DEVICE auto
    operator()(const BRes& res_b,
               const BCoords& cached_coords_b,
               const ORes& res_o,
               const OCoords& cached_coords_o,
               const OFlags& o_flags, // this should be in sgpr
               CK_TILE_LDS_ADDR void* smem,
               index_t n, // loop along n dim
               const ScaleTensor& scale_,
               index_t tile_offset_b, // stride b is fixed to blockKr * blockW, but still can adjust
               index_t tile_offset_o)
    {
        static_assert(BCoords::size() == 8); // 8
        static_assert(OCoords::size() == 8);

        const index_t tile_stride_b_bytes = tile_offset_b * sizeof(BDataType);
        const index_t tile_stride_o_bytes = tile_offset_o * sizeof(ODataType);

        static_assert(ScaleTensor::size() == 2);
        float s0 = scale_[number<0>{}];
        float s1 = scale_[number<1>{}];

        index_t loop_cnt = n / Block_N;

        register float v_c0 asm("v64");
        register float v_c1 asm("v65");
        register float v_c2 asm("v66");
        register float v_c3 asm("v67");
        register float v_c4 asm("v68");
        register float v_c5 asm("v69");
        register float v_c6 asm("v70");
        register float v_c7 asm("v71");
        register float v_c8 asm("v72");
        register float v_c9 asm("v73");
        register float v_c10 asm("v74");
        register float v_c11 asm("v75");
        register float v_c12 asm("v76");
        register float v_c13 asm("v77");
        register float v_c14 asm("v78");
        register float v_c15 asm("v79");
        register float v_c16 asm("v80");
        register float v_c17 asm("v81");
        register float v_c18 asm("v82");
        register float v_c19 asm("v83");
        register float v_c20 asm("v84");
        register float v_c21 asm("v85");
        register float v_c22 asm("v86");
        register float v_c23 asm("v87");
        register float v_c24 asm("v88");
        register float v_c25 asm("v89");
        register float v_c26 asm("v90");
        register float v_c27 asm("v91");
        register float v_c28 asm("v92");
        register float v_c29 asm("v93");
        register float v_c30 asm("v94");
        register float v_c31 asm("v95");
        int32_t nan_hi = 0x7fff0000;
        int32_t nan_lo = 0x00007fff;

        // in smem, the layout is  M0(2)*K0(128)*M1(16)*K1(4)
        // every threads need 8xK in contiguous register
        // ... and every wave need the same data
        int lane_id  = threadIdx.x % 64;
        int sld_y_os = (lane_id % 16) * 4 + (lane_id / 16) * 128;
        sld_y_os *= 2;

        //                    y     y     p     p      p      y
        // reg before shfl  M0(2)*N0(2)*Nl(4)*Nw(4)*Mw(16)*Nv(4)
        // but order is N0*M0*Nv
        // in LDS we need store as
        //          M0(2)* N0(2) *  Nl(4) * Nw(4) * (Mw(16)*Nv(4) + 4)
        //             y    y       wave-id  lid/16  lid%16   v
        // sst(v3) = (v0/16*34 + v0%16 * 2 + wid*136) * 4
        int sfl_sst = (threadIdx.x % 16 * 4) + (threadIdx.x / 16) * (64 + 4);
        sfl_sst *= 2;

        // from LDS we need load as
        //          M0(2)*    N0(2) *  Nl(4) * Nw(4) * (Mw(16)         *  Nv(4) + 4)
        //        ( 2 issue)    (rem 32-lane)        (4 wave*4issue)   2lane*1ussue(pk2)
        // sld(v4) = v0/2 *34*4  + v0 % 2 *4 + wid*2 *4
        int sfl_sld = (lane_id % 2) * 2 + (lane_id / 2) * (64 + 4) + (threadIdx.x / 64) * 4;
        sfl_sld *= 2;

        // B nr->kr
        // clang-format off
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winline-asm"
        asm volatile(
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_FP16
#include "uk/flatmm_sn_uk_gfx9_32x128x512_1x4x1_16x16x16.inc"
            :[smem_]"+r"(smem),
            [s_loop_cnt]"+s"(loop_cnt),
                [c0]"+v" (v_c0),
                [c1]"+v" (v_c1),
                [c2]"+v" (v_c2),
                [c3]"+v" (v_c3),
                [c4]"+v" (v_c4),
                [c5]"+v" (v_c5),
                [c6]"+v" (v_c6),
                [c7]"+v" (v_c7),
                [c8]"+v" (v_c8),
                [c9]"+v" (v_c9),
                [c10]"+v"(v_c10),
                [c11]"+v"(v_c11),
                [c12]"+v"(v_c12),
                [c13]"+v"(v_c13),
                [c14]"+v"(v_c14),
                [c15]"+v"(v_c15),
                [c16]"+v"(v_c16),
                [c17]"+v"(v_c17),
                [c18]"+v"(v_c18),
                [c19]"+v"(v_c19),
                [c20]"+v"(v_c20),
                [c21]"+v"(v_c21),
                [c22]"+v"(v_c22),
                [c23]"+v"(v_c23),
                [c24]"+v"(v_c24),
                [c25]"+v"(v_c25),
                [c26]"+v"(v_c26),
                [c27]"+v"(v_c27),
                [c28]"+v"(v_c28),
                [c29]"+v"(v_c29),
                [c30]"+v"(v_c30),
                [c31]"+v"(v_c31)
            :
            [sld_a_base]"n"(0),
            [shfl_base]"n"(0),
            [v_sld_y_os]"v"(sld_y_os),
            [v_sfl_sld]"v"(sfl_sld),
            [v_sfl_sst]"v"(sfl_sst),
            [s_res_o0]"s"(res_o[0]),
                [s_res_o1]"s"(res_o[1]),
                //[s_res_o2]"s"(res_o[2]),
                //[s_res_o3]"s"(res_o[3]),
                [s_res_b0]"s"(res_b[0]),
                [s_res_b1]"s"(res_b[1]),
                [s_res_b2]"s"(res_b[2]),
                [s_res_b3]"s"(res_b[3]),
                [v_os_o0]"v"(static_cast<index_t>(cached_coords_o[number<0>{}] * sizeof(ODataType))),
                [v_os_o1]"v"(static_cast<index_t>(cached_coords_o[number<1>{}] * sizeof(ODataType))),
                [v_os_o2]"v"(static_cast<index_t>(cached_coords_o[number<2>{}] * sizeof(ODataType))),
                [v_os_o3]"v"(static_cast<index_t>(cached_coords_o[number<3>{}] * sizeof(ODataType))),
                [v_os_o4]"v"(static_cast<index_t>(cached_coords_o[number<4>{}] * sizeof(ODataType))),
                [v_os_o5]"v"(static_cast<index_t>(cached_coords_o[number<5>{}] * sizeof(ODataType))),
                [v_os_o6]"v"(static_cast<index_t>(cached_coords_o[number<6>{}] * sizeof(ODataType))),
                [v_os_o7]"v"(static_cast<index_t>(cached_coords_o[number<7>{}] * sizeof(ODataType))),
                [v_os_b0]"v"(static_cast<index_t>(cached_coords_b[number<0>{}] * sizeof(BDataType))),
                [v_os_b1]"v"(static_cast<index_t>(cached_coords_b[number<1>{}] * sizeof(BDataType))),
                [v_os_b2]"v"(static_cast<index_t>(cached_coords_b[number<2>{}] * sizeof(BDataType))),
                [v_os_b3]"v"(static_cast<index_t>(cached_coords_b[number<3>{}] * sizeof(BDataType))),
                [v_os_b4]"v"(static_cast<index_t>(cached_coords_b[number<4>{}] * sizeof(BDataType))),
                [v_os_b5]"v"(static_cast<index_t>(cached_coords_b[number<5>{}] * sizeof(BDataType))),
                [v_os_b6]"v"(static_cast<index_t>(cached_coords_b[number<6>{}] * sizeof(BDataType))),
                [v_os_b7]"v"(static_cast<index_t>(cached_coords_b[number<7>{}] * sizeof(BDataType))),

                [s_tile_os_o]"s"(tile_stride_o_bytes),
                [s_tile_os_b]"s"(tile_stride_b_bytes),
                [scale_0]"v"(s0),
                [scale_1]"v"(s1),
                [v_nan_lo]"v"(nan_lo),
                [v_nan_hi]"v"(nan_hi),
                [s_execflag_0]"s"(o_flags[number<0>{}]),
                [s_execflag_1]"s"(o_flags[number<1>{}]),
                [s_execflag_2]"s"(o_flags[number<2>{}]),
                [s_execflag_3]"s"(o_flags[number<3>{}]),
                [s_execflag_4]"s"(o_flags[number<4>{}]),
                [s_execflag_5]"s"(o_flags[number<5>{}]),
                [s_execflag_6]"s"(o_flags[number<6>{}]),
                [s_execflag_7]"s"(o_flags[number<7>{}])
            :
          "memory", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
          "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19",
          "a20", "a21", "a22", "a23", "a24", "a25", "a26", "a27", "a28", "a29",
          "a30", "a31", "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39",
          "a40", "a41", "a42", "a43", "a44", "a45", "a46", "a47", "a48", "a49",
          "a50", "a51", "a52", "a53", "a54", "a55", "a56", "a57", "a58", "a59",
          "a60", "a61", "a62", "a63", "a64", "a65", "a66", "a67", "a68", "a69",
          "a70", "a71", "a72", "a73", "a74", "a75", "a76", "a77", "a78", "a79",
          "a80", "a81", "a82", "a83", "a84", "a85", "a86", "a87", "a88", "a89",
          "a90", "a91", "a92", "a93", "a94", "a95", "a96", "a97", "a98", "a99",
          "a100", "a101", "a102", "a103", "a104", "a105", "a106", "a107",
          "a108", "a109", "a110", "a111", "a112", "a113", "a114", "a115",
          "a116", "a117", "a118", "a119", "a120", "a121", "a122", "a123",
          "a124", "a125", "a126", "a127", "a128", "a129", "a130", "a131",
          "a132", "a133", "a134", "a135", "a136", "a137", "a138", "a139",
          "a140", "a141", "a142", "a143", "a144", "a145", "a146", "a147",
          "a148", "a149", "a150", "a151", "a152", "a153", "a154", "a155",
          "a156", "a157", "a158", "a159", "a160", "a161", "a162", "a163",
          "a164", "a165", "a166", "a167", "a168", "a169", "a170", "a171",
          "a172", "a173", "a174", "a175", "a176", "a177", "a178", "a179",
          "a180", "a181", "a182", "a183", "a184", "a185", "a186", "a187",
          "a188", "a189", "a190", "a191", "a192", "a193", "a194", "a195",
          "a196", "a197", "a198", "a199", "a200", "a201", "a202", "a203",
          "a204", "a205", "a206", "a207", "a208", "a209", "a210", "a211",
          "a212", "a213", "a214", "a215", "a216", "a217", "a218", "a219",
          "a220", "a221", "a222", "a223", "a224", "a225", "a226", "a227",
          "a228", "a229", "a230", "a231", "a232", "a233", "a234", "a235",
          "a236", "a237", "a238", "a239", "a240", "a241", "a242", "a243",
          "a244", "a245", "a246", "a247", "a248", "a249", "a250", "a251",
          "a252", "a253", "a254", "a255", 
          "s8", "s9", "s12", "s13", "s14", "s15", "s38", "s39", "s52", "s86",
           "s36", "s37",
          "v50", "v54", "v55",
          "v64","v65","v66","v67","v68","v69","v70","v71",
          "v72","v73","v74","v75","v76","v77","v78","v79",
          "v80","v81","v82","v83","v84","v85","v86","v87",
          "v88","v89","v90","v91","v92","v93","v94","v95",
          "v128", "v129", "v130", "v131",
          "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139",
          "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147",
          "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155",
          "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163",
          "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171",
          "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179",
          "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187",
          "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195",
          "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203",
          "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211",
          "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219",
          "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227",
          "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235",
          "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243",
          "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251",
          "v252", "v253", "v254", "v255"
        );
#pragma clang diagnostic pop
        // clang-format on
    }
};

} // namespace ck_tile
