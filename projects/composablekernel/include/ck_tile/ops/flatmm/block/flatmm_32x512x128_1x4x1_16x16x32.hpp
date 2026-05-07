// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/flatmm/block/flatmm_uk_config.hpp"

namespace ck_tile {

// A async load to LDS, B direct to AGPR
// B matrix preshuffled in br*kr*w
// require 4 wave, occupancy=1c
// agpr useage:256
// vgpr usage:64(A local) + 64(acc) + 8(os_a) + 8(os_b) = 144 (rem:112)
//
// for this gemm, 4 16x16x16 transposed layout
//  input A vpgpr layout
//   v0-v15: [ 0:15](gemm_m)x128(gemm_k)
//  v16-v31: [16:31](gemm_m)x128(gemm_k)

//  input B vpgpr layout
//   v0-v15: [  0: 15](gemm_n)x128(gemm_k)
//  v16-v31: [ 64: 79](gemm_n)x128(gemm_k)
//  ......................
//  v111-v127: [448:463](gemm_n)x128(gemm_k)

//  output C vpgpr layout
//   v0-v3 : [ 0:15](gemm_m)x[ 0: 15](gemm_n)
//   v4-v7 : [16:31](gemm_m)x[ 0: 15](gemm_n)
//   v8-v11: [ 0:15](gemm_m)x[64: 79](gemm_n)
//  v12-v15: [16:31](gemm_m)x[64: 79](gemm_n)
//  ......................
//  v56-v59: [ 0:15](gemm_m)x[448:463](gemm_n)
//  v60-v63: [16:31](gemm_m)x[448:463](gemm_n)
struct Flatmm_32x512x128_1x4x1_16x16x32_Base // for f16/bf16
{
    static constexpr index_t Block_M = 32;
    static constexpr index_t Block_N = 512;
    static constexpr index_t Block_K = 128;

    static constexpr index_t WarpPerBlock_M = 1;
    static constexpr index_t WarpPerBlock_N = 4;
    static constexpr index_t WarpPerBlock_K = 1;

    static constexpr index_t NumWarps = 4;

    static constexpr index_t Warp_M = 16;
    static constexpr index_t Warp_N = 16;
    static constexpr index_t Warp_K = 32; // 16 * SubKPacks

    static constexpr index_t BlockSize = 256;

    static constexpr index_t SubKPacks = 2; // this is used to gurantee every threads can do dwordx4

    // TODO: note Nr/Kr/W need consider SubKPacks
    static constexpr index_t Block_W  = Warp_N * Warp_K;  // 512 element
    static constexpr index_t Block_Nr = Block_N / Warp_N; // 32 element, 4 per wave
    static constexpr index_t Block_Kr = Block_K / Warp_K; // 4

    static constexpr index_t Repeat_M = Block_M / (Warp_M * WarpPerBlock_M); // 2
    static constexpr index_t Repeat_N = Block_N / (Warp_N * WarpPerBlock_N); // 8
    static constexpr index_t Repeat_K = Block_K / (Warp_K * WarpPerBlock_K); // 8/2=4

    private:
    template <index_t LanesPerK, index_t WarpSize, typename = void>
    struct LdsStoreDescSelector;

    template <index_t LanesPerK, index_t WarpSize>
    struct LdsStoreDescSelector<LanesPerK, WarpSize, std::enable_if_t<(LanesPerK >= WarpSize)>>
    {
        template <index_t NumWarps, index_t Block_M, index_t Block_K, index_t KVector, index_t KPad>
        static CK_TILE_HOST_DEVICE constexpr auto MakeDesc()
        {
            // need multiple waves to load K
            static_assert(LanesPerK % WarpSize == 0);
            constexpr index_t wavesPerK = LanesPerK / WarpSize;
            if constexpr(wavesPerK > NumWarps)
            {
                // TODO: need multiple issues along K to load all data
            }
            else
            {
                constexpr index_t wavesPerM     = NumWarps / wavesPerK;
                constexpr index_t NumIssues     = Block_M / wavesPerM;
                constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<NumIssues>{},                             // m0
                               number<wavesPerM>{},                             // m1
                               number<wavesPerK>{},                             // k0
                               number<WarpSize>{},                              // k1
                               number<KVector>{}),                              // k2
                    make_tuple(number<NumWarps*(WarpSize * KVector + KPad)>{},  // m0
                               number<wavesPerK*(WarpSize * KVector + KPad)>{}, // m1
                               number<WarpSize * KVector + KPad>{},             // k0
                               number<KVector>{},                               // k1
                               number<1>{}),                                    // k2
                    number<KVector>{}, // lds store vector(actually no explicit store)
                    number<1>{});

                constexpr auto lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
                    lds_block_desc_0,
                    make_tuple(
                        make_pass_through_transform(number<NumIssues>{}),
                        make_merge_transform(make_tuple(number<wavesPerM>{}, number<wavesPerK>{})),
                        make_merge_transform(make_tuple(number<WarpSize>{}, number<KVector>{}))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                return lds_block_desc_issues_warps_lanes;
            }
        }
    };

    template <index_t LanesPerK, index_t WarpSize>
    struct LdsStoreDescSelector<LanesPerK, WarpSize, std::enable_if_t<(LanesPerK < WarpSize)>>
    {
        template <index_t NumWarps, index_t Block_M, index_t Block_K, index_t KVector, index_t KPad>
        static CK_TILE_HOST_DEVICE constexpr auto MakeDesc()
        {
            // lanes within a wave load different M but same K
            static_assert(WarpSize % LanesPerK == 0);
            constexpr index_t LaneGroups = WarpSize / LanesPerK; // along m
            constexpr index_t NumIssues  = Block_M / (LaneGroups * NumWarps);

            constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumIssues>{},                            // m0
                           number<LaneGroups>{},                           // m1
                           number<NumWarps>{},                             // m2
                           number<LanesPerK>{},                            // k0
                           number<KVector>{}),                             // k1
                make_tuple(number<NumWarps*(WarpSize * KVector + KPad)>{}, // m0
                           number<Block_K>{},                              // m1
                           number<WarpSize * KVector + KPad>{},            // m2
                           number<KVector>{},                              // k0
                           number<1>{}),                                   // k1
                number<KVector>{}, // lds store vector(actually no explicit store)
                number<1>{});

            constexpr auto lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
                lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<NumIssues>{}),
                           make_pass_through_transform(number<NumWarps>{}),
                           make_merge_transform(make_tuple(
                               number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
                make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

            return lds_block_desc_issues_warps_lanes;
        }
    };

    public:
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

    static CK_TILE_DEVICE constexpr auto MakeCBlockTile()
    {
        using CDataType             = float;
        constexpr auto c_block_dstr = MakeCBlockDist();
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsStoreDesc_A()
    {
        // A async->LDS
        constexpr index_t WarpSize = ck_tile::get_warp_size();

        constexpr index_t KPack_  = 8;      // GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t KVector = 2;      // GetAlignment_A<Problem>(); // async copy 1 dword
        constexpr index_t KPad    = KPack_; // pad between warps

        static_assert(Block_K % KVector == 0);
        constexpr index_t LanesPerK = Block_K / KVector; // how many thread loading K

        return LdsStoreDescSelector<LanesPerK, WarpSize>::
            template MakeDesc<NumWarps, Block_M, Block_K, KVector, KPad>();
    }

    // template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsLoadDesc_A()
    {
        // load from LDS to register, every wave has same layout
        constexpr index_t KPack_ = 8;      // GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t KPad   = KPack_; // pad between warps

        constexpr index_t kAMLane     = 16;
        constexpr index_t kABKLane    = 4;
        constexpr index_t kABKPerLane = 4;
        constexpr index_t kKIter      = 2;
        static_assert(KPack_ == (kABKPerLane * kKIter));

        constexpr auto lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<Repeat_M>{}, // m0 y
                                                    number<kAMLane>{},  // m1 p
                                                    number<Repeat_K>{}, // k0 y
                                                    number<kABKLane>{}, // k1 p
                                                    number<KPack_>{}),  // k2 y-vector
                                         make_tuple(number<kAMLane*(Block_K + KPad)>{}, // m0
                                                    number<Block_K + KPad>{},           // m1
                                                    number<kABKLane * KPack_>{},        // k0
                                                    number<KPack_>{},                   // k1
                                                    number<1>{}),                       // k2
                                         number<KPack_>{}, // lds load vector
                                         number<1>{});

        constexpr auto lds_desc_m_k = transform_tensor_descriptor(
            lds_block_desc_0,
            make_tuple(make_merge_transform(make_tuple(number<Repeat_M>{}, number<kAMLane>{})),
                       make_merge_transform(
                           make_tuple(number<Repeat_K>{}, number<kABKLane>{}, number<KPack_>{}))),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc_m_k;
    }

    static constexpr auto GetGemm_AWarpEnc()
    {
        constexpr index_t kAMLane     = 16;
        constexpr index_t kABKLane    = 4;
        constexpr index_t kABKPerLane = 4;
        constexpr index_t kKIter      = 2;

        using enc_ = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<kAMLane>, sequence<kABKLane, kABKPerLane * kKIter>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>;
        return enc_{};
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // return 32 * (128 + 8) * sizeof(bf16_t);
        return MakeLdsLoadDesc_A().get_element_space_size() * sizeof(bf16_t) * 2; // 2 lds buffers
    }
};

// clang-format off
#define _EXPAND_ASM_ARGS_OUT_ONE_ACC        \
            [s_loop_cnt]"+s"(loop_cnt),     \
                [v_acc_0]"+v"(v_acc[0]),    \
                [v_acc_1]"+v"(v_acc[1]),    \
                [v_acc_2]"+v"(v_acc[2]),    \
                [v_acc_3]"+v"(v_acc[3]),    \
                [v_acc_4]"+v"(v_acc[4]),    \
                [v_acc_5]"+v"(v_acc[5]),    \
                [v_acc_6]"+v"(v_acc[6]),    \
                [v_acc_7]"+v"(v_acc[7]),    \
                [v_acc_8]"+v"(v_acc[8]),    \
                [v_acc_9]"+v"(v_acc[9]),    \
                [v_acc_10]"+v"(v_acc[10]),    \
                [v_acc_11]"+v"(v_acc[11]),    \
                [v_acc_12]"+v"(v_acc[12]),    \
                [v_acc_13]"+v"(v_acc[13]),    \
                [v_acc_14]"+v"(v_acc[14]),    \
                [v_acc_15]"+v"(v_acc[15]),    \
                [s_mem_]"+r"(smem)

#define _EXPAND_ASM_ARGS_OUT_TWO_ACC        \
            [s_loop_cnt]"+s"(loop_cnt),     \
                [v_acc_0]"+v"(v_acc[0]),    \
                [v_acc_1]"+v"(v_acc[1]),    \
                [v_acc_2]"+v"(v_acc[2]),    \
                [v_acc_3]"+v"(v_acc[3]),    \
                [v_acc_4]"+v"(v_acc[4]),    \
                [v_acc_5]"+v"(v_acc[5]),    \
                [v_acc_6]"+v"(v_acc[6]),    \
                [v_acc_7]"+v"(v_acc[7]),    \
                [v_acc_8]"+v"(v_acc[8]),    \
                [v_acc_9]"+v"(v_acc[9]),    \
                [v_acc_10]"+v"(v_acc[10]),    \
                [v_acc_11]"+v"(v_acc[11]),    \
                [v_acc_12]"+v"(v_acc[12]),    \
                [v_acc_13]"+v"(v_acc[13]),    \
                [v_acc_14]"+v"(v_acc[14]),    \
                [v_acc_15]"+v"(v_acc[15]),    \
                [v_acc_16]"+v"(v_acc[16]),    \
                [v_acc_17]"+v"(v_acc[17]),    \
                [v_acc_18]"+v"(v_acc[18]),    \
                [v_acc_19]"+v"(v_acc[19]),    \
                [v_acc_20]"+v"(v_acc[20]),    \
                [v_acc_21]"+v"(v_acc[21]),    \
                [v_acc_22]"+v"(v_acc[22]),    \
                [v_acc_23]"+v"(v_acc[23]),    \
                [v_acc_24]"+v"(v_acc[24]),    \
                [v_acc_25]"+v"(v_acc[25]),    \
                [v_acc_26]"+v"(v_acc[26]),    \
                [v_acc_27]"+v"(v_acc[27]),    \
                [v_acc_28]"+v"(v_acc[28]),    \
                [v_acc_29]"+v"(v_acc[29]),    \
                [v_acc_30]"+v"(v_acc[30]),    \
                [v_acc_31]"+v"(v_acc[31]),    \
                [s_mem_]"+r"(smem)

#define _EXPAND_ASM_ARGS_IN     \
              [s_res_a0]"s"(res_a[0]),    \
                [s_res_a1]"s"(res_a[1]),    \
                [s_res_a2]"s"(res_a[2]),    \
                [s_res_a3]"s"(res_a[3]),    \
                [s_res_b0]"s"(res_b[0]),    \
                [s_res_b1]"s"(res_b[1]),    \
                [s_res_b2]"s"(res_b[2]),    \
                [s_res_b3]"s"(res_b[3]),    \
                [v_os_a0]"v"(static_cast<index_t>(cached_coords_a[number<0>{}] * sizeof(ADataType))),    \
                [v_os_a1]"v"(static_cast<index_t>(cached_coords_a[number<1>{}] * sizeof(ADataType))),    \
                [v_os_a2]"v"(static_cast<index_t>(cached_coords_a[number<2>{}] * sizeof(ADataType))),    \
                [v_os_a3]"v"(static_cast<index_t>(cached_coords_a[number<3>{}] * sizeof(ADataType))),    \
                [v_os_a4]"v"(static_cast<index_t>(cached_coords_a[number<4>{}] * sizeof(ADataType))),    \
                [v_os_a5]"v"(static_cast<index_t>(cached_coords_a[number<5>{}] * sizeof(ADataType))),    \
                [v_os_a6]"v"(static_cast<index_t>(cached_coords_a[number<6>{}] * sizeof(ADataType))),    \
                [v_os_a7]"v"(static_cast<index_t>(cached_coords_a[number<7>{}] * sizeof(ADataType))),    \
                                                                                                        \
                [v_os_b0]"v"(static_cast<index_t>(cached_coords_b[number<0>{}] * sizeof(BDataType))),    \
                [v_os_b1]"v"(static_cast<index_t>(cached_coords_b[number<1>{}] * sizeof(BDataType))),    \
                [v_os_b2]"v"(static_cast<index_t>(cached_coords_b[number<2>{}] * sizeof(BDataType))),    \
                [v_os_b3]"v"(static_cast<index_t>(cached_coords_b[number<3>{}] * sizeof(BDataType))),    \
                [v_os_b4]"v"(static_cast<index_t>(cached_coords_b[number<4>{}] * sizeof(BDataType))),    \
                [v_os_b5]"v"(static_cast<index_t>(cached_coords_b[number<5>{}] * sizeof(BDataType))),    \
                [v_os_b6]"v"(static_cast<index_t>(cached_coords_b[number<6>{}] * sizeof(BDataType))),    \
                [v_os_b7]"v"(static_cast<index_t>(cached_coords_b[number<7>{}] * sizeof(BDataType))),    \
                                                                                                            \
                [v_os_slda]"v"(static_cast<index_t>(a_sld.cached_coords_[number<0>{}].get_offset() * sizeof(ADataType))),\
                [s_m0_init]"s"(m0_init_value),    \
                [s_size_per_issue]"s"(size_per_issue),    \
                [smem_sz]"n"(smem_buf_size),   \
                [sld_os_0]"n"(sld_os[number<0>{}].value),    \
                [sld_os_1]"n"(sld_os[number<1>{}].value),    \
                [sld_os_2]"n"(sld_os[number<2>{}].value),    \
                [sld_os_3]"n"(sld_os[number<3>{}].value),    \
                [sld_os_4]"n"(sld_os[number<4>{}].value),    \
                [sld_os_5]"n"(sld_os[number<5>{}].value),    \
                [sld_os_6]"n"(sld_os[number<6>{}].value),    \
                [sld_os_7]"n"(sld_os[number<7>{}].value),    \
                [s_tile_os_a]"s"(tile_offset_a_bytes),    \
                [s_tile_os_b]"s"(tile_offset_b_bytes)

#define _EXPAND_ASM_ARGS_CLOBBER     \
          "memory", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",    \
          "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19",    \
          "a20", "a21", "a22", "a23", "a24", "a25", "a26", "a27", "a28", "a29",    \
          "a30", "a31", "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39",    \
          "a40", "a41", "a42", "a43", "a44", "a45", "a46", "a47", "a48", "a49",    \
          "a50", "a51", "a52", "a53", "a54", "a55", "a56", "a57", "a58", "a59",    \
          "a60", "a61", "a62", "a63", "a64", "a65", "a66", "a67", "a68", "a69",    \
          "a70", "a71", "a72", "a73", "a74", "a75", "a76", "a77", "a78", "a79",    \
          "a80", "a81", "a82", "a83", "a84", "a85", "a86", "a87", "a88", "a89",    \
          "a90", "a91", "a92", "a93", "a94", "a95", "a96", "a97", "a98", "a99",    \
          "a100", "a101", "a102", "a103", "a104", "a105", "a106", "a107",    \
          "a108", "a109", "a110", "a111", "a112", "a113", "a114", "a115",    \
          "a116", "a117", "a118", "a119", "a120", "a121", "a122", "a123",    \
          "a124", "a125", "a126", "a127", "a128", "a129", "a130", "a131",    \
          "a132", "a133", "a134", "a135", "a136", "a137", "a138", "a139",    \
          "a140", "a141", "a142", "a143", "a144", "a145", "a146", "a147",    \
          "a148", "a149", "a150", "a151", "a152", "a153", "a154", "a155",    \
          "a156", "a157", "a158", "a159", "a160", "a161", "a162", "a163",    \
          "a164", "a165", "a166", "a167", "a168", "a169", "a170", "a171",    \
          "a172", "a173", "a174", "a175", "a176", "a177", "a178", "a179",    \
          "a180", "a181", "a182", "a183", "a184", "a185", "a186", "a187",    \
          "a188", "a189", "a190", "a191", "a192", "a193", "a194", "a195",    \
          "a196", "a197", "a198", "a199", "a200", "a201", "a202", "a203",    \
          "a204", "a205", "a206", "a207", "a208", "a209", "a210", "a211",    \
          "a212", "a213", "a214", "a215", "a216", "a217", "a218", "a219",    \
          "a220", "a221", "a222", "a223", "a224", "a225", "a226", "a227",    \
          "a228", "a229", "a230", "a231", "a232", "a233", "a234", "a235",    \
          "a236", "a237", "a238", "a239", "a240", "a241", "a242", "a243",    \
          "a244", "a245", "a246", "a247", "a248", "a249", "a250", "a251",    \
          "a252", "a253", "a254", "a255",     \
          "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",    \
          "s86",                         \
          "v64", "v65", "v66", "v67", "v68", "v69",                 \
          "v70", "v71", "v72", "v73", "v74", "v75", "v76", "v77", "v78", "v79",     \
          "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89",    \
          "v90", "v91", "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99",    \
          "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107",    \
          "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115",    \
          "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123",    \
          "v124", "v125", "v126", "v127"
// clang-format on

struct Flatmm_32x512x128_1x4x1_16x16x32_BF16 : public Flatmm_32x512x128_1x4x1_16x16x32_Base
{
    using ADataType = bf16_t;
    using BDataType = bf16_t;

    // TODO: need paired with tile_window_linear!
    // TODO: need call init_raw() before call this function!
    // Is2B: originally for B matrix we have 2 prefetch buffers. If set this to true
    // we can support A matric serve 2 B matrix, B0/B1, each B0/B1 still have same tile size
    template <typename ARes, typename ACoords, typename BRes, typename BCoords, bool Is2B = false>
    CK_TILE_DEVICE auto
    operator()(const ARes& res_a,
               const ACoords& cached_coords_a,
               const BRes& res_b,
               const BCoords& cached_coords_b,
               CK_TILE_LDS_ADDR void* smem,
               index_t k,
               index_t tile_offset_a, // for each tile, the offset to move for each unroll
               index_t tile_offset_b,
               bool_constant<Is2B> = {}) // for each tile, the offset to move for each unroll
    {
        static_assert(ACoords::size() == Block_M * Block_K / BlockSize / 2 /*2x per dword*/); // 8
        static_assert(BCoords::size() == Repeat_N);

        auto a_sst = make_tile_window(
            make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem), MakeLdsStoreDesc_A()),
            MakeLdsStoreDesc_A().get_lengths(),
            {0, 0, 0});

        auto a_sld = [&]() {
            constexpr auto a_warp_enc_      = GetGemm_AWarpEnc();
            constexpr auto a_outer_dstr_enc = tile_distribution_encoding<
                sequence<WarpPerBlock_N>,
                tuple<sequence<Repeat_M, WarpPerBlock_M>, sequence<Repeat_K>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto a_block_dstr_encode =
                detail::make_embed_tile_distribution_encoding(a_outer_dstr_enc, a_warp_enc_);
            return make_tile_window_linear(
                make_tensor_view<address_space_enum::lds>(
                    reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem), MakeLdsLoadDesc_A()),
                MakeLdsLoadDesc_A().get_lengths(),
                {0, 0},
                make_static_tile_distribution(a_block_dstr_encode));
        }();

        const index_t tile_offset_a_bytes = tile_offset_a * sizeof(ADataType);
        const index_t tile_offset_b_bytes = tile_offset_b * sizeof(BDataType);

        const auto [m0_init_value, size_per_issue] = get_async_store_smem_info(a_sst);
        constexpr auto smem_buf_size =
            MakeLdsLoadDesc_A().get_element_space_size() * sizeof(ADataType);
        static_assert(a_sld.get_num_of_access() == 8);
        constexpr auto sld_os = generate_tuple(
            [&](auto i_access) {
                return number<a_sld.get_bottom_linear_offset(i_access) * sizeof(ADataType)>{};
            },
            number<a_sld.get_num_of_access()>{});

        index_t loop_cnt = k / Block_K;

        if constexpr(Is2B)
        {
            // this is the acc thread buffer
            fp32x4_t v_acc[32]{.0f};

            // B nr->kr
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winline-asm"
            // clang-format off
            asm volatile(
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_BF16
#define CK_TILE_FLATMM_UK_2B 1
#include "uk/flatmm_uk_gfx9_32x512x128_1x1x1_16x16x16.inc"
                : _EXPAND_ASM_ARGS_OUT_TWO_ACC
                : _EXPAND_ASM_ARGS_IN, 
                    [s_res_b4]"s"(res_b[4]), 
                    [s_res_b5]"s"(res_b[5]),
                    [s_res_b6]"s"(res_b[6]),
                    [s_res_b7]"s"(res_b[7])
                : _EXPAND_ASM_ARGS_CLOBBER, "s24", "s25", "s26", "s27"
            );
            // clang-format on
#pragma clang diagnostic pop

            // return local scratch
            auto c = make_tuple(MakeCBlockTile(), MakeCBlockTile());
            for(auto i = 0; i < 16; i++)
            {
                c.at(number<0>{}).get_thread_buffer()[4 * i + 0] = v_acc[i].x;
                c.at(number<0>{}).get_thread_buffer()[4 * i + 1] = v_acc[i].y;
                c.at(number<0>{}).get_thread_buffer()[4 * i + 2] = v_acc[i].z;
                c.at(number<0>{}).get_thread_buffer()[4 * i + 3] = v_acc[i].w;
            }
            for(auto i = 0; i < 16; i++)
            {
                c.at(number<1>{}).get_thread_buffer()[4 * i + 0] = v_acc[16 + i].x;
                c.at(number<1>{}).get_thread_buffer()[4 * i + 1] = v_acc[16 + i].y;
                c.at(number<1>{}).get_thread_buffer()[4 * i + 2] = v_acc[16 + i].z;
                c.at(number<1>{}).get_thread_buffer()[4 * i + 3] = v_acc[16 + i].w;
            }
            return c;
        }
        else
        {
            // this is the acc thread buffer
            fp32x4_t v_acc[16]{.0f};

            // B nr->kr
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winline-asm"
            // clang-format off
            asm volatile(
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_BF16
#include "uk/flatmm_uk_gfx9_32x512x128_1x1x1_16x16x16.inc"
                : _EXPAND_ASM_ARGS_OUT_ONE_ACC
                : _EXPAND_ASM_ARGS_IN
                : _EXPAND_ASM_ARGS_CLOBBER
            );
            // clang-format on
#pragma clang diagnostic pop

            // return local scratch
            auto c = MakeCBlockTile();
            for(auto i = 0; i < 16; i++)
            {
                c.get_thread_buffer()[4 * i + 0] = v_acc[i].x;
                c.get_thread_buffer()[4 * i + 1] = v_acc[i].y;
                c.get_thread_buffer()[4 * i + 2] = v_acc[i].z;
                c.get_thread_buffer()[4 * i + 3] = v_acc[i].w;
            }
            return c;
        }
    }
};

struct Flatmm_32x512x128_1x4x1_16x16x32_FP16 : public Flatmm_32x512x128_1x4x1_16x16x32_Base
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;

    // TODO: need paired with tile_window_linear!
    // TODO: need call init_raw() before call this function!
    template <typename ARes, typename ACoords, typename BRes, typename BCoords, bool Is2B = false>
    CK_TILE_DEVICE auto
    operator()(const ARes& res_a,
               const ACoords& cached_coords_a,
               const BRes& res_b,
               const BCoords& cached_coords_b,
               CK_TILE_LDS_ADDR void* smem,
               index_t k,
               index_t tile_offset_a, // for each tile, the offset to move for each unroll
               index_t tile_offset_b, // for each tile, the offset to move for each unroll
               bool_constant<Is2B> = {})
    {
        static_assert(ACoords::size() == Block_M * Block_K / BlockSize / 2 /*2x per dword*/); // 8
        static_assert(BCoords::size() == Repeat_N);

        auto a_sst = make_tile_window(
            make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem), MakeLdsStoreDesc_A()),
            MakeLdsStoreDesc_A().get_lengths(),
            {0, 0, 0});

        auto a_sld = [&]() {
            constexpr auto a_warp_enc_      = GetGemm_AWarpEnc();
            constexpr auto a_outer_dstr_enc = tile_distribution_encoding<
                sequence<WarpPerBlock_N>,
                tuple<sequence<Repeat_M, WarpPerBlock_M>, sequence<Repeat_K>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto a_block_dstr_encode =
                detail::make_embed_tile_distribution_encoding(a_outer_dstr_enc, a_warp_enc_);
            return make_tile_window_linear(
                make_tensor_view<address_space_enum::lds>(
                    reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem), MakeLdsLoadDesc_A()),
                MakeLdsLoadDesc_A().get_lengths(),
                {0, 0},
                make_static_tile_distribution(a_block_dstr_encode));
        }();

        const index_t tile_offset_a_bytes = tile_offset_a * sizeof(ADataType);
        const index_t tile_offset_b_bytes = tile_offset_b * sizeof(BDataType);

        const auto [m0_init_value, size_per_issue] = get_async_store_smem_info(a_sst);
        constexpr auto smem_buf_size =
            MakeLdsLoadDesc_A().get_element_space_size() * sizeof(ADataType);
        static_assert(a_sld.get_num_of_access() == 8);
        constexpr auto sld_os = generate_tuple(
            [&](auto i_access) {
                return number<a_sld.get_bottom_linear_offset(i_access) * sizeof(ADataType)>{};
            },
            number<a_sld.get_num_of_access()>{});

        index_t loop_cnt = k / Block_K;

        if constexpr(Is2B)
        {
            // this is the acc thread buffer
            fp32x4_t v_acc[32]{.0f};

            // B nr->kr
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winline-asm"
            // clang-format off
            asm volatile(
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_FP16
#define CK_TILE_FLATMM_UK_2B 1
#include "uk/flatmm_uk_gfx9_32x512x128_1x1x1_16x16x16.inc"
                : _EXPAND_ASM_ARGS_OUT_TWO_ACC
                : _EXPAND_ASM_ARGS_IN, 
                    [s_res_b4]"s"(res_b[4]), 
                    [s_res_b5]"s"(res_b[5]),
                    [s_res_b6]"s"(res_b[6]),
                    [s_res_b7]"s"(res_b[7])
                : _EXPAND_ASM_ARGS_CLOBBER, "s24", "s25", "s26", "s27"
            );
            // clang-format on
#pragma clang diagnostic pop

            // return local scratch
            auto c = make_tuple(MakeCBlockTile(), MakeCBlockTile());
            for(auto i = 0; i < 16; i++)
            {
                c.at(number<0>{}).get_thread_buffer()[4 * i + 0] = v_acc[i].x;
                c.at(number<0>{}).get_thread_buffer()[4 * i + 1] = v_acc[i].y;
                c.at(number<0>{}).get_thread_buffer()[4 * i + 2] = v_acc[i].z;
                c.at(number<0>{}).get_thread_buffer()[4 * i + 3] = v_acc[i].w;
            }
            for(auto i = 0; i < 16; i++)
            {
                c.at(number<1>{}).get_thread_buffer()[4 * i + 0] = v_acc[16 + i].x;
                c.at(number<1>{}).get_thread_buffer()[4 * i + 1] = v_acc[16 + i].y;
                c.at(number<1>{}).get_thread_buffer()[4 * i + 2] = v_acc[16 + i].z;
                c.at(number<1>{}).get_thread_buffer()[4 * i + 3] = v_acc[16 + i].w;
            }
            return c;
        }
        else
        {
            // this is the acc thread buffer
            fp32x4_t v_acc[16]{.0f};

            // B nr->kr
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winline-asm"
            // clang-format off
            asm volatile(
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_FP16
#include "uk/flatmm_uk_gfx9_32x512x128_1x1x1_16x16x16.inc"
                : _EXPAND_ASM_ARGS_OUT_ONE_ACC
                : _EXPAND_ASM_ARGS_IN
                : _EXPAND_ASM_ARGS_CLOBBER
            );
            // clang-format on
#pragma clang diagnostic pop

            // return local scratch
            auto c = MakeCBlockTile();
            for(auto i = 0; i < 16; i++)
            {
                c.get_thread_buffer()[4 * i + 0] = v_acc[i].x;
                c.get_thread_buffer()[4 * i + 1] = v_acc[i].y;
                c.get_thread_buffer()[4 * i + 2] = v_acc[i].z;
                c.get_thread_buffer()[4 * i + 3] = v_acc[i].w;
            }
            return c;
        }
    }
};
#undef _EXPAND_ASM_ARGS_OUT_ONE_ACC
#undef _EXPAND_ASM_ARGS_OUT_TWO_ACC
#undef _EXPAND_ASM_ARGS_IN
#undef _EXPAND_ASM_ARGS_CLOBBER
} // namespace ck_tile
