// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_trload_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineTrLoadDefaultPolicy>
struct BlockFmhaBwdDQDKDVPipelineTrLoadQRQTRDOR
{
    static constexpr auto is_qr_qtr_dor_pipeline = true;

    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType          = remove_cvref_t<typename Problem::GemmDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using DDataType             = remove_cvref_t<typename Problem::DDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType         = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType         = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType         = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType         = remove_cvref_t<typename Problem::VGradDataType>;
    using BiasGradDataType      = remove_cvref_t<typename Problem::BiasGradDataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using FmhaDropout           = remove_cvref_t<typename Problem::FmhaDropout>;
    // using HotLoopScheduler      = typename Policy::template HotLoopScheduler<Problem>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;

    static constexpr index_t kM0        = BlockFmhaShape::kM0;
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t kK0        = BlockFmhaShape::kK0;
    static constexpr index_t kK1        = BlockFmhaShape::kK1;
    static constexpr index_t kK2        = BlockFmhaShape::kK2;
    static constexpr index_t kK3        = BlockFmhaShape::kK3;
    static constexpr index_t kK4        = BlockFmhaShape::kK4;
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kVHeaddim  = BlockFmhaShape::kVHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr index_t kPadHeadDimQ  = Problem::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum         = Problem::BiasEnum;
    static constexpr bool kHasBiasGrad     = Problem::kHasBiasGrad;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;
    static constexpr bool kUseTrLoad       = Problem::kUseTrLoad;
    static_assert(kUseTrLoad, "This pipeline uses trload!");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentOGrad<Problem>();
    static constexpr index_t kAlignmentQGrad = 1;
    static constexpr index_t kAlignmentKGrad =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentKGrad<Problem>();
    static constexpr index_t kAlignmentVGrad =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentVGrad<Problem>();
    static constexpr index_t kAlignmentBias = 1;

    static constexpr const char* name = "trload_kr_ktr_vr";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static LSEDataType get_validated_lse(const LSEDataType raw_lse)
    {
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS || FmhaMask::IsMasking)
            return (raw_lse == -numeric<LSEDataType>::infinity()) //
                       ? type_convert<LSEDataType>(0.f)
                       : raw_lse;
        else
            return raw_lse;
    };

    template <typename... Ts>
    CK_TILE_DEVICE auto operator()(void* smem_ptr, Ts&&... args) const
    {
        // LDS allocation
        const auto smem_ptr_ =
            reinterpret_cast<char*>(smem_ptr); // cast to char* to do pointer arithmetic

        const auto k_lds_ptr = reinterpret_cast<KDataType* __restrict__>(smem_ptr_);
        const auto v_lds_ptr = reinterpret_cast<VDataType* __restrict__>(
            smem_ptr_ + Policy::template GetSmemSizeK<Problem>());

        const auto do_lds_ptr  = reinterpret_cast<OGradDataType*>(smem_ptr_);
        const auto q_lds_ptr   = reinterpret_cast<QDataType*>( //
            smem_ptr_ + Policy::template GetSmemSizeOGrad<Problem>());
        const auto lse_lds_ptr = reinterpret_cast<LSEDataType*>( //
            smem_ptr_ + Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeQ<Problem>());
        const auto d_lds_ptr   = reinterpret_cast<DDataType*>(
            smem_ptr_ + Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>());

        const auto ds_lds_ptr =
            reinterpret_cast<GemmDataType*>(smem_ptr_ + Policy::template GetSmemSizeK<Problem>() +
                                            Policy::template GetSmemSizeV<Problem>());
        const auto bias_lds_ptr = reinterpret_cast<BiasDataType*>(ds_lds_ptr);
        return run(k_lds_ptr,
                   v_lds_ptr,
                   do_lds_ptr,
                   q_lds_ptr,
                   lse_lds_ptr,
                   d_lds_ptr,
                   ds_lds_ptr,
                   bias_lds_ptr,
                   std::forward<Ts>(args)...);
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp,
              typename KGradDramBlockWindowTmp,
              typename VGradDramBlockWindowTmp,
              typename BiasGradDramBlockWindowTmp,
              typename QGradEpilogue,
              typename KGradEpilogue,
              typename VGradEpilogue,
              typename PositionEncoding>
    CK_TILE_DEVICE auto run( //
        KDataType* __restrict__ k_lds_ptr,
        VDataType* __restrict__ v_lds_ptr,
        OGradDataType* __restrict__ do_lds_ptr,
        QDataType* __restrict__ q_lds_ptr,
        LSEDataType* __restrict__ lse_lds_ptr,
        DDataType* __restrict__ d_lds_ptr,
        GemmDataType* __restrict__ ds_lds_ptr,
        BiasDataType* __restrict__ bias_lds_ptr,
        const QDramBlockWindowTmp& q_dram_block_window_tmp,
        const KDramBlockWindowTmp& k_dram_block_window_tmp,
        const VDramBlockWindowTmp& v_dram_block_window_tmp,
        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
        const RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
        const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
        const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
        const DDramBlockWindowTmp& d_dram_block_window_tmp,
        const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
        const KGradDramBlockWindowTmp& dk_dram_block_window_tmp,
        const VGradDramBlockWindowTmp& dv_dram_block_window_tmp,
        const BiasGradDramBlockWindowTmp& dbias_dram_block_window_tmp,
        const QGradEpilogue& dq_epilogue,
        const KGradEpilogue& dk_epilogue,
        const VGradEpilogue& dv_epilogue,
        FmhaMask mask,
        PositionEncoding position_encoding,
        float raw_scale,
        float scale,
        float rp_undrop,
        float scale_rp_undrop,
        FmhaDropout& dropout) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>> &&
                std::is_same_v<OGradDataType,
                               remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                std::is_same_v<LSEDataType,
                               remove_cvref_t<typename LSEDramBlockWindowTmp::DataType>> &&
                std::is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == OGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == LSEDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == DDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        // Early termination
        const auto [seqlen_kv_start, seqlen_kv_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_kv_end - seqlen_kv_start, kN0);

        // K, HBM ->LDS ->Reg
        auto k_dram_window =
            make_tile_window(Policy::template TransformXDramTensorView<KDataType>(
                                 k_dram_block_window_tmp.get_bottom_tensor_view()),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_kv_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem>());

        auto k_lds = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsWriteBlockDescriptor<Problem>());
        auto k_lds_write_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        //------------------------------------------------------------------
        // V, HBM ->LDS ->Reg
        auto v_dram_window =
            make_tile_window(Policy::template TransformXDramTensorView<VDataType>(
                                 v_dram_block_window_tmp.get_bottom_tensor_view()),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_kv_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());
        auto v_lds_write_window =
            make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kVHeaddim>{}), {0, 0});

        //------------------------------------------------------------------
        // KT, HBM -> LDS --trload-->Reg

        //------------------------------------------------------------------
        // Pre-Load KV into Registers
        auto k_lds_read = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsReadBlockDescriptor<Problem>());
        auto k_lds_read_window =
            make_tile_window(k_lds_read,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKRegBlockDescriptor<Problem>());

        auto kt_lds_read_window =
            make_tile_window(k_lds_read,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKTRegBlockDescriptor<Problem>());

        auto v_lds_read = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr, Policy::template MakeVLdsReadBlockDescriptor<Problem>());
        auto v_lds_read_window =
            make_tile_window(v_lds_read,
                             make_tuple(number<kN0>{}, number<kK2>{}),
                             v_lds_write_window.get_window_origin(),
                             Policy::template MakeVRegBlockDescriptor<Problem>());

        //---------------------------- Loop Load in ----------------------------//
        // Q: HBM -->LDS
        auto q_dram_window =
            make_tile_window(Policy::template TransformXDramTensorView<QDataType>(
                                 q_dram_block_window_tmp.get_bottom_tensor_view()),
                             q_dram_block_window_tmp.get_window_lengths(),
                             {0, 0},
                             Policy::template MakeQDramTileDistribution<Problem>());

        auto q_lds = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsWriteBlockDescriptor<Problem>());
        auto q_lds_write_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto q_lds_read = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsReadBlockDescriptor<Problem>());
        auto q_lds_read_window =
            make_tile_window(q_lds_read,
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_write_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());
        auto qt_lds_read_window =
            make_tile_window(q_lds_read,
                             make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                             {0, 0},
                             Policy::template MakeQTRegSliceBlockDescriptor<Problem>());

        // dO: HBM ->LDS ---load--> Reg
        // dOT:          \-loadtr-> Reg
        auto do_dram_window =
            make_tile_window(Policy::template TransformXDramTensorView<OGradDataType>(
                                 do_dram_block_window_tmp.get_bottom_tensor_view()),
                             do_dram_block_window_tmp.get_window_lengths(),
                             {0, 0},
                             Policy::template MakeOGradDramTileDistribution<Problem>());

        auto do_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsWriteBlockDescriptor<Problem>());
        auto do_lds_write_window =
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto do_lds_read = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsReadBlockDescriptor<Problem>());
        auto do_lds_read_window =
            make_tile_window(do_lds_read,
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             do_lds_write_window.get_window_origin(),
                             Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());
        auto dot_lds_read_window =
            make_tile_window(do_lds_read,
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             {0, 0},
                             Policy::template MakeOGradTRegSliceBlockDescriptor<Problem>());

        // dS: Reg -> Reg -> LDS
        auto ds_lds = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());

        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        // transform it to make it from col-major to row-major; prepared for load_tile_transpose
        auto ds_lds_t = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem, true>());
        auto ds_lds_read_window =
            make_tile_window(ds_lds_t,
                             make_tuple(number<kM0>{}, number<kK4>{}),
                             {0, 0},
                             Policy::template MakeSGradRegSliceBlockDescriptor<Problem>());

        // Bias: HBM ->Reg ->Reg ->LDS
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();

        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), seqlen_kv_start},
                             Policy::template MakeBiasTileDistribution<Problem>());

        auto bias_lds = make_tensor_view<address_space_enum::lds>(
            bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());
        auto bias_lds_write_window =
            make_tile_window(bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        auto bias_s_lds_read_window =
            make_tile_window(bias_lds_write_window.get_bottom_tensor_view(),
                             bias_lds_write_window.get_window_lengths(),
                             bias_lds_write_window.get_window_origin(),
                             Policy::template MakeBiasSTileDistribution<decltype(gemm_0)>());

        static_assert(std::is_same_v<BiasDataType, BiasGradDataType>,
                      "BiasDataType and BiasGradDataType should be the same!");

        // LSE: HBM -> LDS ->Reg
        auto lse_dram_window =
            make_tile_window(lse_dram_block_window_tmp.get_bottom_tensor_view(),
                             lse_dram_block_window_tmp.get_window_lengths(),
                             {0},
                             Policy::template MakeLSEDDramTileDistribution<Problem>());

        auto lse_lds = make_tensor_view<address_space_enum::lds>(
            lse_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto lse_lds_write_window = make_tile_window(lse_lds, make_tuple(number<kM0>{}), {0});

        auto lse_lds_read_window =
            make_tile_window(lse_lds,
                             make_tuple(number<kM0>{}),
                             {0},
                             Policy::template MakeLSEDLdsReadBlockDescriptor<Problem>());

        // D: HBM ->Reg
        auto d_dram_window =
            make_tile_window(d_dram_block_window_tmp.get_bottom_tensor_view(),
                             d_dram_block_window_tmp.get_window_lengths(),
                             {0},
                             Policy::template MakeLSEDDramTileDistribution<Problem>());

        auto d_lds = make_tensor_view<address_space_enum::lds>(
            d_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());
        auto d_lds_write_window = make_tile_window(d_lds, make_tuple(number<kM0>{}), {0});
        auto d_lds_read_window =
            make_tile_window(d_lds,
                             make_tuple(number<kM0>{}),
                             {0},
                             Policy::template MakeLSEDLdsReadBlockDescriptor<Problem>());

        // RandVal: HBM ->Reg
        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0), true>(
            randval_dram_block_window_tmp, seqlen_kv_start);

        // BiasGrad
        // Reg ->LDS ->Reg ->HBM
        const auto dbias_origin = dbias_dram_block_window_tmp.get_window_origin();

        auto dbias_dram_window =
            make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                             dbias_dram_block_window_tmp.get_window_lengths(),
                             {dbias_origin.at(number<0>{}), seqlen_kv_start}); // M/N

        auto dbias_lds_read_window =
            make_tile_window(bias_lds,
                             make_tuple(number<kM0>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeShuffledBiasTileDistribution<Problem>());

        // ----------------------------Loop write out------------------------------//
        auto dq_dram_window = make_tile_window(dq_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dq_dram_block_window_tmp.get_window_lengths(),
                                               {0, 0});
        auto dk_dram_window = make_tile_window(dk_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dk_dram_block_window_tmp.get_window_lengths(),
                                               {0, 0});
        auto dv_dram_window = make_tile_window(dv_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dv_dram_block_window_tmp.get_window_lengths(),
                                               {0, 0});

        index_t i_total_loops  = 0;
        index_t seqlen_kv_step = seqlen_kv_start;
        static_assert(kQKHeaddim >= kK0, "kQKHeaddim should be equal or greater than kK0");
        static_assert(kM0 == kK1, "kM0 should equal to kK1");
        static_assert(kVHeaddim >= kK2, "kVHeaddim should be equal or greater than kK2");
        static_assert(kM0 == kK3, "kM0 should equal to kK3");
        constexpr index_t k4_loops = kN0 / kK4;

        __builtin_amdgcn_sched_barrier(0);

        decltype(load_tile(q_lds_read_window)) q_reg_tensor;
        decltype(load_tile(lse_lds_read_window)) lse;
        decltype(load_tile_transpose(ds_lds_read_window)) ds_reg_tensor;
        decltype(load_tile_transpose(ds_lds_read_window)) ds_reg_tensor_next;
        decltype(load_tile(do_lds_read_window)) do_reg_tensor;
        decltype(load_tile_transpose(dot_lds_read_window)) dot_reg_tensor;
        decltype(load_tile(d_lds_read_window)) d;
        decltype(load_tile_transpose(qt_lds_read_window)) qt_reg_tensor;
        decltype(gemm_0.MakeCBlockTile()) s_acc, p;
        decltype(gemm_2.MakeCBlockTile()) dp_acc, ds;
        decltype(gemm_4.MakeCBlockTile()) dq_acc;
        clear_tile(dq_acc);

        decltype(load_tile(lse_dram_window)) lse_block_tile;
        decltype(load_tile(d_dram_window)) d_block_tile;

        async_load_tile(q_lds_write_window, q_dram_window);
        async_load_tile(do_lds_write_window, do_dram_window);
        __builtin_amdgcn_s_waitcnt(0);
        qt_reg_tensor  = load_tile_transpose(qt_lds_read_window);
        q_reg_tensor   = load_tile(q_lds_read_window);
        dot_reg_tensor = load_tile_transpose(dot_lds_read_window);
        do_reg_tensor  = load_tile(do_lds_read_window);

        lse_block_tile = load_tile(lse_dram_window);
        d_block_tile   = load_tile(d_dram_window);
        __builtin_amdgcn_s_waitcnt(0);
        store_tile(lse_lds_write_window, lse_block_tile);
        store_tile(d_lds_write_window, d_block_tile);
        __builtin_amdgcn_s_waitcnt(0);
        lse = load_tile(lse_lds_read_window);
        d   = load_tile(d_lds_read_window);

        auto main_body = [&](auto is_prologue_, auto is_epilogue_) mutable {
            constexpr bool is_prologue = is_prologue_.value;
            constexpr bool is_epilogue = is_epilogue_.value;
            static_assert(is_prologue || is_epilogue, "is_prologue or is_epilogue should be true");
            constexpr bool is_main_body = is_prologue && is_epilogue;

            // init VGrad & KGrad
            decltype(gemm_1.MakeCBlockTile()) dv_acc;
            decltype(gemm_3.MakeCBlockTile()) dk_acc;

            decltype(load_tile(k_lds_read_window)) k_reg_tensor;
            decltype(load_tile(v_lds_read_window)) v_reg_tensor;
            decltype(load_tile_transpose(kt_lds_read_window)) kt_reg_tensor;

            if constexpr(is_epilogue)
            {
                async_load_tile(k_lds_write_window, k_dram_window);
                move_tile_window(k_dram_window, {kN0, 0});
                async_load_tile(v_lds_write_window, v_dram_window);
                move_tile_window(v_dram_window, {kN0, 0});
                s_waitcnt</*vmcnt=*/0>();
                k_reg_tensor  = load_tile(k_lds_read_window);
                v_reg_tensor  = load_tile(v_lds_read_window);
                kt_reg_tensor = load_tile_transpose(kt_lds_read_window);
            }
            if constexpr(is_epilogue)
            {
                // STAGE 1, Q@K Gemm0
                s_acc = gemm_0(q_reg_tensor, k_reg_tensor);
            }
            if constexpr(is_main_body)
                Policy::template HotLoopScheduler<Problem>::SchedulerGemm0();
            __builtin_amdgcn_sched_barrier(0);
            if constexpr(is_epilogue)
            {
                // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    const auto bias_tile    = load_tile(bias_dram_window);
                    auto shuffled_bias_tile = make_static_distributed_tensor<BiasDataType>(
                        Policy::template MakeShuffledBiasTileDistribution<Problem>());
                    shuffle_tile(shuffled_bias_tile, bias_tile);
                    store_tile(bias_lds_write_window, shuffled_bias_tile);
                    block_sync_lds();
                    auto bias_s_tile = load_tile(bias_s_lds_read_window);
                    tile_elementwise_inout(
                        [&](auto& x, const auto& y) {
                            x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                        },
                        s_acc,
                        bias_s_tile);
                    move_tile_window(bias_dram_window, {kM0, 0});
                    __builtin_amdgcn_sched_barrier(0);
                }
                else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                    sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                            const auto tile_idx = get_x_indices_from_distributed_indices(
                                s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                            const auto row         = tile_idx.at(number<0>{});
                            const auto col         = seqlen_kv_step + tile_idx.at(number<1>{});
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);

                            s_acc(i_j_idx) *= scale;
                            position_encoding.update(s_acc(i_j_idx), row, col);
                        });
                    });
                }

                {
                    bool need_perpixel_check =
                        mask.IsEdgeTile(0, seqlen_kv_step, number<kM0>{}, number<kN0>{});
                    if(need_perpixel_check)
                    {
                        set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = tile_idx.at(number<0>{});
                            const auto col = seqlen_kv_step + tile_idx.at(number<1>{});
                            return mask.IsOutOfBound(row, col);
                        });
                    }
                }

                constexpr auto p_spans = decltype(p)::get_distributed_spans();
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                     BiasEnum == BlockAttentionBiasEnum::ALIBI)
                            p(i_j_idx) = exp2(s_acc[i_j_idx] - row_lse);
                        else
                            p(i_j_idx) = exp2(scale * s_acc[i_j_idx] - row_lse);
                    });
                });

                if constexpr(FmhaDropout::IsDropout)
                {
                    dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                        0, seqlen_kv_step, p, randval_dram_window);
                }
                const auto p_gemm = [&]() { // dropout / type conversion
                    if constexpr(FmhaDropout::IsDropout)
                    {
                        return tile_elementwise_in(
                            [](const auto& x) {
                                return type_convert<GemmDataType>(x > 0.f ? x : 0.f);
                            },
                            p);
                    }
                    else
                    {
                        return cast_tile<GemmDataType>(p);
                    }
                }();

                // STAGE 4, OGrad@V Gemm2
                dp_acc = gemm_2(do_reg_tensor, v_reg_tensor);

                // STAGE 3, P^T@OGrad^T Gemm1
                auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                    Policy::template MakePTRegSliceBlockDescriptor<Problem>());
                pt_reg_tensor.get_thread_buffer() = p_gemm.get_thread_buffer();

                dv_acc = gemm_1(pt_reg_tensor, dot_reg_tensor);
            }
            block_sync_lds();
            if constexpr(is_main_body)
                Policy::template HotLoopScheduler<Problem>::SchedulerGemm12();
            __builtin_amdgcn_sched_barrier(0);
            if constexpr(is_epilogue)
            {
                // STAGE 5, P^T(PGrad^T - D)
                constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
                sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        bool undrop_flag       = p[i_j_idx] >= 0;
                        ds(i_j_idx) = p[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                        ? (dp_acc[i_j_idx] - d[i_idx])
                                                        : d[i_idx]);
                    });
                });

                if constexpr(kHasBiasGrad)
                {
                    const auto dbias = [&]() {
                        if constexpr(FmhaDropout::IsDropout)
                        {
                            return tile_elementwise_in(
                                [&rp_undrop](const auto& x) {
                                    return type_convert<BiasGradDataType>(x * rp_undrop);
                                },
                                ds);
                        }
                        else
                        {
                            return cast_tile<BiasGradDataType>(ds);
                        }
                    }();
                    store_tile(bias_lds_write_window, dbias);
                    s_waitcnt</*vmcnt=*/0>();
                    block_sync_lds();
                    auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
                    auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                        Policy::template MakeBiasTileDistribution<Problem>());
                    shuffle_tile(dbias_tile, shuffled_dbias_tile);
                    store_tile(dbias_dram_window, dbias_tile);
                    move_tile_window(dbias_dram_window, {kM0, 0});
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
            if constexpr(is_epilogue)
            {
                // STAGE 6, SGrad^T@Q^T Gemm3
                const auto ds_gemm  = cast_tile<GemmDataType>(ds);
                auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                    Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
                dst_reg_tensor.get_thread_buffer() = ds_gemm.get_thread_buffer();
                dk_acc                             = gemm_3(dst_reg_tensor, qt_reg_tensor);

                if constexpr(kHasBiasGrad)
                {
                    // SGrad and BiasGrad use the same address in LDS, finish loading dbias to reuse
                    // LDS.
                    block_sync_lds();
                }
                store_tile(ds_lds_window, ds_gemm);
            }
            s_waitcnt</*vmcnt=*/0>();
            block_sync_lds();
            if constexpr(is_epilogue)
            {
                ds_reg_tensor = load_tile_transpose(ds_lds_read_window);
                move_tile_window(ds_lds_read_window, {kK4, 0});
            }
            if constexpr(is_main_body)
                Policy::template HotLoopScheduler<Problem>::SchedulerGemm3();
            __builtin_amdgcn_sched_barrier(0);
            if constexpr(is_epilogue)
            {
                // STAGE7 SGrad@K^T Gemm4
                static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                    if constexpr(i_k4 < k4_loops - 1)
                    {
                        ds_reg_tensor_next = load_tile_transpose(ds_lds_read_window);
                        move_tile_window(ds_lds_read_window, {kK4, 0});
                    }
                    auto kt_reg_tensor_slice = get_slice_tile( //
                        kt_reg_tensor,
                        sequence<0, i_k4 * kK4>{},
                        sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});
                    gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice);

                    if constexpr(i_k4 < k4_loops - 1)
                    {
                        ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
                    }
                });
                move_tile_window(ds_lds_read_window, {-kN0, 0});
            }
            block_sync_lds();
            if constexpr(is_main_body)
                Policy::template HotLoopScheduler<Problem>::SchedulerGemm4();
            if constexpr(is_epilogue)
            {
                // Results Scale
                if constexpr(FmhaDropout::IsDropout)
                {
                    tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                           dk_acc);
                    tile_elementwise_inout([&rp_undrop](auto& x) { x = x * rp_undrop; }, dv_acc);
                }
                else
                {
                    tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);
                }

                dk_epilogue(dk_dram_window, dk_acc, nullptr);
                move_tile_window(dk_dram_window, {kN0, 0});
                dv_epilogue(dv_dram_window, dv_acc, nullptr);
                move_tile_window(dv_dram_window, {kN0, 0});
            }
        };

        for(index_t i = 0; i < seqlen_kv_start; i += kN0)
        {
            dk_epilogue(dk_dram_window, decltype(gemm_3.MakeCBlockTile()){0}, nullptr);
            move_tile_window(dk_dram_window, {kN0, 0});
            dv_epilogue(dv_dram_window, decltype(gemm_1.MakeCBlockTile()){0}, nullptr);
            move_tile_window(dv_dram_window, {kN0, 0});
        }

        main_body(std::true_type{}, std::false_type{});
        // Hot loop
        if(num_total_loop > 1)
        {
            do
            {
                main_body(std::true_type{}, std::true_type{});
                i_total_loops += 1;
                seqlen_kv_step += kN0;
            } while(i_total_loops < num_total_loop - 1);
        }
        main_body(std::false_type{}, std::true_type{});
        seqlen_kv_step += kN0;

        const auto k_length         = k_dram_block_window_tmp.get_window_lengths();
        const auto seqlen_kv_length = k_length.at(number<0>{});
        for(; seqlen_kv_step < seqlen_kv_length; seqlen_kv_step += kN0)
        {
            dk_epilogue(dk_dram_window, decltype(gemm_3.MakeCBlockTile()){0}, nullptr);
            move_tile_window(dk_dram_window, {kN0, 0});
            dv_epilogue(dv_dram_window, decltype(gemm_1.MakeCBlockTile()){0}, nullptr);
            move_tile_window(dv_dram_window, {kN0, 0});
        }

        // QGrad Scale
        if constexpr(FmhaDropout::IsDropout)
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dq_acc);
        else
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
        dq_epilogue(dq_dram_window, dq_acc, nullptr);
        return;
    }
};

// We don't support C++20 concepts yet, so we use SFINAE check the existence and truthiness
// of is_qr_qtr_dor_pipeline static member instead of using concepts directly.
//
// The template struct's value field is equivalent to the following commented concept definition.
//
// template <class T>
// concept fmha_bwd_qr_qtr_dor_pipeline_c = T::is_qr_qtr_dor_pipeline;

// SFINAE test for existence and truthiness of static member is_qr_qtr_dor_pipeline.
template <typename, typename = void>
struct fmha_bwd_qr_qtr_dor_pipeline : std::false_type
{
};

template <typename T>
struct fmha_bwd_qr_qtr_dor_pipeline<T, std::void_t<decltype(T::is_qr_qtr_dor_pipeline)>>
    : std::bool_constant<T::is_qr_qtr_dor_pipeline>
{
};

} // namespace ck_tile
