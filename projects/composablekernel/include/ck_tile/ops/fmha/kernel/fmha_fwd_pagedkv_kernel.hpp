// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

// TODO: This class is a variant of the existing FmhaFwdSplitKVKernel pipeline.
//       Refactoring to extract shared logic is recommended as future work.
template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdPagedKVKernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FmhaPipeline::Problem::kBlockPerCu;

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    using VLayout = ck_tile::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kIsGroupMode      = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = FmhaPipeline::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = FmhaPipeline::BiasEnum;
    static constexpr bool kStoreLSE         = FmhaPipeline::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant = FmhaPipeline::Problem::kDoFp8StaticQuant;
    static constexpr bool kSkipMinSeqlenQ   = FmhaPipeline::Problem::kSkipMinSeqlenQ;
    static constexpr bool kIsPagedKV        = FmhaPipeline::Problem::kIsPagedKV;
    static constexpr bool kHasSink          = FmhaPipeline::kHasSink;

    using AttentionVariant = ck_tile::remove_cvref_t<typename FmhaPipeline::AttentionVariant>;
    using FmhaMask         = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    static constexpr bool kUseAsyncCopy = FmhaPipeline::Policy::AsyncCopy;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<ck_tile::fp8_t> { static constexpr const char * name = "fp8"; };
    template <> struct t2s<ck_tile::bf8_t> { static constexpr const char * name = "bf8"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        using bfs = typename FmhaPipeline::BlockFmhaShape;
        using g0br = typename bfs::Gemm0BlockWarps;
        using g1br = typename bfs::Gemm1BlockWarps;
        using g0wt = typename bfs::Gemm0WarpTile;
        using g1wt = typename bfs::Gemm1WarpTile;
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadSeqLenK) n += "sk";
            if (kPadHeadDimQ) n += "d";
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_fwd_pagedkv_d") + _TS_(bfs::kQKHeaddim) + "_" + _SS_(t2s<QDataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_"
            "b" + _TS_(bfs::kM0) + "x" + _TS_(bfs::kN0) + "x" + _TS_(bfs::kK0) + "x" +
                    _TS_(bfs::kN1) + "x" + _TS_(bfs::kK1) + "x" + _TS_(bfs::kQKHeaddim) + "_" +
            "r" + _TS_(g0br::at(ck_tile::number<0>{})) + "x" + _TS_(g0br::at(ck_tile::number<1>{})) + "x" + _TS_(g0br::at(ck_tile::number<2>{})) + "_" +
            "r" + _TS_(g1br::at(ck_tile::number<0>{})) + "x" + _TS_(g1br::at(ck_tile::number<1>{})) + "x" + _TS_(g1br::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(g0wt::at(ck_tile::number<0>{})) + "x" + _TS_(g0wt::at(ck_tile::number<1>{})) + "x" + _TS_(g0wt::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(g1wt::at(ck_tile::number<0>{})) + "x" + _TS_(g1wt::at(ck_tile::number<1>{})) + "x" + _TS_(g1wt::at(ck_tile::number<2>{})) + "_" +
            (kBlockPerCuInput == -1 ? "" : ("o" + _TS_(kBlockPerCu) + "_")) + _SS_(FmhaPipeline::name) + "_" +
            "v" + (std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor> ? "r" : "c") + (pn.empty() ? "_npad" : "_" + pn) +
            (kHasLogitsSoftCap ? "_logits" : "_nlogits" ) + (BiasEnum == BlockAttentionBiasEnum::NO_BIAS ? _SS_("_nbias") : (_SS_("_") + BlockAttentionBiasEnumToStr<BiasEnum>::name)) +
            (kHasMask ? "_" + _SS_(FmhaMask::name) : "_nmask") + (kStoreLSE ? "_lse" : "_nlse" )  + (kSkipMinSeqlenQ ? "_skip" : "_nskip" )  + (kDoFp8StaticQuant ? "_squant" : "_nsquant" ) + (kIsPagedKV ? "_pagedkv" : "_npagedkv" ) + (kHasSink ? "_sink" : "_nsink" );
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct FmhaFwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_ratio_qk;
        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o;
    };

    struct FmhaFwdLogitsSoftCapKargs
    {
        FmhaFwdLogitsSoftCapKargs() = default;

        void init_logits_soft_cap(float logits_soft_cap_)
        {
            if(0 < logits_soft_cap_)
            {
                logits_soft_cap     = logits_soft_cap_;
                logits_soft_cap_rcp = 1.f / logits_soft_cap;
            }
            else
            {
                logits_soft_cap     = 0.f;
                logits_soft_cap_rcp = 0.f;
            }
        }

        float logits_soft_cap;
        float logits_soft_cap_rcp;
    };

    struct FmhaFwdCommonBiasKargs
    {
        const void* bias_ptr               = nullptr;
        ck_tile::index_t stride_bias       = 0;
        ck_tile::index_t nhead_stride_bias = 0;
    };

    struct FmhaFwdBatchModeBiasKargs : FmhaFwdCommonBiasKargs
    {
        ck_tile::index_t batch_stride_bias = 0;
    };

    struct FmhaFwdAlibiKargs
    {
        // alibi is batch*nhead*1, no matter in batch/group mode, they are the same
        const void* alibi_slope_ptr;
        ck_tile::index_t alibi_slope_stride; // stride in batch, or 0 for all batch share same slope
    };

    struct FmhaFwdMaskKargs
    {
        // ck_tile::index_t window_size_left, window_size_right;
        ck_tile::index_t window_size_left, window_size_right, sink_size;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct FmhaFwdFp8StaticQuantKargs
    {
        float scale_p;
        float scale_o;
    };

    struct FmhaFwdCommonLSEKargs
    {
        void* lse_ptr                     = nullptr;
        ck_tile::index_t nhead_stride_lse = 0;
        ck_tile::index_t batch_stride_lse = 0;
    };

    struct FmhaFwdSkipMinSeqlenQKargs
    {
        ck_tile::index_t min_seqlen_q = 0;
    };

    struct CommonPageBlockTableKargs
    {
        const int32_t* block_table_ptr;
        ck_tile::index_t batch_stride_block_table;
        ck_tile::index_t page_block_size;
    };

    struct GroupModePageBlockTableKargs : CommonPageBlockTableKargs
    {
        bool is_gappy = false;
    };

    struct CacheBatchIdxKargs
    {
        const int32_t* cache_batch_idx;
    };

    struct FmhaFwdBatchModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaFwdBatchModeBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaFwdAlibiKargs,
                                                FmhaFwdEmptyKargs<0>>>,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kDoFp8StaticQuant, FmhaFwdFp8StaticQuantKargs, FmhaFwdEmptyKargs<3>>,
          std::conditional_t<kIsPagedKV, CommonPageBlockTableKargs, CacheBatchIdxKargs>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<4>>
    {
        const int32_t* seqlen_k_ptr;

        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k; // when using paged-kvcache, this will be stride/size for
                                         // single kcache page-block
        ck_tile::index_t batch_stride_v; // when using paged-kvcache, this will be stride/size for
                                         // single vcache page-block
        ck_tile::index_t batch_stride_o;
    };

    struct FmhaFwdGroupModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaFwdCommonBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaFwdAlibiKargs,
                                                FmhaFwdEmptyKargs<0>>>,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kDoFp8StaticQuant, FmhaFwdFp8StaticQuantKargs, FmhaFwdEmptyKargs<3>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<4>>,
          std::conditional_t<kIsPagedKV, GroupModePageBlockTableKargs, FmhaFwdEmptyKargs<5>>,
          std::conditional_t<kSkipMinSeqlenQ, FmhaFwdSkipMinSeqlenQKargs, FmhaFwdEmptyKargs<6>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;

        ck_tile::index_t batch_stride_k; // only used for paged-kvcache, this will be stride/size
                                         // for single kcache page-block
        ck_tile::index_t batch_stride_v; // only used for paged-kvcache, this will be stride/size
                                         // for single vcache page-block
    };

    using Kargs = std::conditional_t<kIsGroupMode, FmhaFwdGroupModeKargs, FmhaFwdBatchModeKargs>;

    struct BlockIndices
    {
        ck_tile::index_t batch_idx;
        ck_tile::index_t qo_head_idx;
        ck_tile::index_t kv_head_idx;
    };

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  void* lse_ptr,
                  void* o_ptr,
                  ck_tile::index_t seqlen_q,
                  ck_tile::index_t seqlen_k,
                  const void* seqlen_k_ptr, // only used for (paged-) kvcache
                  ck_tile::index_t hdim_q,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t num_head_q,
                  ck_tile::index_t nhead_ratio_qk,
                  const void* block_table_ptr,
                  ck_tile::index_t batch_stride_block_table,
                  ck_tile::index_t page_block_size,
                  const void* cache_batch_idx,
                  float scale_s,
                  float scale_p,
                  float scale_o,
                  float logits_soft_cap,
                  ck_tile::index_t stride_q,
                  ck_tile::index_t stride_k,
                  ck_tile::index_t stride_v,
                  ck_tile::index_t stride_bias,
                  ck_tile::index_t stride_o,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_lse,
                  ck_tile::index_t nhead_stride_o,
                  ck_tile::index_t batch_stride_q,
                  ck_tile::index_t batch_stride_k,
                  ck_tile::index_t batch_stride_v,
                  ck_tile::index_t batch_stride_bias,
                  ck_tile::index_t batch_stride_lse,
                  ck_tile::index_t batch_stride_o,
                  ck_tile::index_t window_size_left,
                  ck_tile::index_t window_size_right,
                  ck_tile::index_t sink_size,
                  ck_tile::index_t mask_type)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for bias
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for fp8_static_quant args
                    {},               // placeholder for pagedkv
                    {},               // placeholder for logits_soft_cap
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr),
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.sink_size         = sink_size;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
            kargs.batch_stride_lse = batch_stride_lse;
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_p = scale_p;
            kargs.scale_o = scale_o;
        }
        if constexpr(kIsPagedKV)
        {
            kargs.block_table_ptr          = reinterpret_cast<const int32_t*>(block_table_ptr);
            kargs.batch_stride_block_table = batch_stride_block_table;
            kargs.page_block_size          = page_block_size;
        }
        else
        {
            kargs.cache_batch_idx = reinterpret_cast<const int32_t*>(cache_batch_idx);
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }

        return kargs;
    }

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              const void* seqlen_k_ptr, // only used for (paged-) kvcache
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              const void* block_table_ptr,
              ck_tile::index_t batch_stride_block_table,
              ck_tile::index_t page_block_size,
              const void* cache_batch_idx,
              float scale_s,
              float scale_p,
              float scale_o,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_bias,
              ck_tile::index_t batch_stride_lse,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t sink_size,
              ck_tile::index_t mask_type)
    {
        return MakeKargsImpl(q_ptr,
                             k_ptr,
                             v_ptr,
                             bias_ptr,
                             lse_ptr,
                             o_ptr,
                             seqlen_q,
                             seqlen_k,
                             seqlen_k_ptr,
                             hdim_q,
                             hdim_v,
                             num_head_q,
                             nhead_ratio_qk,
                             block_table_ptr,
                             batch_stride_block_table,
                             page_block_size,
                             cache_batch_idx,
                             scale_s,
                             scale_p,
                             scale_o,
                             logits_soft_cap,
                             stride_q,
                             stride_k,
                             stride_v,
                             stride_bias,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_lse,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_lse,
                             batch_stride_o,
                             window_size_left,
                             window_size_right,
                             sink_size,
                             mask_type);
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  void* lse_ptr,
                  void* o_ptr,
                  const void* seqstart_q_ptr,
                  const void* seqstart_k_ptr,
                  const void* seqlen_k_ptr,
                  ck_tile::index_t hdim_q,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t num_head_q,
                  ck_tile::index_t nhead_ratio_qk,
                  const void* block_table_ptr,
                  ck_tile::index_t batch_stride_block_table,
                  ck_tile::index_t page_block_size,
                  bool is_gappy,
                  float scale_s,
                  float scale_p,
                  float scale_o,
                  float logits_soft_cap,
                  ck_tile::index_t stride_q,
                  ck_tile::index_t stride_k,
                  ck_tile::index_t stride_v,
                  ck_tile::index_t stride_bias,
                  ck_tile::index_t stride_o,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_lse,
                  ck_tile::index_t nhead_stride_o,
                  ck_tile::index_t batch_stride_k, // only used for paged-kvcache
                  ck_tile::index_t batch_stride_v, // only used for paged-kvcache
                  ck_tile::index_t window_size_left,
                  ck_tile::index_t window_size_right,
                  ck_tile::index_t sink_size,
                  ck_tile::index_t mask_type,
                  ck_tile::index_t min_seqlen_q)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for bias
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for fp8_static_quant args
                    {},               // placeholder for logits_soft_cap
                    {},               // placeholder for pagdkv
                    {},               // placeholder for min_seqlen_q
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr),
                    batch_stride_k,
                    batch_stride_v};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.sink_size         = sink_size;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_p = scale_p;
            kargs.scale_o = scale_o;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }
        if constexpr(kIsPagedKV)
        {
            kargs.block_table_ptr          = reinterpret_cast<const int32_t*>(block_table_ptr);
            kargs.batch_stride_block_table = batch_stride_block_table;
            kargs.page_block_size          = page_block_size;
            kargs.is_gappy                 = is_gappy;
        }
        if constexpr(kSkipMinSeqlenQ)
        {
            kargs.min_seqlen_q = min_seqlen_q;
        }

        return kargs;
    }

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* lse_ptr,
              void* o_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_k_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              const void* block_table_ptr,
              ck_tile::index_t batch_stride_block_table,
              ck_tile::index_t page_block_size,
              bool is_gappy,
              float scale_s,
              float scale_p,
              float scale_o,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_k, // only used for paged-kvcache
              ck_tile::index_t batch_stride_v, // only used for paged-kvcache
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t sink_size,
              ck_tile::index_t mask_type,
              ck_tile::index_t min_seqlen_q)
    {
        return MakeKargsImpl(q_ptr,
                             k_ptr,
                             v_ptr,
                             bias_ptr,
                             lse_ptr,
                             o_ptr,
                             seqstart_q_ptr,
                             seqstart_k_ptr,
                             seqlen_k_ptr,
                             hdim_q,
                             hdim_v,
                             num_head_q,
                             nhead_ratio_qk,
                             block_table_ptr,
                             batch_stride_block_table,
                             page_block_size,
                             is_gappy,
                             scale_s,
                             scale_p,
                             scale_o,
                             logits_soft_cap,
                             stride_q,
                             stride_k,
                             stride_v,
                             stride_bias,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_lse,
                             nhead_stride_o,
                             batch_stride_k,
                             batch_stride_v,
                             window_size_left,
                             window_size_right,
                             sink_size,
                             mask_type,
                             min_seqlen_q);
    }

    CK_TILE_HOST static void PrintParameters(const Kargs& kargs, int num_batches)
    {
        static bool dummy = [&]() {
            std::cout << std::endl;

            std::cout << " q_ptr: " << kargs.q_ptr << " k_ptr:" << kargs.k_ptr
                      << " v_ptr: " << kargs.v_ptr << " o_ptr:" << kargs.o_ptr
                      << " hdim_q: " << kargs.hdim_q << " hdim_v: " << kargs.hdim_v
                      << " num_head_q:" << kargs.num_head_q
                      << " nhead_ratio_qk: " << kargs.nhead_ratio_qk << " scale_s:" << kargs.scale_s
                      << " stride_q:" << kargs.stride_q << " stride_k:" << kargs.stride_k
                      << " stride_v:" << kargs.stride_v << " stride_o:" << kargs.stride_o
                      << " nhead_stride_q: " << kargs.nhead_stride_q
                      << " nhead_stride_k: " << kargs.nhead_stride_k
                      << " nhead_stride_v:" << kargs.nhead_stride_v
                      << " nhead_stride_o: " << kargs.nhead_stride_o;
            if constexpr(!kIsGroupMode)
            {
                std::cout << " batch_stride_q:" << kargs.batch_stride_q;
            }
            std::cout << " batch_stride_k:" << kargs.batch_stride_k
                      << " batch_stride_v:" << kargs.batch_stride_v;

            if constexpr(kIsGroupMode)
            {
                if constexpr(kSkipMinSeqlenQ)
                {
                    std::cout << " min_seqlen_q: " << kargs.min_seqlen_q;
                }

                std::cout << " seqstart_q_ptr:" << kargs.seqstart_q_ptr
                          << " seqstart_k_ptr: " << kargs.seqstart_k_ptr
                          << " seqlen_k_ptr:" << kargs.seqlen_k_ptr;
                if(kargs.seqlen_k_ptr != nullptr)
                {
                    std::cout << "{";
                    for(int i_batch = 0; i_batch < num_batches; i_batch++)
                        std::cout << kargs.seqlen_k_ptr[i_batch] << ",";
                    std::cout << "}";
                }
            }
            if constexpr(kHasMask)
            {
                std::cout << " window_size_left: " << kargs.window_size_left
                          << " window_size_right:" << kargs.window_size_right
                          << " mask_type: " << static_cast<int>(kargs.mask_type);
            }

            if constexpr(kIsPagedKV)
            {
                std::cout << " block_table_ptr: " << kargs.block_table_ptr
                          << " batch_stride_block_table:" << kargs.batch_stride_block_table
                          << " page_block_size: " << kargs.page_block_size;

                std::cout << "table value: [";
                for(int b = 0; b < num_batches; b++)
                {
                    std::cout << "[ ";
                    for(int i = 0; i < kargs.batch_stride_block_table; i++)
                    {
                        std::cout << kargs.block_table_ptr[b * kargs.batch_stride_block_table + i]
                                  << ",";
                    }
                    std::cout << " ]";
                }
                std::cout << " ]";
            }
            std::cout << std::endl;
            return true;
        }();
        (void)dummy;
    }
    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_,
                                                bool has_padded_seqlen_k)
    {
        // has_padded_seqlen_k is determined by checking (seqlen_k_ptr != nullptr)
        if(has_padded_seqlen_k)
        {
            // TODO: this may need tuning
            return dim3(nhead_,
                        batch_size_,
                        ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1));
        }
        else
        {
            // TODO: this may need tuning
            return dim3(ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1),
                        nhead_,
                        batch_size_);
        }
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        bool has_padded_seqlen_k = false;

        if constexpr(kIsGroupMode)
            has_padded_seqlen_k = (kargs.seqlen_k_ptr != nullptr);

        if(has_padded_seqlen_k)
        {
            // const index_t num_tile_m0 = seqlen_q / kM0;
            const index_t num_tile_n1 =
                ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1);

            const index_t i_block = blockIdx.z;
            const index_t i_nhead = blockIdx.x;
            const index_t i_batch = blockIdx.y;

            const auto f = [](index_t dividend, index_t divisor) {
                index_t quotient = dividend / divisor;
                index_t modulus  = dividend - quotient * divisor;
                return ck_tile::make_tuple(quotient, modulus);
            };

            const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

            if constexpr(kHasMask)
            {
                // assume that num_tile_n1 is always 1
                return ck_tile::make_tuple(gridDim.z - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
            }
            else
            {
                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
            }
        }
        else
        {
            // const index_t num_tile_m0 = seqlen_q / kM0;
            const index_t num_tile_n1 =
                ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1);

            const index_t i_block = blockIdx.x;
            const index_t i_nhead = blockIdx.y;
            const index_t i_batch = blockIdx.z;

            const auto f = [](index_t dividend, index_t divisor) {
                index_t quotient = dividend / divisor;
                index_t modulus  = dividend - quotient * divisor;
                return ck_tile::make_tuple(quotient, modulus);
            };

            const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

            if constexpr(kHasMask)
            {
                // assume that num_tile_n1 is always 1
                return ck_tile::make_tuple(gridDim.x - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
            }
            else
            {
                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
            }
        }
    }

    CK_TILE_HOST static dim3 BlockSize()
    {
        if(is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);

        const index_t i_m0 = amd_wave_read_first_lane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = amd_wave_read_first_lane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q    = 0;
        long_index_t batch_offset_k    = 0;
        long_index_t batch_offset_v    = 0;
        long_index_t batch_offset_bias = 0;
        long_index_t batch_offset_lse  = 0;
        long_index_t batch_offset_o    = 0;
        index_t kv_l2p_offset          = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
            batch_offset_k = key_start * kargs.stride_k;
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                batch_offset_v = key_start * kargs.stride_v;
            }
            else
            {
                batch_offset_v = key_start;
            }
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = query_start * kargs.stride_bias;
            }
            if constexpr(kStoreLSE)
            {
                batch_offset_lse = query_start;
            }

            batch_offset_o = query_start * kargs.stride_o;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];

            if constexpr(kSkipMinSeqlenQ)
            {
                if(kargs.seqlen_q <= kargs.min_seqlen_q)
                {
                    return;
                }
            }

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }

            if constexpr(kIsPagedKV)
            {
                if(kargs.is_gappy)
                {
                    // seqstart_k_ptr has different meaning in this case
                    kv_l2p_offset = kargs.seqstart_k_ptr[i_batch];
                }
            }
        }
        else
        {
            const index_t i_cache_batch = [&, i_batch_ = i_batch] {
                if constexpr(kIsPagedKV)
                {
                    return i_batch_;
                }
                else
                {
                    return (kargs.cache_batch_idx != nullptr ? kargs.cache_batch_idx[i_batch_]
                                                             : i_batch_);
                }
            }();

            batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k = static_cast<long_index_t>(i_cache_batch) * kargs.batch_stride_k;
            batch_offset_v = static_cast<long_index_t>(i_cache_batch) * kargs.batch_stride_v;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
            }
            if constexpr(kStoreLSE)
            {
                batch_offset_lse = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
            }

            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
        }

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                number<FmhaPipeline::kAlignmentQ>{},
                number<1>{});
            if constexpr(FmhaPipeline::kQLoadOnce)
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
            else
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
        }();

        const auto make_k_dram = [&](const KDataType* data, index_t height) {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                data, // will update this pointer if using paged-kvcache
                make_tuple(height, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<false, kPadHeadDimQ>{});
        };
        const auto k_dram = [&]() {
            if constexpr(kIsPagedKV)
            {
                return make_k_dram(nullptr, kargs.page_block_size);
            }
            else
            {
                return make_k_dram(k_ptr, kargs.seqlen_k);
            }
        }();

        const auto make_v_dram = [&](const VDataType* data, index_t length) {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    data, // will update this pointer if using paged-kvcache
                    make_tuple(length, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                const auto v_dram_transposed =
                    transform_tensor_view(v_dram_naive,
                                          make_tuple(make_pass_through_transform(kargs.hdim_v),
                                                     make_pass_through_transform(length)),
                                          make_tuple(sequence<1>{}, sequence<0>{}),
                                          make_tuple(sequence<0>{}, sequence<1>{}));

                return pad_tensor_view(
                    v_dram_transposed,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    data, // will update this pointer if using paged-kvcache
                    make_tuple(kargs.hdim_v, length),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<false, kPadSeqLenK>{});
            }
        };
        const auto v_dram = [&]() {
            if constexpr(kIsPagedKV)
            {
                return make_v_dram(nullptr, kargs.page_block_size);
            }
            else
            {
                return make_v_dram(v_ptr, kargs.seqlen_k);
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(number<FmhaPipeline::kM0>{},
                                      number<FmhaPipeline::kSubQKHeaddim>{});
                else
                    return make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{});
            }(),
            {i_m0, 0});

        auto k_page_block_navigator =
            [&, i_batch_ = i_batch, i_nhead_ = i_nhead / kargs.nhead_ratio_qk]() {
                if constexpr(kIsPagedKV)
                {
                    const auto* block_indices =
                        reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                        i_batch_ * kargs.batch_stride_block_table;
                    const index_t num_blocks =
                        integer_divide_ceil(kv_l2p_offset + kargs.seqlen_k, kargs.page_block_size);

                    const long_index_t fixed_offset =
                        static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_k;

                    return make_page_block_navigator<const KDataType, 0>(
                        kargs.k_ptr,
                        kargs.batch_stride_k, // kcache page-block stride/size
                        fixed_offset,
                        block_indices,
                        num_blocks,
                        kargs.page_block_size,
                        k_dram,
                        make_k_dram(nullptr,
                                    (kv_l2p_offset + kargs.seqlen_k) -
                                        (num_blocks - 1) * kargs.page_block_size));
                }
                else
                {
                    return make_page_block_navigator(k_dram);
                }
            }();

        auto v_page_block_navigator =
            [&, i_batch_ = i_batch, i_nhead_ = i_nhead / kargs.nhead_ratio_qk]() {
                if constexpr(kIsPagedKV)
                {
                    const auto* block_indices =
                        reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                        i_batch_ * kargs.batch_stride_block_table;
                    const index_t num_blocks =
                        integer_divide_ceil(kv_l2p_offset + kargs.seqlen_k, kargs.page_block_size);

                    const long_index_t fixed_offset =
                        static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_v;

                    return make_page_block_navigator<const VDataType, 1>(
                        kargs.v_ptr,
                        kargs.batch_stride_v, // vcache page-block stride/size
                        fixed_offset,
                        block_indices,
                        num_blocks,
                        kargs.page_block_size,
                        v_dram,
                        make_v_dram(nullptr,
                                    (kv_l2p_offset + kargs.seqlen_k) -
                                        (num_blocks - 1) * kargs.page_block_size));
                }
                else
                {
                    return make_page_block_navigator(v_dram);
                }
            }();

        auto k_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{});
        auto v_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{});

        /// FIXME: Before C++20, capturing structured binding variables are not supported. Remove
        /// following copy capture of the 'i_nhead' if in C++20
        const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto bias_dram_window_lengths =
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const BiasDataType* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                    batch_offset_bias;

                const auto bias_dram = [&]() {
                    const auto bias_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        bias_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_bias, 1),
                        number<FmhaPipeline::kAlignmentBias>{},
                        number<1>{});

                    return pad_tensor_view(bias_dram_naive,
                                           bias_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        // lse
        auto lse_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto lse_dram_window_lengths = make_tuple(number<FmhaPipeline::kM0>{});
            if constexpr(kStoreLSE)
            {
                LSEDataType* lse_ptr =
                    reinterpret_cast<LSEDataType*>(kargs.lse_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_lse + batch_offset_lse;

                const auto lse_dram = [&]() {
                    const auto lse_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        lse_ptr,
                        make_tuple(kargs.seqlen_q),
                        make_tuple(1),
                        number<1>{},
                        number<1>{});

                    return pad_tensor_view(
                        lse_dram_naive, lse_dram_window_lengths, sequence<kPadSeqLenQ>{});
                }();

                return make_tile_window(lse_dram, lse_dram_window_lengths, {i_m0});
            }
            else
            {
                return make_null_tile_window(lse_dram_window_lengths);
            }
        }();

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.sink_size,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        // WA i_batch capture structure binding before c++20
        auto position_encoding = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                // data loading, shared by entire wg
                // TODO: how to use s_read?
                SaccDataType slope =
                    *(reinterpret_cast<const SaccDataType*>(kargs.alibi_slope_ptr) +
                      i_batch_ * kargs.alibi_slope_stride + i_nhead_);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                slope *= ck_tile::log2e_v<>;
#endif
                if constexpr(kHasMask)
                {
                    return make_alibi_from_lr_mask<SaccDataType, true>(slope,
                                                                       kargs.window_size_left,
                                                                       kargs.window_size_right,
                                                                       kargs.seqlen_q,
                                                                       kargs.seqlen_k,
                                                                       kargs.mask_type);
                }
                else
                {
                    return Alibi<SaccDataType, true>{
                        slope, kargs.seqlen_q, kargs.seqlen_k, AlibiMode::FROM_BOTTOM_RIGHT};
                }
            }
            else
            {
                return EmptyPositionEncoding<SaccDataType>{};
            }
        }();

        AttentionVariant variant;
        const auto variant_params = [&] {
            if constexpr(kHasLogitsSoftCap)
            {
                return ck_tile::LogitsSoftCapParams<FmhaMask, CK_TILE_FMHA_FWD_FAST_EXP2>{
                    mask, kargs.scale_s, kargs.logits_soft_cap, kargs.logits_soft_cap_rcp};
            }
            else
            {
                return ck_tile::StandardAttentionParams<FmhaMask>{mask, kargs.scale_s};
            }
        }();

        BlockIndices block_indices{i_batch, i_nhead, i_nhead / kargs.nhead_ratio_qk};

        auto o_acc_tile = [&]() {
            if constexpr(kDoFp8StaticQuant)
            {
                return FmhaPipeline{}(
                    q_dram_window,
                    identity{}, // q_element_func
                    k_dram_window_lengths,
                    k_page_block_navigator,
                    identity{}, // k_element_func
                    v_dram_window_lengths,
                    v_page_block_navigator,
                    identity{}, // v_element_func
                    bias_dram_window,
                    identity{}, // bias_element_func
                    lse_dram_window,
                    identity{},                                          // lse_element_func
                    identity{},                                          // s_acc_element_func
                    scales{kargs.scale_p},                               // p_compute_element_func
                    composes(saturates<fp8_t>{}, scales{kargs.scale_o}), // o_acc_element_func
                    mask,
                    position_encoding,
                    kargs.scale_s,
                    variant,
                    variant_params,
                    block_indices,
                    kv_l2p_offset,
                    smem_ptr);
            }
            else
            {
                return FmhaPipeline{}(q_dram_window,
                                      k_dram_window_lengths,
                                      k_page_block_navigator,
                                      v_dram_window_lengths,
                                      v_page_block_navigator,
                                      bias_dram_window,
                                      lse_dram_window,
                                      mask,
                                      position_encoding,
                                      kargs.scale_s,
                                      variant,
                                      variant_params,
                                      block_indices,
                                      kv_l2p_offset,
                                      smem_ptr);
            }
        }();

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<FmhaPipeline::kAlignmentO>{},
                number<1>{});
            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};

} // namespace ck_tile
