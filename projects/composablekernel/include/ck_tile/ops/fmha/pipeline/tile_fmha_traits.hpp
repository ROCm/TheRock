// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_rotary_embedding.hpp"

namespace ck_tile {

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          bool kHasLogitsSoftCap_,
          BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          bool kStoreLSE_,
          bool kHasDropout_,
          BlockAttentionQuantScaleEnum QScaleEnum_,
          index_t kBlockPerCu_  = -1,    /* overwrite occupancy if not -1 */
          bool kSkipMinSeqlenQ_ = false, /* skip min seqlen q while chunked prefill */
          bool kHasSink_        = false>
struct TileFmhaTraits
{
    static constexpr bool kPadSeqLenQ       = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK       = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ      = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV      = kPadHeadDimV_;
    static constexpr bool kHasLogitsSoftCap = kHasLogitsSoftCap_;
    static constexpr auto BiasEnum          = BiasEnum_;
    static constexpr bool kHasBiasGrad      = kHasBiasGrad_;
    static constexpr bool kStoreLSE         = kStoreLSE_;
    static constexpr bool kHasDropout       = kHasDropout_;
    static constexpr auto QScaleEnum        = QScaleEnum_;
    static constexpr index_t kBlockPerCu    = kBlockPerCu_;
    static constexpr bool kSkipMinSeqlenQ   = kSkipMinSeqlenQ_;
    static constexpr bool kHasSink          = kHasSink_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* padding for hdim_q */,
          bool kPadHeadDimV_ /* padding for hdim_v */,
          bool kHasLogitsSoftCap_,
          BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          bool kStoreLSE_,
          bool kHasDropout_,
          BlockAttentionQuantScaleEnum QScaleEnum_,
          index_t kBlockPerCu_    = -1,    /* overwrite occupancy if not -1 */
          bool kSkipMinSeqlenQ_   = false, /* skip min seqlen q while chunked prefill */
          index_t kPageBlockSize_ = 1,
          BlockAttentionKVCacheMemoryLayoutEnum kKVMemoryLayout_ =
              BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT,
          BlockAttentionKVCacheLookupTableEnum kKVLookupTable_ =
              BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D>
struct TileFmhaBatchPrefillTraits : public TileFmhaTraits<kPadSeqLenQ_,
                                                          kPadSeqLenK_,
                                                          kPadHeadDimQ_,
                                                          kPadHeadDimV_,
                                                          kHasLogitsSoftCap_,
                                                          BiasEnum_,
                                                          kHasBiasGrad_,
                                                          kStoreLSE_,
                                                          kHasDropout_,
                                                          QScaleEnum_,
                                                          kBlockPerCu_,
                                                          kSkipMinSeqlenQ_,
                                                          false>
{
    static constexpr auto kKVMemoryLayout   = kKVMemoryLayout_;
    static constexpr auto kKVLookupTable    = kKVLookupTable_;
    static constexpr index_t kPageBlockSize = kPageBlockSize_;
    static_assert(kKVMemoryLayout == BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT ||
                      kKVMemoryLayout == BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT,
                  "Batch prefill only supports vectorized or linear KV cache layout.");
    static_assert(kPageBlockSize > 0 && ((kPageBlockSize & (kPageBlockSize - 1)) == 0),
                  "kPageBlockSize should be a power of 2 to support efficient page-based KV cache "
                  "addressing.");
};

template <index_t kPadHeadDimQ_ /* paddding for hdim_q */,
          index_t kPadHeadDimV_ /* paddding for hdim_v */,
          BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaBwdTraits
{
    static constexpr index_t kPadHeadDimQ = kPadHeadDimQ_;
    static constexpr index_t kPadHeadDimV = kPadHeadDimV_;
    static constexpr auto BiasEnum        = BiasEnum_;
    static constexpr bool kHasBiasGrad    = kHasBiasGrad_;
    static constexpr index_t kBlockPerCu  = kBlockPerCu_;

    static_assert(kPadHeadDimQ == 0 || kPadHeadDimQ == 8 || kPadHeadDimQ == 1);
    static_assert(kPadHeadDimV == 0 || kPadHeadDimV == 8 || kPadHeadDimV == 1);
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          bool kHasLogitsSoftCap_,
          BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          bool kStoreLSE_, /* set to true if either num_splits > 1 or fwd training is running */
          bool kIsPagedKV_,
          bool kDoFp8StaticQuant_,
          index_t kBlockPerCu_  = -1,    /* overwrite occupancy if not -1 */
          bool kSkipMinSeqlenQ_ = false, /* skip min seqlen q while chunked prefill */
          bool kHasSink_        = false>
struct TileFmhaFwdPagedKVTraits
{
    static constexpr bool kPadSeqLenQ       = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK       = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ      = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV      = kPadHeadDimV_;
    static constexpr bool kHasLogitsSoftCap = kHasLogitsSoftCap_;
    static constexpr auto BiasEnum          = BiasEnum_;
    static constexpr bool kHasBiasGrad      = kHasBiasGrad_;
    static constexpr bool kStoreLSE         = kStoreLSE_;
    static constexpr bool kIsPagedKV        = kIsPagedKV_;
    static constexpr bool kDoFp8StaticQuant = kDoFp8StaticQuant_;
    static constexpr index_t kBlockPerCu    = kBlockPerCu_;
    static constexpr bool kSkipMinSeqlenQ   = kSkipMinSeqlenQ_;
    static constexpr bool kHasSink          = kHasSink_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          bool kHasLogitsSoftCap_,
          BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          bool kStoreLSE_, /* set to true if either num_splits > 1 or fwd training is running */
          bool kDoFp8StaticQuant_,
          bool kIsPagedKV_,
          bool kHasUnevenSplits_,
          bool kMergeNumHeadGroupsSeqLenQ_ = false,
          index_t kBlockPerCu_             = -1, /* overwrite occupancy if not -1 */
          bool kHasSink_                   = false>
struct TileFmhaFwdSplitKVTraits
{
    static constexpr bool kPadSeqLenQ       = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK       = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ      = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV      = kPadHeadDimV_;
    static constexpr bool kHasLogitsSoftCap = kHasLogitsSoftCap_;
    static constexpr auto BiasEnum          = BiasEnum_;
    static constexpr bool kHasBiasGrad      = kHasBiasGrad_;
    static constexpr bool kStoreLSE         = kStoreLSE_;
    static constexpr bool kDoFp8StaticQuant = kDoFp8StaticQuant_;
    static constexpr bool kIsPagedKV        = kIsPagedKV_;
    // determine if some split (length) is not divisible by tile size
    static constexpr bool kHasUnevenSplits           = kHasUnevenSplits_;
    static constexpr bool kMergeNumHeadGroupsSeqLenQ = kMergeNumHeadGroupsSeqLenQ_;
    static constexpr index_t kBlockPerCu             = kBlockPerCu_;
    static constexpr bool kHasSink                   = kHasSink_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          bool kStoreLSE_,
          bool kDoFp8StaticQuant_,
          index_t kLogMaxSplits_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaFwdSplitKVCombineTraits
{
    static constexpr bool kPadSeqLenQ       = kPadSeqLenQ_;
    static constexpr bool kPadHeadDimV      = kPadHeadDimV_;
    static constexpr bool kStoreLSE         = kStoreLSE_;
    static constexpr bool kDoFp8StaticQuant = kDoFp8StaticQuant_;

    static constexpr index_t kMaxSplits = (1 << kLogMaxSplits_);
    static_assert(kMaxSplits <= get_warp_size() || kMaxSplits % get_warp_size() == 0);
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaFwdAppendKVTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK    = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ   = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV   = kPadHeadDimV_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          index_t kBlockPerCu_ = 2 /* hint to occupancy */>
struct TileFmhaBwdOGradDotOTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadHeadDimV   = kPadHeadDimV_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          index_t kBlockPerCu_ = 2 /* hint to occupancy */>
struct TileFmhaBwdConvertQGradTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadHeadDimQ   = kPadHeadDimQ_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

} // namespace ck_tile
