// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

// KV cache memory layout selector.
//
// Layout summary (kVectorSize = 16 / sizeof(KDataType)):
// - VECTORIZED_LAYOUT (swizzled):
//   K: [NumBlocks, NumHeads, HeadDim/kVectorSize, PageSize, kVectorSize]
//   V: [NumBlocks, NumHeads, PageSize/kVectorSize, HeadDim, kVectorSize]
// - LINEAR_LAYOUT:
//   K: [NumBlocks, PageSize, NumHeads, HeadDim]
//   V: [NumBlocks, PageSize, NumHeads, HeadDim]
enum class BlockAttentionKVCacheMemoryLayoutEnum
{
    VECTORIZED_LAYOUT = 0,
    LINEAR_LAYOUT     = 1,
};

// KV cache lookup table layout selector.
// - VLLM_BLOCK_TABLE_2D: block_table[batch, max_blocks_per_seq]
// - SGLANG_PAGE_TABLE_1D: kv_page_indices[kv_indptr[b] ... kv_indptr[b+1])
enum class BlockAttentionKVCacheLookupTableEnum
{
    VLLM_BLOCK_TABLE_2D  = 0,
    SGLANG_PAGE_TABLE_1D = 1,
};

} // namespace ck_tile
