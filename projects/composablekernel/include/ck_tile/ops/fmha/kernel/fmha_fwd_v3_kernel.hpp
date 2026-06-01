// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

#include <type_traits>
#include <utility>

namespace ck_tile {

/// NOTICE: This kernel is a work in progress and is awaiting upcoming compiler fixes and
/// instruction scheduling optimizations.
template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdV3Kernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    static constexpr bool kIsGroupMode      = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = FmhaPipeline::kHasLogitsSoftCap;
    static constexpr bool kStoreLSE         = FmhaPipeline::kStoreLSE;

    using AttentionVariant = ck_tile::remove_cvref_t<typename FmhaPipeline::AttentionVariant>;
    using FmhaMask         = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

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

    struct FmhaFwdMaskKargs
    {
        // ck_tile::index_t window_size_left, window_size_right;
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
        ck_tile::index_t remap_opt;
    };

    struct FmhaFwdCommonLSEKargs
    {
        void* lse_ptr                     = nullptr;
        ck_tile::index_t nhead_stride_lse = 0;
        ck_tile::index_t batch_stride_lse = 0;
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

    struct FmhaFwdBatchModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<0>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<2>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;

        // Optional cumulative sequence length pointers for batch mode
        // If provided, they override seqlen_q / seqlen_k per-batch to skip tail padding.
        const ck_tile::index_t* cu_seqlen_q_ptr = nullptr; // [batch+1]
        const ck_tile::index_t* cu_seqlen_k_ptr = nullptr; // [batch+1]
    };

    struct FmhaFwdGroupModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<0>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<2>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_q_ptr;
        const int32_t* seqlen_k_ptr;

        // Optional cumulative padded sequence starts (including PAD tokens)
        // Used solely to compute memory offsets when sequences are physically padded.
        const int32_t* cu_seqlen_q_ptr = nullptr; // [batch+1]
        const int32_t* cu_seqlen_k_ptr = nullptr; // [batch+1]
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
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_lse,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              ck_tile::index_t remap_opt,
              const void* cu_seqlen_q_ptr = nullptr,
              const void* cu_seqlen_k_ptr = nullptr)
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
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for logits_soft_cap
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
            kargs.remap_opt         = remap_opt;
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
            kargs.batch_stride_lse = batch_stride_lse;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }

        kargs.cu_seqlen_q_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr);
        kargs.cu_seqlen_k_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_k_ptr);
        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              void* lse_ptr,
              void* o_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_q_ptr,
              const void* seqlen_k_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              ck_tile::index_t remap_opt,
              const void* cu_seqlen_q_ptr = nullptr,
              const void* cu_seqlen_k_ptr = nullptr)
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
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for logits_soft_cap
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_q_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr)};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
            kargs.remap_opt         = remap_opt;
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }

        kargs.cu_seqlen_q_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr);
        kargs.cu_seqlen_k_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_k_ptr);
        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size,
                                                ck_tile::index_t nhead,
                                                ck_tile::index_t max_seqlen_q,
                                                ck_tile::index_t hdim_v)
    {
        if constexpr(kIsGroupMode)
        {
            return dim3(nhead,
                        batch_size,
                        ck_tile::integer_divide_ceil(max_seqlen_q, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v, FmhaPipeline::kN1));
        }
        else
        {
            return dim3(nhead,
                        ck_tile::integer_divide_ceil(max_seqlen_q, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v, FmhaPipeline::kN1),
                        batch_size);
        }
    }

    CK_TILE_DEVICE static constexpr auto
    RemapTileIndices(int32_t tg_idx, int32_t tg_idy, int32_t remap_option)
    {
        if(remap_option < 1)
        {
            return make_tuple(static_cast<int32_t>(gridDim.x - tg_idx - 1), tg_idy);
        }

        int32_t remapped_tg_idx = tg_idx;
        int32_t remapped_tg_idy = tg_idy;

        if(remap_option == 2)
        { // special remapping
            int32_t tmp0 = (remapped_tg_idy & 0x7) * gridDim.x + remapped_tg_idx;
            int32_t tmp1 = tmp0 & 0x7;

            remapped_tg_idx = tmp0 >> 3;
            remapped_tg_idy = (remapped_tg_idy & 0xfffffff8) + tmp1;
        }
        else
        { // normal remapping
            int32_t cus_per_xdim_per_xcc = gridDim.x >> 3;
            int32_t tgs_cu_id            = remapped_tg_idx >> 3;

            if(tgs_cu_id < cus_per_xdim_per_xcc)
            {
                int32_t tgs_xcc_id = remapped_tg_idx & 0x7;
                int32_t new_tg_idx = tgs_xcc_id * cus_per_xdim_per_xcc + tgs_cu_id;

                remapped_tg_idx = new_tg_idx;
            }
        }

        return make_tuple(remapped_tg_idx, remapped_tg_idy);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs&)
    {
        using namespace ck_tile;

        // const index_t num_tile_n1 = ck_tile::integer_divide_ceil(kargs.hdim_v,
        // FmhaPipeline::kN1);

        // assume that num_tile_n1 is always 1
        if constexpr(kIsGroupMode)
        {
            const index_t i_nhead = blockIdx.x;
            const index_t i_batch = blockIdx.y;
            const index_t i_block = blockIdx.z;

            if constexpr(kHasMask)
            {
                return ck_tile::make_tuple(gridDim.z - 1 - i_block, 0, i_nhead, i_batch);
            }
            else
            {
                return ck_tile::make_tuple(i_block, 0, i_nhead, i_batch);
            }
        }
        else
        {
            const index_t i_nhead = blockIdx.x;
            const index_t i_block = blockIdx.y;
            const index_t i_batch = blockIdx.z;

            if constexpr(kHasMask)
            {
                return ck_tile::make_tuple(gridDim.y - 1 - i_block, 0, i_nhead, i_batch);
            }
            else
            {
                return ck_tile::make_tuple(i_block, 0, i_nhead, i_batch);
            }
        }
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using namespace ck_tile;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);

        const index_t i_m0 = amd_wave_read_first_lane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = amd_wave_read_first_lane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q   = 0;
        long_index_t batch_offset_k   = 0;
        long_index_t batch_offset_v   = 0;
        long_index_t batch_offset_lse = 0;
        long_index_t batch_offset_o   = 0;

        if constexpr(kIsGroupMode)
        {
            // Use seqstart_q_ptr and seqstart_k_ptr for physical starts
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
            batch_offset_k = key_start * kargs.stride_k;
            batch_offset_v = key_start * kargs.stride_v;

            if constexpr(kStoreLSE)
            {
                // LSE layout is [nhead, total_seqlen], index by unpadded start
                batch_offset_lse = query_start;
            }
            batch_offset_o = query_start * kargs.stride_o;

            // real logical lengths (exclude PAD)
            // Priority: seqlen_q_ptr > cu_seqlen_q_ptr > calculated from seqstart_q_ptr
            if(kargs.seqlen_q_ptr != nullptr)
            {
                kargs.seqlen_q = kargs.seqlen_q_ptr[i_batch];
            }
            else if(kargs.cu_seqlen_q_ptr != nullptr)
            {
                kargs.seqlen_q =
                    kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
            }
            else
            {
                kargs.seqlen_q = kargs.seqstart_q_ptr[i_batch + 1] - kargs.seqstart_q_ptr[i_batch];
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
            else if(kargs.cu_seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k =
                    kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
            }
            else
            {
                kargs.seqlen_k = kargs.seqstart_k_ptr[i_batch + 1] - kargs.seqstart_k_ptr[i_batch];
            }
        }
        else
        {
            batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            if constexpr(kStoreLSE)
            {
                batch_offset_lse = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
            }
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;

            // If cumulative seqlen pointers are provided, override per-batch effective lengths
            if(kargs.cu_seqlen_q_ptr != nullptr)
            {
                kargs.seqlen_q =
                    kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
            }
            if(kargs.cu_seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k =
                    kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
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

            return pad_tensor_view(
                q_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
                sequence<kPadSeqLenQ, kPadHeadDimQ>{});
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                number<FmhaPipeline::kAlignmentV>{},
                number<1>{});

            return pad_tensor_view(
                v_dram_naive,
                make_tuple(number<FmhaPipeline::kK1>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenK, kPadHeadDimV>{});
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(number<FmhaPipeline::kK1>{}, number<FmhaPipeline::kN1>{}),
                             {0, i_n1});

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
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
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
            return FmhaPipeline{}(q_dram_window,
                                  k_dram_window,
                                  v_dram_window,
                                  lse_dram_window,
                                  mask,
                                  kargs.scale_s,
                                  variant,
                                  variant_params,
                                  block_indices,
                                  smem_ptr);
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
