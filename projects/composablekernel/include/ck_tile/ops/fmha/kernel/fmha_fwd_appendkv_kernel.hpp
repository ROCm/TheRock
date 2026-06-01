// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename FmhaPipeline_>
struct FmhaFwdAppendKVKernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FmhaPipeline::Problem::kBlockPerCu;

    using QDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;

    using VLayout                    = ck_tile::remove_cvref_t<typename FmhaPipeline::VLayout>;
    static constexpr bool kApplyRoPE = FmhaPipeline::RotaryEnum != RotaryEmbeddingEnum::NONE;
    static constexpr bool kIsPagedKV = FmhaPipeline::kIsPagedKV;

    static constexpr bool kPadSeqLenQ  = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = FmhaPipeline::kPadHeadDimV;

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
            _SS_("fmha_fwd_appendkv_d") + _TS_(FmhaPipeline::kK0) + "_" + _SS_(t2s<QDataType>::name) + "_"
            "b" + _TS_(FmhaPipeline::kM0) + "x" + _TS_(FmhaPipeline::kN0) + "x" + _TS_(FmhaPipeline::kK0) + "x" +
                  _TS_(FmhaPipeline::kN1) + "_" + (kBlockPerCuInput == -1 ? "" : ("o" + _TS_(kBlockPerCu) + "_")) +
            "v" + (std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor> ? "r" : "c") + (pn.empty() ? "" : "_" + pn) 
            + (!kApplyRoPE ? _SS_("") : (_SS_("_") + RotaryEmbeddingEnumToStr<FmhaPipeline::RotaryEnum>::name))
            + (kIsPagedKV ? "_pagedkv" : "" );
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct EmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct BasicKargs
    {
        void* q_ptr;
        void* k_ptr;
        const void* knew_ptr;
        void* v_ptr;
        const void* vnew_ptr;

        const int32_t* seqlen_k_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t seqlen_knew;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_ratio_qk;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_knew;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_vnew;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_knew;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_vnew;

        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_knew;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_vnew;
    };

    struct RoPEKargs
    {
        const void* rotary_cos_ptr;
        const void* rotary_sin_ptr;
        ck_tile::index_t rotary_dim;
        bool has_mask;
    };

    struct PageBlockTableKargs
    {
        const int32_t* block_table_ptr;
        ck_tile::index_t batch_stride_block_table;
        ck_tile::index_t page_block_size;
    };

    struct CacheBatchIdxKargs
    {
        const int32_t* cache_batch_idx;
    };

    struct Kargs : BasicKargs,
                   std::conditional_t<kApplyRoPE, RoPEKargs, EmptyKargs<0>>,
                   std::conditional_t<kIsPagedKV, PageBlockTableKargs, CacheBatchIdxKargs>
    {
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(void* q_ptr,
                                                  void* k_ptr,
                                                  const void* knew_ptr,
                                                  void* v_ptr,
                                                  const void* vnew_ptr,
                                                  ck_tile::index_t seqlen_q,
                                                  const void* seqlen_k_ptr,
                                                  ck_tile::index_t seqlen_knew,
                                                  ck_tile::index_t hdim_q,
                                                  ck_tile::index_t hdim_v,
                                                  ck_tile::index_t num_head_q,
                                                  ck_tile::index_t nhead_ratio_qk,
                                                  const void* rotary_cos_ptr,
                                                  const void* rotary_sin_ptr,
                                                  ck_tile::index_t rotary_dim,
                                                  bool has_mask,
                                                  const void* block_table_ptr,
                                                  ck_tile::index_t batch_stride_block_table,
                                                  ck_tile::index_t page_block_size,
                                                  const void* cache_batch_idx,
                                                  ck_tile::index_t stride_q,
                                                  ck_tile::index_t stride_k,
                                                  ck_tile::index_t stride_knew,
                                                  ck_tile::index_t stride_v,
                                                  ck_tile::index_t stride_vnew,
                                                  ck_tile::index_t nhead_stride_q,
                                                  ck_tile::index_t nhead_stride_k,
                                                  ck_tile::index_t nhead_stride_knew,
                                                  ck_tile::index_t nhead_stride_v,
                                                  ck_tile::index_t nhead_stride_vnew,
                                                  ck_tile::index_t batch_stride_q,
                                                  ck_tile::index_t batch_stride_k,
                                                  ck_tile::index_t batch_stride_knew,
                                                  ck_tile::index_t batch_stride_v,
                                                  ck_tile::index_t batch_stride_vnew)
    {
        Kargs kargs{
            {q_ptr,
             k_ptr,
             knew_ptr,
             v_ptr,
             vnew_ptr,
             reinterpret_cast<const int32_t*>(seqlen_k_ptr),
             seqlen_q,
             -1, // seqlen_k will be updated by content of seqlen_k_ptr
             seqlen_knew,
             hdim_q,
             hdim_v,
             num_head_q,
             nhead_ratio_qk,
             stride_q,
             stride_k,
             stride_knew,
             stride_v,
             stride_vnew,
             nhead_stride_q,
             nhead_stride_k,
             nhead_stride_knew,
             nhead_stride_v,
             nhead_stride_vnew,
             batch_stride_q,
             batch_stride_k,
             batch_stride_knew,
             batch_stride_v,
             batch_stride_vnew}, // args for common karg
            {},                  // placeholder for rope
            {}                   // placeholder for paged-block table or cache_batch_idx
        };

        if constexpr(kApplyRoPE)
        {
            kargs.rotary_cos_ptr = rotary_cos_ptr;
            kargs.rotary_sin_ptr = rotary_sin_ptr;
            kargs.rotary_dim     = rotary_dim;
            kargs.has_mask       = has_mask;
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

        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size,
                                                ck_tile::index_t nhead,
                                                ck_tile::index_t seqlen_q,
                                                ck_tile::index_t seqlen_knew)
    {
        // TODO: this may need tuning
        return dim3(std::max(ck_tile::integer_divide_ceil(seqlen_q, FmhaPipeline::kM0),
                             ck_tile::integer_divide_ceil(seqlen_knew, FmhaPipeline::kN0)),
                    nhead,
                    batch_size);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& /* kargs */)
    {
        const index_t i_tile  = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        return ck_tile::make_tuple(i_tile, i_nhead, i_batch);
    }

    CK_TILE_HOST static dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // divide problem
        const auto [i_tile, i_nhead, i_batch] = GetTileIndex(kargs);

        const index_t i_m0 = amd_wave_read_first_lane(i_tile * FmhaPipeline::kM0);
        const index_t i_n0 = amd_wave_read_first_lane(i_tile * FmhaPipeline::kN0);

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

        const long_index_t batch_offset_q =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
        const long_index_t batch_offset_k =
            static_cast<long_index_t>(i_cache_batch) * kargs.batch_stride_k;
        const long_index_t batch_offset_knew =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_knew;
        const long_index_t batch_offset_v =
            static_cast<long_index_t>(i_cache_batch) * kargs.batch_stride_v;
        const long_index_t batch_offset_vnew =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_vnew;

        kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];

        // for simplicity, batch stride we just modify the pointer
        QDataType* q_ptr = reinterpret_cast<QDataType*>(kargs.q_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                           batch_offset_q;
        KDataType* k_ptr =
            reinterpret_cast<KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const KDataType* knew_ptr =
            reinterpret_cast<const KDataType*>(kargs.knew_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_knew +
            batch_offset_knew;
        VDataType* v_ptr =
            reinterpret_cast<VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const VDataType* vnew_ptr =
            reinterpret_cast<const VDataType*>(kargs.vnew_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_vnew +
            batch_offset_vnew;

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
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenQ, kPadHeadDimQ>{});
        }();

        const auto make_k_dram = [&](KDataType* data, index_t height) {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                data, // will update this pointer if using paged-kvcache
                make_tuple(height, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        };
        const auto k_dram = [&]() {
            if constexpr(kIsPagedKV)
            {
                return make_k_dram(nullptr, kargs.page_block_size);
            }
            else
            {
                return make_k_dram(k_ptr, kargs.seqlen_k + kargs.seqlen_knew);
            }
        }();

        const auto knew_dram = [&]() {
            const auto knew_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                knew_ptr,
                make_tuple(kargs.seqlen_knew, kargs.hdim_q),
                make_tuple(kargs.stride_knew, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                knew_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();

        const auto make_v_dram = [&](VDataType* data, index_t length) {
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
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kN0>{}),
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
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kN0>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
        };
        const auto v_dram = [&]() {
            if constexpr(kIsPagedKV)
            {
                return make_v_dram(nullptr, kargs.page_block_size);
            }
            else
            {
                return make_v_dram(v_ptr, kargs.seqlen_k + kargs.seqlen_knew);
            }
        }();

        const auto vnew_dram = [&]() {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto vnew_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    vnew_ptr,
                    make_tuple(kargs.seqlen_knew, kargs.hdim_v),
                    make_tuple(kargs.stride_vnew, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                const auto vnew_dram_transposed = transform_tensor_view(
                    vnew_dram_naive,
                    make_tuple(make_pass_through_transform(kargs.hdim_v),
                               make_pass_through_transform(kargs.seqlen_knew)),
                    make_tuple(sequence<1>{}, sequence<0>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return pad_tensor_view(
                    vnew_dram_transposed,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kN0>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
            else
            {
                const auto vnew_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    vnew_ptr,
                    make_tuple(kargs.hdim_v, kargs.seqlen_knew),
                    make_tuple(kargs.stride_vnew, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                return pad_tensor_view(
                    vnew_dram_naive,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kN0>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
        }();

        constexpr auto q_rotary_cos_sin_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0 / 2>{});
        const auto q_rotary_cos_dram_window = [&]() {
            if constexpr(kApplyRoPE)
            {
                const auto rotary_cos_dram_native =
                    make_naive_tensor_view<address_space_enum::global>(
                        reinterpret_cast<const QDataType*>(kargs.rotary_cos_ptr) +
                            kargs.seqlen_k * (kargs.rotary_dim / 2),
                        make_tuple(kargs.seqlen_q, kargs.rotary_dim / 2),
                        make_tuple(kargs.has_mask * (kargs.rotary_dim / 2), 1),
                        number<8>{},
                        number<1>{});

                const auto rotary_cos_dram = [&]() {
                    return pad_tensor_view(rotary_cos_dram_native,
                                           q_rotary_cos_sin_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadHeadDimQ>{});
                }();

                return make_tile_window(
                    rotary_cos_dram, q_rotary_cos_sin_dram_window_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(q_rotary_cos_sin_dram_window_lengths);
            }
        }();
        const auto q_rotary_sin_dram_window = [&]() {
            if constexpr(kApplyRoPE)
            {
                const auto rotary_sin_dram_native =
                    make_naive_tensor_view<address_space_enum::global>(
                        reinterpret_cast<const QDataType*>(kargs.rotary_sin_ptr) +
                            kargs.seqlen_k * (kargs.rotary_dim / 2),
                        make_tuple(kargs.seqlen_q, kargs.rotary_dim / 2),
                        make_tuple(kargs.has_mask * (kargs.rotary_dim / 2), 1),
                        number<8>{},
                        number<1>{});

                const auto rotary_sin_dram = [&]() {
                    return pad_tensor_view(rotary_sin_dram_native,
                                           q_rotary_cos_sin_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadHeadDimQ>{});
                }();

                return make_tile_window(
                    rotary_sin_dram, q_rotary_cos_sin_dram_window_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(q_rotary_cos_sin_dram_window_lengths);
            }
        }();

        constexpr auto knew_rotary_cos_sin_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0 / 2>{});
        const auto knew_rotary_cos_dram_window = [&]() {
            if constexpr(kApplyRoPE)
            {
                const auto rotary_cos_dram_native =
                    make_naive_tensor_view<address_space_enum::global>(
                        reinterpret_cast<const KDataType*>(kargs.rotary_cos_ptr) +
                            kargs.seqlen_k * (kargs.rotary_dim / 2),
                        make_tuple(kargs.seqlen_knew, kargs.rotary_dim / 2),
                        make_tuple(kargs.rotary_dim / 2, 1),
                        number<8>{},
                        number<1>{});

                const auto rotary_cos_dram = [&]() {
                    return pad_tensor_view(rotary_cos_dram_native,
                                           knew_rotary_cos_sin_dram_window_lengths,
                                           sequence<kPadSeqLenK, kPadHeadDimQ>{});
                }();

                return make_tile_window(
                    rotary_cos_dram, knew_rotary_cos_sin_dram_window_lengths, {i_n0, 0});
            }
            else
            {
                return make_null_tile_window(knew_rotary_cos_sin_dram_window_lengths);
            }
        }();
        const auto knew_rotary_sin_dram_window = [&]() {
            if constexpr(kApplyRoPE)
            {
                const auto rotary_sin_dram_native =
                    make_naive_tensor_view<address_space_enum::global>(
                        reinterpret_cast<const KDataType*>(kargs.rotary_sin_ptr) +
                            kargs.seqlen_k * (kargs.rotary_dim / 2),
                        make_tuple(kargs.seqlen_knew, kargs.rotary_dim / 2),
                        make_tuple(kargs.rotary_dim / 2, 1),
                        number<8>{},
                        number<1>{});

                const auto rotary_sin_dram = [&]() {
                    return pad_tensor_view(rotary_sin_dram_native,
                                           knew_rotary_cos_sin_dram_window_lengths,
                                           sequence<kPadSeqLenK, kPadHeadDimQ>{});
                }();

                return make_tile_window(
                    rotary_sin_dram, knew_rotary_cos_sin_dram_window_lengths, {i_n0, 0});
            }
            else
            {
                return make_null_tile_window(knew_rotary_cos_sin_dram_window_lengths);
            }
        }();

        auto k_page_block_navigator = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(kIsPagedKV)
            {
                const auto* block_indices =
                    reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                    i_batch_ * kargs.batch_stride_block_table;
                const index_t num_blocks =
                    integer_divide_ceil(kargs.seqlen_k + kargs.seqlen_knew, kargs.page_block_size);

                const long_index_t fixed_offset =
                    static_cast<long_index_t>(i_nhead_ / kargs.nhead_ratio_qk) *
                    kargs.nhead_stride_k;

                return make_page_block_navigator<KDataType, 0>(
                    kargs.k_ptr,
                    kargs.batch_stride_k,
                    fixed_offset,
                    block_indices,
                    num_blocks,
                    kargs.page_block_size,
                    k_dram,
                    make_k_dram(nullptr,
                                (kargs.seqlen_k + kargs.seqlen_knew) -
                                    (num_blocks - 1) * kargs.page_block_size));
            }
            else
            {
                return make_page_block_navigator(k_dram);
            }
        }();

        auto v_page_block_navigator = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(kIsPagedKV)
            {
                const auto* block_indices =
                    reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                    i_batch_ * kargs.batch_stride_block_table;
                const index_t num_blocks =
                    integer_divide_ceil(kargs.seqlen_k + kargs.seqlen_knew, kargs.page_block_size);

                const long_index_t fixed_offset =
                    static_cast<long_index_t>(i_nhead_ / kargs.nhead_ratio_qk) *
                    kargs.nhead_stride_v;

                return make_page_block_navigator<VDataType, 1>(
                    kargs.v_ptr,
                    kargs.batch_stride_v,
                    fixed_offset,
                    block_indices,
                    num_blocks,
                    kargs.page_block_size,
                    v_dram,
                    make_v_dram(nullptr,
                                (kargs.seqlen_k + kargs.seqlen_knew) -
                                    (num_blocks - 1) * kargs.page_block_size));
            }
            else
            {
                return make_page_block_navigator(v_dram);
            }
        }();

        auto q_dram_window =
            make_tile_window(q_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                             {i_m0, 0});

        const bool skip_append_kv = kargs.seqlen_knew <= i_n0;
        // window origin = (0, 0) if no work to do for current block
        auto [i_page_block_k, k_dram_window] = k_page_block_navigator.make_tile_window(
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
            {!skip_append_kv * (kargs.seqlen_k + i_n0), 0});

        auto knew_dram_window =
            make_tile_window(knew_dram,
                             make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                             {i_n0, 0});

        // window origin = (0, 0) if no work to do for current block
        auto [i_page_block_v, v_dram_window] = v_page_block_navigator.make_tile_window(
            make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kN0>{}),
            {0, !skip_append_kv * (kargs.seqlen_k + i_n0)});

        auto vnew_dram_window =
            make_tile_window(vnew_dram,
                             make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kN0>{}),
                             {0, i_n0});

        // If kApplyRoPe is false, we set the rotary_dim to 0
        auto rotary_dim = [&]() {
            if constexpr(kApplyRoPE)
                return kargs.rotary_dim;
            else
                return 0;
        }();
        FmhaPipeline{}(q_dram_window,
                       k_dram_window,
                       i_page_block_k,
                       k_page_block_navigator,
                       knew_dram_window,
                       v_dram_window,
                       i_page_block_v,
                       v_page_block_navigator,
                       vnew_dram_window,
                       q_rotary_cos_dram_window,
                       q_rotary_sin_dram_window,
                       knew_rotary_cos_dram_window,
                       knew_rotary_sin_dram_window,
                       rotary_dim,
                       kargs.seqlen_q <= i_m0,
                       skip_append_kv);
    }
};

} // namespace ck_tile
