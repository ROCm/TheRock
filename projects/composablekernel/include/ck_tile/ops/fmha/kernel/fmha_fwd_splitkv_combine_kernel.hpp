// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdSplitKVCombineKernel
{
    using FmhaPipeline     = remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    static constexpr index_t kNumWarps   = FmhaPipeline::kNumWarps;
    static constexpr index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    static_assert(kBlockPerCu > 0);
    static constexpr index_t kBlockPerCuInput = FmhaPipeline::Problem::kBlockPerCu;

    using LSEDataType  = remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using OaccDataType = remove_cvref_t<typename FmhaPipeline::OaccDataType>;
    using ODataType    = remove_cvref_t<typename FmhaPipeline::ODataType>;

    static constexpr bool kIsGroupMode      = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kStoreLSE         = FmhaPipeline::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant = FmhaPipeline::Problem::kDoFp8StaticQuant;

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
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_fwd_splitkv_combine_d") + _TS_(FmhaPipeline::kHeadDimV) + "_" + _SS_(t2s<ODataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_"
            "b" + _TS_(FmhaPipeline::kN1) + "_" +
            (kBlockPerCuInput == -1 ? "" : ("o" + _TS_(kBlockPerCu) + "_")) +
            _SS_(FmhaPipeline::name) +
            (pn.empty() ? "_npad" : "_" + pn) +
            (kStoreLSE ? "_lse" : "_nlse" ) +
            (kDoFp8StaticQuant ? "_squant" : "_nsquant" );
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
    struct CommonKargs
    {
        const void* lse_acc_ptr;
        const void* o_acc_ptr;
        void* o_ptr;

        ck_tile::index_t batch;
        ck_tile::index_t seqlen_q;
        ck_tile::index_t hdim_v;
        ck_tile::index_t num_splits;

        ck_tile::index_t row_stride_o_acc;
        ck_tile::index_t row_stride_o;

        ck_tile::index_t nhead_stride_lse_acc;
        ck_tile::index_t nhead_stride_o_acc;
        ck_tile::index_t nhead_stride_o;

        ck_tile::index_t split_stride_lse_acc;
        ck_tile::index_t split_stride_o_acc;
    };

    struct CommonLSEKargs
    {
        void* lse_ptr                     = nullptr;
        ck_tile::index_t nhead_stride_lse = 0;
        ck_tile::index_t batch_stride_lse = 0;
    };

    struct Fp8StaticQuantKargs
    {
        float scale_o;
    };

    struct BatchModeKargs
        : CommonKargs,
          std::conditional_t<kStoreLSE, CommonLSEKargs, EmptyKargs<0>>,
          std::conditional_t<kDoFp8StaticQuant, Fp8StaticQuantKargs, EmptyKargs<1>>
    {
        ck_tile::index_t batch_stride_lse_acc;
        ck_tile::index_t batch_stride_o_acc;
        ck_tile::index_t batch_stride_o;
    };

    struct GroupModeKargs
        : CommonKargs,
          std::conditional_t<kStoreLSE, CommonLSEKargs, EmptyKargs<0>>,
          std::conditional_t<kDoFp8StaticQuant, Fp8StaticQuantKargs, EmptyKargs<3>>
    {
        const int32_t* seqstart_q_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode, GroupModeKargs, BatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* lse_acc_ptr,
              const void* o_acc_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t batch,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_splits,
              float scale_o,
              ck_tile::index_t row_stride_o_acc,
              ck_tile::index_t row_stride_o,
              ck_tile::index_t nhead_stride_lse_acc,
              ck_tile::index_t nhead_stride_o_acc,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_lse_acc,
              ck_tile::index_t batch_stride_o_acc,
              ck_tile::index_t batch_stride_lse,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t split_stride_lse_acc,
              ck_tile::index_t split_stride_o_acc)
    {
        Kargs kargs{{lse_acc_ptr,
                     o_acc_ptr,
                     o_ptr,
                     batch,
                     seqlen_q,
                     hdim_v,
                     num_splits,
                     row_stride_o_acc,
                     row_stride_o,
                     nhead_stride_lse_acc,
                     nhead_stride_o_acc,
                     nhead_stride_o,
                     split_stride_lse_acc,
                     split_stride_o_acc}, // args for common karg
                    {},                   // placeholder for lse
                    {},                   // placeholder for fp8_static_quant args
                    batch_stride_lse_acc,
                    batch_stride_o_acc,
                    batch_stride_o};

        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
            kargs.batch_stride_lse = batch_stride_lse;
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_o = scale_o;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* lse_acc_ptr,
              const void* o_acc_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t batch,
              const void* seqstart_q_ptr,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_splits,
              float scale_o,
              ck_tile::index_t row_stride_o_acc,
              ck_tile::index_t row_stride_o,
              ck_tile::index_t nhead_stride_lse_acc,
              ck_tile::index_t nhead_stride_o_acc,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t split_stride_lse_acc,
              ck_tile::index_t split_stride_o_acc)
    {
        Kargs kargs{{lse_acc_ptr,
                     o_acc_ptr,
                     o_ptr,
                     batch,
                     -1, // seqlen will be updated by another pointer
                     hdim_v,
                     num_splits,
                     row_stride_o_acc,
                     row_stride_o,
                     nhead_stride_lse_acc,
                     nhead_stride_o_acc,
                     nhead_stride_o,
                     split_stride_lse_acc,
                     split_stride_o_acc}, // args for common karg
                    {},                   // placeholder for lse
                    {},                   // placeholder for fp8_static_quant args
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr)};

        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_o = scale_o;
        }

        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size,
                                                ck_tile::index_t nhead,
                                                ck_tile::index_t max_seqlen_q,
                                                ck_tile::index_t hdim_v)
    {
        // Recalculate kM0 = get_warp_size() / NThreads on host
        const index_t m0 = (is_wave32() ? 32 : 64) / FmhaPipeline::Problem::NThreads;
        // TODO: this may need tuning
        return dim3(ck_tile::integer_divide_ceil(max_seqlen_q, m0) *
                        ck_tile::integer_divide_ceil(hdim_v, FmhaPipeline::kN1),
                    nhead,
                    batch_size);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        const index_t num_tile_n1 = ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1);

        const index_t i_block = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

        return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
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

        long_index_t batch_offset_lse_acc = 0;
        long_index_t batch_offset_o_acc   = 0;
        long_index_t batch_offset_lse     = 0;
        long_index_t batch_offset_o       = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_lse_acc = query_start;
            batch_offset_o_acc   = query_start * kargs.row_stride_o_acc;

            if constexpr(kStoreLSE)
            {
                batch_offset_lse = query_start;
            }

            batch_offset_o = query_start * kargs.row_stride_o;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }
        }
        else
        {
            batch_offset_lse_acc = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse_acc;
            batch_offset_o_acc   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o_acc;

            if constexpr(kStoreLSE)
            {
                batch_offset_lse = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
            }

            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
        }

        // for simplicity, batch stride we just modify the pointer
        const LSEDataType* lse_acc_ptr =
            reinterpret_cast<const LSEDataType*>(kargs.lse_acc_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lse_acc + batch_offset_lse_acc;
        const OaccDataType* o_acc_ptr =
            reinterpret_cast<const OaccDataType*>(kargs.o_acc_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o_acc + batch_offset_o_acc;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // LSEacc/Oacc DRAM and DRAM windows
        const auto lse_acc_dram = [&]() {
            const auto lse_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                lse_acc_ptr,
                make_tuple(kargs.num_splits, kargs.seqlen_q),
                make_tuple(kargs.split_stride_lse_acc, number<1>{}),
                number<FmhaPipeline::kAlignmentLSEacc>{},
                number<1>{});

            return pad_tensor_view(
                lse_acc_dram_naive,
                make_tuple(number<FmhaPipeline::kMaxSplits>{}, number<FmhaPipeline::kM0>{}),
                sequence<true, kPadSeqLenQ>{});
        }();

        auto o_acc_dram = [&]() {
            const auto o_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_acc_ptr,
                make_tuple(kargs.num_splits, kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.split_stride_o_acc, kargs.row_stride_o_acc, number<1>{}),
                number<FmhaPipeline::kAlignmentOacc>{},
                number<1>{});

            // read kNumWarps * (kM0, kN1) o_acc tiles simultaneously by kNumWarps warps
            const auto o_acc_dram_view = pad_tensor_view(
                o_acc_dram_naive,
                make_tuple(
                    number<kNumWarps>{}, number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                sequence<true, kPadSeqLenQ, kPadHeadDimV>{});

            const index_t padded_num_splits =
                o_acc_dram_view.get_tensor_descriptor().get_lengths()[number<0>{}];
            const index_t padded_seqlen_q =
                o_acc_dram_view.get_tensor_descriptor().get_lengths()[number<1>{}];
            const index_t padded_hdim_v =
                o_acc_dram_view.get_tensor_descriptor().get_lengths()[number<2>{}];

            const index_t num_m_tiles = integer_divide_floor(padded_seqlen_q, FmhaPipeline::kM0);

            // transform tensor view by following steps, given shape: (padded_num_splits,
            // padded_seqlen_q, padded_hdim_v)
            //     1. unmerge to (padded_num_splits, num_m_tiles, kM0, padded_hdim_v)
            //     2. transpose to (num_m_tiles, padded_num_splits, kM0, padded_hdim_v)
            //     3. merge to (num_m_tiles * padded_num_splits * kM0, padded_hdim_v)
            auto transposed = transform_tensor_view(
                o_acc_dram_view,
                make_tuple(make_pass_through_transform(padded_num_splits),
                           make_unmerge_transform(make_tuple(num_m_tiles, FmhaPipeline::kM0)),
                           make_pass_through_transform(padded_hdim_v)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<1>{}, sequence<0, 2>{}, sequence<3>{}));

            return transform_tensor_view(
                transposed,
                make_tuple(make_merge_transform(
                               make_tuple(num_m_tiles, padded_num_splits, FmhaPipeline::kM0)),
                           make_pass_through_transform(padded_hdim_v)),
                make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }();

        auto lse_acc_dram_window = make_tile_window(
            lse_acc_dram,
            make_tuple(number<FmhaPipeline::kMaxSplits>{}, number<FmhaPipeline::kM0>{}),
            {0, i_m0});

        const index_t padded_num_splits =
            integer_divide_ceil(kargs.num_splits, kNumWarps) * kNumWarps;

        auto o_acc_dram_window = make_tile_window(
            o_acc_dram,
            make_tuple(number<kNumWarps * FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
            {i_tile_m * padded_num_splits * FmhaPipeline::kM0, i_n1});

        // LSE DRAM window
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
                        number<FmhaPipeline::kAlignmentLSE>{},
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

        auto o_acc_tile = [&]() {
            if constexpr(kDoFp8StaticQuant)
            {
                return FmhaPipeline{}(
                    lse_acc_dram_window,
                    o_acc_dram_window,
                    lse_dram_window,
                    identity{},                                          // lse_element_func
                    composes(saturates<fp8_t>{}, scales{kargs.scale_o}), // o_acc_element_func
                    kargs.num_splits,
                    smem_ptr);
            }
            else
            {
                return FmhaPipeline{}(lse_acc_dram_window,
                                      o_acc_dram_window,
                                      lse_dram_window,
                                      kargs.num_splits,
                                      smem_ptr);
            }
        }();

        // O DRAM and DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.row_stride_o, number<1>{}),
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
