// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"
#include "ck_tile/ref/naive_attention.hpp"
#include "fmha_fwd.hpp"
#include "utils.hpp"
#include "ck_tile/utility/json_dump.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <cmath>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if CK_TILE_FMHA_FWD_APPENDKV_API && !CK_TILE_FMHA_FWD_SPLITKV_API
#error "we should enable fmha_fwd_splitkv() api in order to cooperate with fmha_fwd_appendkv()"
#endif

enum class fwd_result
{
    success,
    failure,
    invalid_args,
    no_instance,
};

// different threshold for different dtype
template <typename DataTypeConfig>
auto get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<FmhaFwdFp32>(std::string /*init_method*/)
{
    double rtol = 1e-5;
    double atol = 1e-5;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<FmhaFwdBf16>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<FmhaFwdFp8>(std::string /*init_method*/)
{
    using TypeConfig  = FmhaFwdTypeConfig<FmhaFwdFp8>;
    using ODataType   = typename TypeConfig::ODataType;
    float o_dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<ODataType>::max());
    double rtol       = 0;
    double atol       = 16 * (o_dtype_max > 240 ? 2 : 1);
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<FmhaFwdFp8Bf16>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1.8e-1;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<FmhaFwdFp8Fp32>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1.8e-1;
    return ck_tile::make_tuple(rtol, atol);
}

int num_splits_heuristic(int batch_nhead_mblocks, int num_SMs, int max_splits)
{
    // If we have enough to almost fill the SMs, then just use 1 split
    if(batch_nhead_mblocks >= 0.8f * num_SMs)
    {
        return 1;
    }
    max_splits           = std::min({max_splits, num_SMs});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        float n_waves = float(batch_nhead_mblocks * num_splits) / num_SMs;
        float eff     = n_waves / ceil(n_waves);
        // printf("num_splits = %d, eff = %f\n", num_splits, eff);
        if(eff > max_efficiency)
        {
            max_efficiency = eff;
        }
        efficiency.push_back(eff);
    }
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if(efficiency[num_splits - 1] >= 0.85 * max_efficiency)
        {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

int override_num_splits_if_necessary(
    int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits)
{
    (void)hdim_v;
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return num_splits;
    }

    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return num_splits;
    }

    // tile size should match the generate.py
    const int kM0 = 64;

    const int num_m_blocks = ck_tile::integer_divide_ceil(max_seqlen_q, kM0);

    if(num_splits < 1 && p_drop == 0.0f)
    {
        return num_splits_heuristic(
            batch * nhead * num_m_blocks, props.multiProcessorCount * 2, 128);
    }

    return num_splits;
}

template <typename DataTypeConfig>
fwd_result fmha_fwd_run(mode_enum mode,
                        ck_tile::index_t batch,
                        ck_tile::index_t nhead,
                        ck_tile::index_t nhead_k,
                        std::vector<ck_tile::index_t> seqlen_qs,
                        std::vector<ck_tile::index_t> seqlen_ks,
                        ck_tile::index_t hdim_q,
                        ck_tile::index_t hdim_v,
                        ck_tile::index_t seqlen_knew,
                        std::vector<ck_tile::index_t> seqlen_qpads,
                        std::vector<ck_tile::index_t> seqlen_kpads,
                        std::vector<ck_tile::index_t> q_eff_lens_per_batch,
                        std::vector<ck_tile::index_t> kv_eff_lens_per_batch,
                        ck_tile::index_t rotary_dim,
                        bool i_perm,
                        bool o_perm,
                        float scale_s,
                        float logits_soft_cap,
                        bool is_v_rowmajor,
                        bool lse,
                        ck_tile::index_t page_block_size,
                        bool use_cache_batch_idx,
                        std::string bias_str,
                        float p_drop,
                        uint64_t drop_seed,
                        uint64_t drop_offset,
                        bool drop_prefs,
                        std::string mask_str,
                        std::string qscale_str,
                        bool is_rotary_interleaved,
                        ck_tile::index_t num_splits,
                        std::string init_method,
                        uint32_t seed,
                        int do_validation,
                        const ck_tile::stream_config& stream_config,
                        std::optional<std::string> json = std::nullopt)
{
    const std::string data_type = []() {
        if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp32>)
            return "fp32";
        else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp16>)
            return "fp16";
        else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdBf16>)
            return "bf16";
        else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp8>)
            return "fp8";
        else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdBf8>)
            return "bf8";
        else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp8Bf16>)
            return "fp8bf16";
        else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp8Fp32>)
            return "fp8fp32";
        else
            static_assert(false);
    }();

    if(nhead_k < 0)
        nhead_k = nhead;
    if(nhead % nhead_k != 0)
    {
        std::cerr << "nhead:" << nhead << " must be multiple of nhead_k:" << nhead_k << std::endl;
        return fwd_result::invalid_args;
    }

    std::mt19937 random_engine(seed != 0 ? seed : std::random_device{}());
    auto next_seed = [&random_engine]() { return static_cast<unsigned int>(random_engine()); };

    if(hdim_v < 0)
        hdim_v = hdim_q;

#if !CK_TILE_FMHA_FWD_APPENDKV_API
    if(seqlen_knew != 0)
    {
        std::cerr << "fmha_fwd_appendkv() is not enabled. ignoring the 's_knew' option"
                  << std::endl;
        seqlen_knew = 0;
    }
#endif
    if(seqlen_knew < 0)
    {
        seqlen_knew = randint<ck_tile::index_t>(1, seqlen_qs[0], random_engine);
    }

    if constexpr(!(std::is_same_v<DataTypeConfig, FmhaFwdFp16> ||
                   std::is_same_v<DataTypeConfig, FmhaFwdBf16>))
    {
        if(0 < rotary_dim)
        {
            std::cerr << "rotary embedding is only available for data type=fp16|bf16" << std::endl;
            return fwd_result::invalid_args;
        }
    }
#if !CK_TILE_FMHA_FWD_APPENDKV_API
    else if(0 < rotary_dim)
    {
        std::cerr << "rotary embedding is not supported. ignoring the 'rotary_dim' option"
                  << std::endl;
        rotary_dim = 0;
    }
#endif
    // to use fmha_fwd_appendkv(), make sure it's in batch mode
    const bool need_append_kvcache = (0 < seqlen_knew || 0 < rotary_dim);
    if(need_append_kvcache && mode == mode_enum::group)
    {
        std::cerr << "fmha_fwd_appendkv() will be invoked. ignoring the 'mode' option" << std::endl;
        mode = mode_enum::batch;
    }
    if(!(rotary_dim <= hdim_q))
    {
        std::cerr << "rotary_dim should be less than or equal to head dim for q" << std::endl;
        return fwd_result::invalid_args;
    }
    else if(!(rotary_dim % 16 == 0))
    {
        std::cerr << "only rotary dimensions divisible by 16 are currently supported" << std::endl;
        return fwd_result::invalid_args;
    }

#if(!(CK_TILE_FMHA_FWD_APPENDKV_API || CK_TILE_FMHA_FWD_SPLITKV_API || \
      CK_TILE_FMHA_FWD_PAGEDKV_API))
    if(0 < page_block_size)
    {
        std::cerr << "paged-kvcache is not supported. ignoring the 'page_block_size' option"
                  << std::endl;
        page_block_size = 0;
    }
#endif
    if(!(page_block_size % 128 == 0))
    {
        std::cerr << "only paged-kvcache block size divisible by 128 are currently supported"
                  << std::endl;
        return fwd_result::invalid_args;
    }

#if !(CK_TILE_FMHA_FWD_APPENDKV_API || CK_TILE_FMHA_FWD_SPLITKV_API || CK_TILE_FMHA_FWD_PAGEDKV_API)
    if(use_cache_batch_idx)
    {
        std::cerr << "split-kv is not supported. ignoring the 'cache_batch_idx' option"
                  << std::endl;
        use_cache_batch_idx = false;
    }
#else
    if(use_cache_batch_idx)
    {
        if(0 < page_block_size)
        {
            std::cerr << "paged-kvcache does not support cache_batch_idx. ignoring the "
                         "'cache_batch_idx' option"
                      << std::endl;
            use_cache_batch_idx = false;
        }
        else if(mode == mode_enum::group)
        {
            std::cerr << "group mode will not use cache_batch_idx. ignoring the "
                         "'cache_batch_idx' option"
                      << std::endl;
            use_cache_batch_idx = false;
        }
    }
#endif
    const bool use_kvcache = (need_append_kvcache || use_cache_batch_idx || 0 < page_block_size);

    // Reject unsupported padding usage in special pipelines (appendkv / splitkv / pagedkv)
    const bool has_group_q_padding =
        mode == mode_enum::group && (!seqlen_qpads.empty() && seqlen_qpads[0] > 0);
    const bool has_group_k_padding =
        mode == mode_enum::group && (!seqlen_kpads.empty() && seqlen_kpads[0] > 0);
    const bool has_group_padding   = has_group_q_padding || has_group_k_padding;
    const bool has_batch_q_padding = mode == mode_enum::batch && !q_eff_lens_per_batch.empty();
    const bool has_batch_k_padding = mode == mode_enum::batch && !kv_eff_lens_per_batch.empty();
    const bool has_batch_padding   = has_batch_q_padding || has_batch_k_padding;
    const bool using_appendkv      = (0 < seqlen_knew || 0 < rotary_dim);
    const bool using_pagedkv       = (0 < page_block_size);
    const bool using_splitkv       = (num_splits > 1) || use_cache_batch_idx;
    if((using_appendkv || using_pagedkv || using_splitkv) &&
       (has_group_padding || has_batch_padding))
    {
        std::cerr << "Padding (physical or effective lengths) is not supported with "
                     "appendkv/splitkv/pagedkv pipelines"
                  << std::endl;
        return fwd_result::invalid_args;
    }

    std::tie(seqlen_qs, seqlen_ks, seqlen_qpads, seqlen_kpads) =
        generate_missing_seqlens(mode,
                                 batch,
                                 seqlen_qs,
                                 seqlen_ks,
                                 seqlen_qpads,
                                 seqlen_kpads,
                                 /*seqlen_k_min=*/0 < seqlen_knew ? seqlen_knew : 0,
                                 need_append_kvcache,
                                 random_engine);
    for(ck_tile::index_t wb = 0; wb < batch; ++wb)
    {
        if(seqlen_kpads[wb] > 0 && seqlen_kpads[wb] < seqlen_ks[wb])
        {
            std::cerr << "kpad must be greater than or equal to seqlen for k" << std::endl;
            return fwd_result::invalid_args;
        }
        if(seqlen_qpads[wb] > 0 && seqlen_qpads[wb] < seqlen_qs[wb])
        {
            std::cerr << "qpad must be greater than or equal to seqlen for q" << std::endl;
            return fwd_result::invalid_args;
        }
    }

    // compute kvcache seqlen_k (before appending knew/vnew)
    auto cache_seqlen_ks = seqlen_ks;
    std::transform(cache_seqlen_ks.begin(),
                   cache_seqlen_ks.end(),
                   cache_seqlen_ks.begin(),
                   [&](auto seqlen_k) { return seqlen_k - seqlen_knew; });

#if 0
    std::cout << "seqlen_qs: " << seqlen_qs << std::endl;
    std::cout << "seqlen_ks: " << seqlen_ks << std::endl;
    std::cout << "seqlen_qpads: " << seqlen_qpads << std::endl;
    std::cout << "seqlen_kpads: " << seqlen_kpads << std::endl;
    std::cout << "cache_seqlen_ks: " << cache_seqlen_ks << std::endl;
#endif

    if(scale_s == .0f)
        scale_s = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q)); // TODO: q ? v ?

    bias_info bias = bias_info::decode(bias_str);

    mask_info mask =
        mask_info::decode(mask_str, seqlen_qs[0], seqlen_ks[0]); // TODO: we don't need x/y anymore

    quant_scale_info qscale = quant_scale_info::decode(qscale_str);

    if(p_drop < 0.0f || p_drop > 1.0f)
    {
        std::cerr << "The value of p_drop should be 0~1" << std::endl;
        return fwd_result::invalid_args;
    }

    bool s_randval = false;
    if(p_drop > 0.0f && do_validation)
    {
        s_randval = true;
    }

#if !CK_TILE_FMHA_FWD_SPLITKV_API
    if(num_splits != 1)
    {
        std::cerr << "split-kv is not supported. ignoring the 'num_splits' option" << std::endl;
        num_splits = 1;
    }
#endif

    const auto seqstart_q_host              = to_seqstarts(seqlen_qs);
    const auto seqstart_k_host              = to_seqstarts(seqlen_ks);
    const auto seqstart_q_with_padding_host = to_seqstarts(seqlen_qpads);
    const auto seqstart_k_with_padding_host = to_seqstarts(seqlen_kpads);

    // Optional batch-mode cumulative seqlen overrides
    std::vector<ck_tile::index_t> cuq_cum, cukv_cum;
    if(mode == mode_enum::batch)
    {
        auto calculate_cumulative = [&](std::vector<ck_tile::index_t>& per_batch_vec,
                                        std::vector<ck_tile::index_t>& cum_vec) {
            if(!per_batch_vec.empty() && per_batch_vec[0] != -1)
            {
                if(per_batch_vec.size() < static_cast<size_t>(batch))
                {
                    per_batch_vec.resize(batch, per_batch_vec.back());
                }
                cum_vec.resize(batch + 1);
                cum_vec[0] = 0;
                for(int i = 0; i < batch; ++i)
                    cum_vec[i + 1] = cum_vec[i] + per_batch_vec[i];
            }
        };

        calculate_cumulative(q_eff_lens_per_batch, cuq_cum);
        calculate_cumulative(kv_eff_lens_per_batch, cukv_cum);
    }

    using TypeConfig = FmhaFwdTypeConfig<DataTypeConfig>;

    using QDataType             = typename TypeConfig::QDataType;
    using KDataType             = typename TypeConfig::KDataType;
    using VDataType             = typename TypeConfig::VDataType;
    using BiasDataType          = typename TypeConfig::BiasDataType;
    using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
    using LSEDataType           = typename TypeConfig::LSEDataType;
    using SaccDataType          = typename TypeConfig::SaccDataType;
    using SMPLComputeDataType   = typename TypeConfig::SMPLComputeDataType;
    using PDataType             = typename TypeConfig::PDataType;
    using OaccDataType          = typename TypeConfig::OaccDataType;
    using ODataType             = typename TypeConfig::ODataType;

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;
    auto max_seqlen_q =
        std::numeric_limits<int32_t>::min(); // we will use max seqlen to decide grid size
    auto max_seqlen_k = std::numeric_limits<int32_t>::min();
    {
        for(ck_tile::index_t wb = 0; wb < batch; ++wb)
        {
            const int32_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
            const int32_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

            if(max_seqlen_q < real_seqlen_q)
            {
                max_seqlen_q = real_seqlen_q;
            }

            if(max_seqlen_k < real_seqlen_k)
            {
                max_seqlen_k = real_seqlen_k;
            }

            flop += nhead * (static_cast<std::size_t>(2) * mask.get_unmaskarea() * hdim_q +
                             static_cast<std::size_t>(2) * mask.get_unmaskarea() * hdim_v);

            num_byte += nhead * (sizeof(QDataType) * real_seqlen_q * hdim_q +
                                 sizeof(ODataType) * real_seqlen_q * hdim_v);
            num_byte += nhead_k * (sizeof(KDataType) * real_seqlen_k * hdim_q +
                                   sizeof(VDataType) * hdim_v * real_seqlen_k);
        }
    }

    const ck_tile::index_t max_num_page_blocks =
        (0 < page_block_size
             ? batch * std::max(1, ck_tile::integer_divide_ceil(max_seqlen_k, page_block_size))
             : 0);

    // legalize num_splits according to other options
    if(num_splits < 1)
    {
        num_splits = override_num_splits_if_necessary(
            batch, nhead, max_seqlen_q, hdim_v, p_drop, num_splits);
    }
    if(128 < num_splits)
    {
        std::cerr << "num_splits greater than 128 is not supported" << std::endl;
        return fwd_result::invalid_args;
    }
#if CK_TILE_FMHA_FWD_SPLITKV_API || CK_TILE_FMHA_FWD_PAGEDKV_API
    if(0 < p_drop && (1 < num_splits || use_kvcache))
    {
        std::cerr << "dropout is not supported by split-kv kernels. ignoring the 'p_drop' option"
                  << std::endl;
        p_drop = 0.0f;
    }
#endif

    static const auto get_lengths = [](bool permute,
                                       ck_tile::index_t b /*batch*/,
                                       ck_tile::index_t h /*nhead*/,
                                       ck_tile::index_t s /*seqlen*/,
                                       ck_tile::index_t d /*hdim*/) {
        if(permute)
            return std::array<ck_tile::index_t, 4>{b, h, s, d};
        else
            return std::array<ck_tile::index_t, 4>{b, s, h, d};
    };

    // host memory for storing all the tensor elements
    const ck_tile::index_t shape_batch = (mode == mode_enum::batch ? batch : 1);
    // physical(padded) total seqlen_q for group when s_qpad is provided; else use logical
    const ck_tile::index_t shape_seqlen_q =
        (mode == mode_enum::batch ? seqlen_qs[0]
                                  : (has_group_q_padding && !seqstart_q_with_padding_host.empty()
                                         ? seqstart_q_with_padding_host.back()
                                         : seqstart_q_host.back()));
    const ck_tile::index_t shape_seqlen_k =
        (mode == mode_enum::batch ? seqlen_ks[0]
                                  : (has_group_k_padding && !seqstart_k_with_padding_host.empty()
                                         ? seqstart_k_with_padding_host.back()
                                         : seqstart_k_host.back()));

    ck_tile::HostTensor<QDataType> q_host(
        get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    ck_tile::HostTensor<KDataType> k_host(
        0 < page_block_size
            ? get_lengths(i_perm, max_num_page_blocks, nhead_k, page_block_size, hdim_q)
            : get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_q));
    /// NOTICE: always use same shape for knew_host & vnew_host in batch/group mode
    ck_tile::HostTensor<KDataType> knew_host(
        0 < seqlen_knew
            ? get_lengths(i_perm, batch, nhead_k, seqlen_knew, hdim_q)
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    ck_tile::HostTensor<VDataType> v_host(
        0 < page_block_size
            ? (is_v_rowmajor
                   ? get_lengths(i_perm, max_num_page_blocks, nhead_k, page_block_size, hdim_v)
                   : get_lengths(i_perm, max_num_page_blocks, nhead_k, hdim_v, page_block_size))
            : (is_v_rowmajor ? get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_v)
                             : get_lengths(i_perm, shape_batch, nhead_k, hdim_v, shape_seqlen_k)));
    ck_tile::HostTensor<VDataType> vnew_host(
        0 < seqlen_knew
            ? (is_v_rowmajor ? get_lengths(i_perm, batch, nhead_k, seqlen_knew, hdim_v)
                             : get_lengths(i_perm, batch, nhead_k, hdim_v, seqlen_knew))
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    ck_tile::HostTensor<BiasDataType> bias_host(
        bias.type == bias_enum::elementwise_bias
            ? get_lengths(i_perm, 1, 1, shape_seqlen_q, max_seqlen_k)
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);

    ck_tile::HostTensor<SaccDataType> alibi_slope_host(
        bias.type == bias_enum::alibi
            ? (bias.rank_info == 0 ? std::array<ck_tile::index_t, 2>{1, nhead}
                                   : std::array<ck_tile::index_t, 2>{batch, nhead})
            : std::array<ck_tile::index_t, 2>{1, 1});

    auto [rotary_cos_host, rotary_sin_host] = generate_rotary_cos_sin<KDataType>(
        std::max(shape_seqlen_q, shape_seqlen_k), rotary_dim, next_seed());

    ck_tile::HostTensor<LSEDataType> lse_acc_host(
        1 < num_splits || use_kvcache
            ? std::array<ck_tile::index_t, 4>{shape_batch, nhead, num_splits, shape_seqlen_q}
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1});
    ck_tile::HostTensor<OaccDataType> o_acc_host(
        1 < num_splits || use_kvcache ? std::array<ck_tile::index_t, 5>{shape_batch,
                                                                        nhead,
                                                                        num_splits,
                                                                        shape_seqlen_q,
                                                                        hdim_v}
                                      : std::array<ck_tile::index_t, 5>{1, 1, 1, 1, 1});

    // TODO - change the tensor length for different quant scale
    ck_tile::HostTensor<float> q_descale_host(get_lengths(i_perm, 1, 1, 1, 1));
    ck_tile::HostTensor<float> k_descale_host(get_lengths(i_perm, 1, 1, 1, 1));
    ck_tile::HostTensor<float> v_descale_host(get_lengths(i_perm, 1, 1, 1, 1));

    // batch mode of lse data layout is [batch, nhead, seqlen_q]
    // group mode of lse data layout is [nhead, total_seqlen_q]
    ck_tile::HostTensor<LSEDataType> lse_host(
        lse ? std::array<ck_tile::index_t, 3>{shape_batch, nhead, shape_seqlen_q}
            : std::array<ck_tile::index_t, 3>{1, 1, 1} /* dummy shape for simplifying code */);

    ck_tile::HostTensor<ODataType> o_host(
        get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));

    ck_tile::HostTensor<RandValOutputDataType> randval_host(
        p_drop > 0 ? get_lengths(true, shape_batch, nhead, shape_seqlen_q, max_seqlen_k)
                   : std::array<ck_tile::index_t, 4>{1, 1, 1, 1});

    ck_tile::HostTensor<int32_t> block_table_host(
        0 < page_block_size ? std::array<ck_tile::index_t, 2>{batch, max_num_page_blocks / batch}
                            : std::array<ck_tile::index_t, 2>{1, 1});

    ck_tile::HostTensor<int32_t> cache_batch_idx_host(use_cache_batch_idx
                                                          ? std::array<ck_tile::index_t, 1>{batch}
                                                          : std::array<ck_tile::index_t, 1>{1});
    if(init_method == "ui" || init_method == "0")
    {
        ck_tile::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f, next_seed()}(q_host);
        ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, next_seed()}(k_host);
        ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, next_seed()}(knew_host);
        ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, next_seed()}(v_host);
        ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, next_seed()}(vnew_host);
        ck_tile::FillUniformDistributionIntegerValue<BiasDataType>{-3.f, 3.f, next_seed()}(
            bias_host);
    }
    else if(init_method == "ni")
    {
        ck_tile::FillNormalDistributionIntegerValue<QDataType>{-3.f, 3.f, next_seed()}(q_host);
        ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, next_seed()}(k_host);
        ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, next_seed()}(knew_host);
        ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, next_seed()}(v_host);
        ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, next_seed()}(vnew_host);
        ck_tile::FillNormalDistributionIntegerValue<BiasDataType>{-3.f, 3.f, next_seed()}(
            bias_host);
    }
    else if(init_method == "uf" || init_method == "1")
    {
        ck_tile::FillUniformDistribution<QDataType>{0.f, 1.f, next_seed()}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, next_seed()}(k_host);
        ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, next_seed()}(knew_host);
        ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, next_seed()}(v_host);
        ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, next_seed()}(vnew_host);
        ck_tile::FillUniformDistribution<BiasDataType>{0.f, 1.f, next_seed()}(bias_host);
    }
    else if(init_method == "nf")
    {
        ck_tile::FillNormalDistribution<QDataType>{0.f, 3.f, next_seed()}(q_host);
        ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, next_seed()}(k_host);
        ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, next_seed()}(knew_host);
        ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, next_seed()}(v_host);
        ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, next_seed()}(vnew_host);
        ck_tile::FillNormalDistribution<BiasDataType>{0.f, 3.f, next_seed()}(bias_host);
    }
    else if(init_method == "tf" || init_method == "2")
    {
        ck_tile::FillTrigValue<QDataType>{}(q_host);
        ck_tile::FillTrigValue<KDataType>{}(k_host);
        ck_tile::FillTrigValue<KDataType>{}(knew_host);
        ck_tile::FillTrigValue<VDataType>{}(v_host);
        ck_tile::FillTrigValue<VDataType>{}(vnew_host);
        ck_tile::FillTrigValue<BiasDataType>{}(bias_host);
    }
    else if(init_method == "3")
    {
        float q_dtype_max    = ck_tile::type_convert<float>(ck_tile::numeric<QDataType>::max());
        float k_dtype_max    = ck_tile::type_convert<float>(ck_tile::numeric<KDataType>::max());
        float v_dtype_max    = ck_tile::type_convert<float>(ck_tile::numeric<VDataType>::max());
        float bias_dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<BiasDataType>::max());

        ck_tile::FillUniformDistribution<QDataType>{-q_dtype_max, q_dtype_max, next_seed()}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{-k_dtype_max, k_dtype_max, next_seed()}(k_host);
        ck_tile::FillUniformDistribution<KDataType>{-k_dtype_max, k_dtype_max, next_seed()}(
            knew_host);
        ck_tile::FillUniformDistribution<VDataType>{-v_dtype_max, v_dtype_max, next_seed()}(v_host);
        ck_tile::FillUniformDistribution<VDataType>{-v_dtype_max, v_dtype_max, next_seed()}(
            vnew_host);
        ck_tile::FillUniformDistribution<BiasDataType>{
            -bias_dtype_max, bias_dtype_max, next_seed()}(bias_host);
    }
    if(bias.type == bias_enum::alibi)
    {
        auto slopes = ck_tile::get_alibi_slopes<SaccDataType>(nhead);
        assert(slopes.size() == static_cast<std::size_t>(nhead));
        if(bias.rank_info == 0)
        {
            // alibi in 1*h
            std::copy(slopes.begin(), slopes.end(), alibi_slope_host.begin());
        }
        else
        {
            // alibi in b*h
            for(auto i_b = 0; i_b < batch; i_b++)
            {
                std::copy(slopes.begin(), slopes.end(), alibi_slope_host.begin() + i_b * nhead);
            }
        }
    }
    if(qscale.type == quant_scale_enum::pertensor)
    {
        float q_dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<QDataType>::max());
        float k_dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<KDataType>::max());
        float v_dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<VDataType>::max());

        float qkv_max     = 3.f;
        q_descale_host(0) = qkv_max / q_dtype_max;
        k_descale_host(0) = qkv_max / k_dtype_max;
        v_descale_host(0) = qkv_max / v_dtype_max;
    }

    iota_shuffle(block_table_host.begin(), block_table_host.end(), 0, random_engine);
    iota_shuffle(cache_batch_idx_host.begin(), cache_batch_idx_host.end(), 0, random_engine);

    ck_tile::DeviceMem q_buf(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem knew_buf(knew_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem vnew_buf(vnew_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem bias_buf(bias_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem q_descale_buf(q_descale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_descale_buf(k_descale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_descale_buf(v_descale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_acc_buf(lse_acc_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_acc_buf(o_acc_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_buf(lse_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqstart_q_padded_buf(seqstart_q_with_padding_host.empty()
                                                 ? 0
                                                 : seqstart_q_with_padding_host.size() *
                                                       sizeof(int32_t));
    ck_tile::DeviceMem seqstart_k_padded_buf(
        seqlen_kpads[0] < 0 ? 0 : seqstart_k_with_padding_host.size() * sizeof(int32_t));
    // Buffers for query per-sequence logical (unpadded) lengths (used in group mode with padding
    // enabled)
    ck_tile::DeviceMem seqlen_q_buf(has_group_q_padding ? seqlen_qs.size() * sizeof(int32_t) : 0);
    // Buffers for key/value per-sequence logical (unpadded) lengths (used in batch mode with
    // kvcache or group mode with padding enabled)
    ck_tile::DeviceMem seqlen_k_buf((mode == mode_enum::batch && use_kvcache) || has_group_k_padding
                                        ? seqlen_ks.size() * sizeof(int32_t)
                                        : 0);
    ck_tile::DeviceMem cu_seqlen_q_buf(cuq_cum.empty() ? 0
                                                       : cuq_cum.size() * sizeof(ck_tile::index_t));
    ck_tile::DeviceMem cu_seqlen_kv_buf(
        cukv_cum.empty() ? 0 : cukv_cum.size() * sizeof(ck_tile::index_t));
    ck_tile::DeviceMem cache_seqlen_k_buf(
        need_append_kvcache ? cache_seqlen_ks.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem rotary_cos_buf(rotary_cos_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem rotary_sin_buf(rotary_sin_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem drop_seed_buf(drop_prefs ? sizeof(uint64_t) : 0);
    ck_tile::DeviceMem drop_offset_buf(drop_prefs ? sizeof(uint64_t) : 0);
    ck_tile::DeviceMem randval_buf(randval_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem alibi_slope_buf(alibi_slope_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem block_table_buf(block_table_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem cache_batch_idx_buf(cache_batch_idx_host.get_element_space_size_in_bytes());

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    knew_buf.ToDevice(knew_host.data());
    vnew_buf.ToDevice(vnew_host.data());
    bias_buf.ToDevice(bias_host.data());
    q_descale_buf.ToDevice(q_descale_host.data());
    k_descale_buf.ToDevice(k_descale_host.data());
    v_descale_buf.ToDevice(v_descale_host.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    // Keep logical starts in seqstart_k; pass padded K via separate pointer
    seqstart_k.ToDevice(seqstart_k_host.data());
    seqstart_q_padded_buf.ToDevice(
        seqstart_q_with_padding_host.empty() ? nullptr : seqstart_q_with_padding_host.data());
    seqstart_k_padded_buf.ToDevice(seqlen_kpads[0] < 0 ? nullptr
                                                       : seqstart_k_with_padding_host.data());
    cu_seqlen_q_buf.ToDevice(cuq_cum.empty() ? nullptr : cuq_cum.data());
    cu_seqlen_kv_buf.ToDevice(cukv_cum.empty() ? nullptr : cukv_cum.data());
    seqlen_q_buf.ToDevice(has_group_q_padding ? seqlen_qs.data() : nullptr);
    seqlen_k_buf.ToDevice((mode == mode_enum::batch && use_kvcache) || has_group_k_padding
                              ? seqlen_ks.data()
                              : nullptr);
    cache_seqlen_k_buf.ToDevice(need_append_kvcache ? cache_seqlen_ks.data() : nullptr);
    rotary_cos_buf.ToDevice(rotary_cos_host.data());
    rotary_sin_buf.ToDevice(rotary_sin_host.data());
    drop_seed_buf.ToDevice(drop_prefs ? &drop_seed : nullptr);
    drop_offset_buf.ToDevice(drop_prefs ? &drop_offset : nullptr);
    alibi_slope_buf.ToDevice(alibi_slope_host.data());
    block_table_buf.ToDevice(block_table_host.data());
    cache_batch_idx_buf.ToDevice(cache_batch_idx_host.data());

    // clang-format off
    auto layout_str = [&](bool permute){
        if(permute) return std::string("bhsd");
        else return std::string("bshd");
    };
    auto io_layout = [&](bool iperm_, bool operm_) {
        if(iperm_ == operm_) return layout_str(iperm_);
        else return layout_str(iperm_) + std::string("-") + layout_str(operm_);
    };
    // clang-format on

    std::cout << "[" << data_type << "|" << mode << "|" << io_layout(i_perm, o_perm)
              << "] b:" << batch << ", h:" << nhead << "/" << nhead_k << ", s:" << seqlen_qs[0]
              << "/" << seqlen_ks[0]
              << (seqlen_kpads[0] < 0 ? ""
                                      : (std::string("(") + std::to_string(seqlen_kpads[0]) + ")"))
              << ", d:" << hdim_q << "/" << hdim_v << ", scale_s:" << scale_s << ", bias:" << bias
              << ", p_drop:" << p_drop << ", lse:" << lse << ", qscale:" << qscale
              << ", mask:" << mask << ", v:" << (is_v_rowmajor ? "r" : "c");
#if CK_TILE_FMHA_FWD_APPENDKV_API
    if(0 < rotary_dim)
    {
        std::cout << ", rotary_dim:" << rotary_dim << "("
                  << (is_rotary_interleaved ? "inter" : "half") << ")";
    }
#endif
#if CK_TILE_FMHA_FWD_SPLITKV_API || CK_TILE_FMHA_FWD_PAGEDKV_API
    if(1 < num_splits)
    {
        std::cout << ", num_splits:" << num_splits;
    }
    if(0 < page_block_size)
    {
        std::cout << ", page_block_size:" << page_block_size;
    }
    if(use_cache_batch_idx)
    {
        std::cout << ", cache_batch_idx:" << use_cache_batch_idx;
    }
#endif
    // Padding / effective length diagnostic logging
    auto print_vec = [&](const char* label, const std::vector<int>& v) {
        if(v.empty())
            return;
        std::cout << ", " << label << ":[";
        for(std::size_t i = 0; i < v.size(); ++i)
        {
            if(i)
                std::cout << ",";
            std::cout << v[i];
        }
        std::cout << "]";
    };

    if(has_group_padding)
    {
        bool has_qpad = !seqstart_q_with_padding_host.empty();
        bool has_kpad = (seqlen_kpads[0] >= 0);
        if(has_qpad)
        {
            print_vec("q_logical", seqlen_qs);
            print_vec("q_padded", seqlen_qpads);
        }
        if(has_kpad)
        {
            print_vec("k_logical", seqlen_ks);
            print_vec("k_padded", seqlen_kpads);
        }
    }
    else if(has_batch_padding)
    {
        // derive effective lengths from cumulative arrays if present
        if(!cuq_cum.empty())
        {
            std::vector<int> eff_q(batch);
            for(int b_i = 0; b_i < batch; ++b_i)
                eff_q[b_i] = static_cast<int>(cuq_cum[b_i + 1] - cuq_cum[b_i]);
            print_vec("q_eff", eff_q);
        }
        if(!cukv_cum.empty())
        {
            std::vector<int> eff_kv(batch);
            for(int b_i = 0; b_i < batch; ++b_i)
                eff_kv[b_i] = static_cast<int>(cukv_cum[b_i + 1] - cukv_cum[b_i]);
            print_vec("kv_eff", eff_kv);
        }
    }

    std::cout << std::flush;

    const auto init_traits = [&](auto& traits) {
        traits.hdim_q        = hdim_q;
        traits.hdim_v        = hdim_v;
        traits.data_type     = data_type;
        traits.is_v_rowmajor = is_v_rowmajor;

        if constexpr(std::is_same_v<fmha_fwd_appendkv_traits, std::decay_t<decltype(traits)>>)
        {
            traits.rope_type = (0 < rotary_dim ? (is_rotary_interleaved ? rope_enum::interleaved
                                                                        : rope_enum::half_rotated)
                                               : rope_enum::none);
        }
        else // fmha_fwd_traits or fmha_splitkv_traits
        {
            traits.is_group_mode       = (mode == mode_enum::group);
            traits.has_logits_soft_cap = 0.f < logits_soft_cap;
            traits.mask_type           = mask.type;
            traits.bias_type           = bias.type;
            traits.has_sink            = mask.sink > 0 ? true : false;
            traits.has_lse             = lse;

            if constexpr(std::is_same_v<fmha_fwd_traits, std::decay_t<decltype(traits)>>)
            {
                traits.has_dropout = (p_drop > 0.0f);
                traits.qscale_type = qscale.type;
            }
            else if constexpr(std::is_same_v<fmha_fwd_pagedkv_traits,
                                             std::decay_t<decltype(traits)>>)
            {
                traits.use_pagedkv = (0 < page_block_size);
            }
        }
    };

    const auto init_args = [&, k_paddings_ = seqlen_kpads](auto& args) {
        /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
        ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
        ///       'nhead_stride_bias' are 0.
        // setup stride_* arguments
        const ck_tile::index_t stride_q    = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k    = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_knew = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v    = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return 0 < page_block_size ? (i_perm ? page_block_size : nhead_k * page_block_size)
                                              : (i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k);
        }();
        const ck_tile::index_t stride_vnew = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return i_perm ? seqlen_knew : nhead_k * seqlen_knew;
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? max_seqlen_k : 1 * max_seqlen_k);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_o_acc   = (hdim_v);
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k =
            (0 < page_block_size ? (i_perm ? page_block_size * hdim_q : hdim_q)
                                 : (i_perm ? shape_seqlen_k * hdim_q : hdim_q));
        const ck_tile::index_t nhead_stride_knew = (i_perm ? seqlen_knew * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v    = [&]() {
            if(is_v_rowmajor)
                return 0 < page_block_size ? (i_perm ? page_block_size * hdim_v : hdim_v)
                                              : (i_perm ? shape_seqlen_k * hdim_v : hdim_v);
            else
                return 0 < page_block_size ? (i_perm ? hdim_v * page_block_size : page_block_size)
                                              : (i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k);
        }();
        const ck_tile::index_t nhead_stride_vnew = [&]() {
            if(is_v_rowmajor)
                return i_perm ? seqlen_knew * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * seqlen_knew : seqlen_knew;
        }();
        const ck_tile::index_t nhead_stride_bias =
            (i_perm ? 0 * shape_seqlen_q * max_seqlen_k : 0 * max_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_lse_acc = (num_splits * shape_seqlen_q);
        const ck_tile::index_t nhead_stride_o_acc   = (num_splits * shape_seqlen_q * hdim_v);
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k =
            (0 < page_block_size ? (nhead_k * page_block_size * hdim_q)
                                 : (nhead_k * shape_seqlen_k * hdim_q));
        const ck_tile::index_t batch_stride_knew = (nhead_k * seqlen_knew * hdim_q);
        const ck_tile::index_t batch_stride_v =
            (0 < page_block_size ? (nhead_k * hdim_v * page_block_size)
                                 : (nhead_k * hdim_v * shape_seqlen_k));
        const ck_tile::index_t batch_stride_vnew    = (nhead_k * hdim_v * seqlen_knew);
        const ck_tile::index_t batch_stride_bias    = (0 * nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_lse_acc = (nhead * num_splits * shape_seqlen_q);
        const ck_tile::index_t batch_stride_o_acc = (nhead * num_splits * shape_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_o     = (nhead * shape_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_block_table = (max_num_page_blocks / batch);
        // setup split_stride_* arguments (only used in split-kv kernel)
        const ck_tile::index_t split_stride_lse_acc = (shape_seqlen_q);
        const ck_tile::index_t split_stride_o_acc   = (shape_seqlen_q * hdim_v);

        args.q_ptr = q_buf.GetDeviceBuffer();
        args.k_ptr = k_buf.GetDeviceBuffer();
        args.v_ptr = v_buf.GetDeviceBuffer();

        args.batch    = batch;
        args.seqlen_q = shape_seqlen_q; // unused in group mode
        args.hdim_q   = hdim_q;
        args.hdim_v   = hdim_v;
        args.nhead_q  = nhead;
        args.nhead_k  = nhead_k;

        args.stride_q       = stride_q;
        args.stride_k       = stride_k;
        args.stride_v       = stride_v;
        args.nhead_stride_q = nhead_stride_q;
        args.nhead_stride_k = nhead_stride_k;
        args.nhead_stride_v = nhead_stride_v;
        args.batch_stride_q = batch_stride_q;
        args.batch_stride_k = batch_stride_k;
        args.batch_stride_v = batch_stride_v;

        if constexpr(std::is_same_v<fmha_fwd_appendkv_args, std::decay_t<decltype(args)>>)
        {
            args.knew_ptr    = knew_buf.GetDeviceBuffer();
            args.vnew_ptr    = vnew_buf.GetDeviceBuffer();
            args.seqlen_knew = seqlen_knew;

            args.seqlen_k_ptr = cache_seqlen_k_buf.GetDeviceBuffer();

            args.rotary_cos_ptr = (0 < rotary_dim ? rotary_cos_buf.GetDeviceBuffer() : nullptr);
            args.rotary_sin_ptr = (0 < rotary_dim ? rotary_sin_buf.GetDeviceBuffer() : nullptr);
            args.rotary_dim     = rotary_dim;
            args.has_mask       = (mask.type != mask_enum::no_mask);

            args.block_table_ptr =
                (0 < page_block_size ? block_table_buf.GetDeviceBuffer() : nullptr);
            args.batch_stride_block_table = batch_stride_block_table;
            args.page_block_size          = page_block_size;

            args.cache_batch_idx =
                (use_cache_batch_idx ? cache_batch_idx_buf.GetDeviceBuffer() : nullptr);

            args.stride_knew       = stride_knew;
            args.stride_vnew       = stride_vnew;
            args.nhead_stride_knew = nhead_stride_knew;
            args.nhead_stride_vnew = nhead_stride_vnew;
            args.batch_stride_knew = batch_stride_knew;
            args.batch_stride_vnew = batch_stride_vnew;
        }
        else // fmha_fwd_args or fmha_fwd_splitkv_args
        {
            args.bias_ptr = bias.type == bias_enum::alibi ? alibi_slope_buf.GetDeviceBuffer()
                                                          : bias_buf.GetDeviceBuffer();
            args.lse_ptr  = lse_buf.GetDeviceBuffer();
            args.o_ptr    = o_buf.GetDeviceBuffer();

            args.seqlen_k     = shape_seqlen_k; // unused in group mode (or kvcache enabled)
            args.max_seqlen_q = max_seqlen_q;

            args.scale_s = scale_s;

            args.logits_soft_cap = logits_soft_cap;

            args.stride_bias =
                (bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias);
            args.stride_o          = stride_o;
            args.nhead_stride_bias = nhead_stride_bias;
            args.nhead_stride_lse  = nhead_stride_lse;
            args.nhead_stride_o    = nhead_stride_o;
            args.batch_stride_bias = batch_stride_bias;
            args.batch_stride_lse  = batch_stride_lse;
            args.batch_stride_o    = batch_stride_o;

            args.window_size_left  = mask.left;
            args.window_size_right = mask.right;
            args.sink_size         = mask.sink;
            args.mask_type         = static_cast<ck_tile::index_t>(mask.type);

            if constexpr(std::is_same_v<fmha_fwd_args, std::decay_t<decltype(args)>>)
            {
                args.q_descale_ptr = q_descale_buf.GetDeviceBuffer();
                args.k_descale_ptr = k_descale_buf.GetDeviceBuffer();
                args.v_descale_ptr = v_descale_buf.GetDeviceBuffer();

                args.rand_val_ptr = randval_buf.GetDeviceBuffer();

                args.stride_randval       = stride_randval;
                args.nhead_stride_randval = nhead_stride_randval;
                args.batch_stride_randval = batch_stride_randval;

                args.p_drop    = p_drop;
                args.s_randval = s_randval;
                if(drop_prefs)
                {
                    args.drop_seed_offset = std::make_pair(drop_seed_buf.GetDeviceBuffer(),
                                                           drop_offset_buf.GetDeviceBuffer());
                }
                else
                {
                    args.drop_seed_offset = std::make_pair(drop_seed, drop_offset);
                }

                // Sequence length and padding parameters (mode-specific)
                if(mode == mode_enum::group)
                {
                    // Group mode: use physical (padded) cumulative starts + logical per-sequence
                    // lengths

                    // Physical cumulative starts (including padding)
                    args.seqstart_q_ptr =
                        has_group_q_padding && !seqstart_q_with_padding_host.empty()
                            ? seqstart_q_padded_buf.GetDeviceBuffer()
                            : seqstart_q.GetDeviceBuffer();
                    args.seqstart_k_ptr =
                        has_group_k_padding && !seqstart_k_with_padding_host.empty()
                            ? seqstart_k_padded_buf.GetDeviceBuffer()
                            : seqstart_k.GetDeviceBuffer();

                    // Logical (unpadded) per-sequence lengths, used when padding is enabled
                    args.seqlen_q_ptr =
                        (has_group_q_padding && !seqstart_q_with_padding_host.empty())
                            ? seqlen_q_buf.GetDeviceBuffer()
                            : nullptr;
                    args.seqlen_k_ptr =
                        (has_group_k_padding && !seqstart_k_with_padding_host.empty())
                            ? seqlen_k_buf.GetDeviceBuffer()
                            : nullptr;
                    // Cumulative lengths not used in group mode
                    args.cu_seqlen_q_ptr = nullptr;
                    args.cu_seqlen_k_ptr = nullptr;
                }
                else // mode == mode_enum::batch
                {
                    // Batch mode: use cumulative logical lengths for tail padding

                    // seqstart pointers not used in batch mode
                    args.seqstart_q_ptr = nullptr;
                    args.seqstart_k_ptr = nullptr;

                    // seqlen_q_ptr/seqlen_k_ptr not used in batch mode
                    args.seqlen_q_ptr = nullptr;
                    args.seqlen_k_ptr = nullptr;

                    // Cumulative logical lengths for effective length handling
                    args.cu_seqlen_q_ptr = has_batch_q_padding && !cuq_cum.empty()
                                               ? cu_seqlen_q_buf.GetDeviceBuffer()
                                               : nullptr;
                    args.cu_seqlen_k_ptr = has_batch_k_padding && !cukv_cum.empty()
                                               ? cu_seqlen_kv_buf.GetDeviceBuffer()
                                               : nullptr;
                }
            }
            else if constexpr(std::is_same_v<fmha_fwd_splitkv_args, std::decay_t<decltype(args)>>)
            {
                args.lse_acc_ptr = lse_acc_buf.GetDeviceBuffer();
                args.o_acc_ptr   = o_acc_buf.GetDeviceBuffer();

                args.block_table_ptr =
                    (0 < page_block_size ? block_table_buf.GetDeviceBuffer() : nullptr);
                args.batch_stride_block_table = batch_stride_block_table;
                args.page_block_size          = page_block_size;
                args.is_gappy = false; // use 'false' for flash-attention integration

                args.cache_batch_idx =
                    (use_cache_batch_idx ? cache_batch_idx_buf.GetDeviceBuffer() : nullptr);

                args.num_splits = num_splits;

                args.stride_o_acc         = stride_o_acc;
                args.nhead_stride_lse_acc = nhead_stride_lse_acc;
                args.nhead_stride_o_acc   = nhead_stride_o_acc;
                args.batch_stride_lse_acc = batch_stride_lse_acc;
                args.batch_stride_o_acc   = batch_stride_o_acc;
                args.split_stride_lse_acc = split_stride_lse_acc;
                args.split_stride_o_acc   = split_stride_o_acc;

                args.seqstart_q_ptr =
                    (mode == mode_enum::group ? seqstart_q.GetDeviceBuffer() : nullptr);
                args.seqstart_k_ptr =
                    (mode == mode_enum::group ? seqstart_k.GetDeviceBuffer() : nullptr);
                args.seqlen_k_ptr =
                    ((mode == mode_enum::batch && use_kvcache) || 0 <= k_paddings_[0]
                         ? seqlen_k_buf.GetDeviceBuffer()
                         : nullptr);
            }
            else if constexpr(std::is_same_v<fmha_fwd_pagedkv_args, std::decay_t<decltype(args)>>)
            {
                args.block_table_ptr =
                    (0 < page_block_size ? block_table_buf.GetDeviceBuffer() : nullptr);
                args.batch_stride_block_table = batch_stride_block_table;
                args.page_block_size          = page_block_size;
                args.is_gappy = false; // use 'false' for flash-attention integration

                args.cache_batch_idx =
                    (use_cache_batch_idx ? cache_batch_idx_buf.GetDeviceBuffer() : nullptr);

                args.seqstart_q_ptr =
                    (mode == mode_enum::group ? seqstart_q.GetDeviceBuffer() : nullptr);
                args.seqstart_k_ptr =
                    (mode == mode_enum::group ? seqstart_k.GetDeviceBuffer() : nullptr);
                args.seqlen_k_ptr =
                    ((mode == mode_enum::batch && use_kvcache) || 0 <= k_paddings_[0]
                         ? seqlen_k_buf.GetDeviceBuffer()
                         : nullptr);
            }
        }
    };

    auto run_appendkv = [&]([[maybe_unused]] const ck_tile::stream_config& sc) {
#if CK_TILE_FMHA_FWD_APPENDKV_API
        if(need_append_kvcache)
        {
            fmha_fwd_appendkv_traits fwd_appendkv_traits;
            init_traits(fwd_appendkv_traits);

            fmha_fwd_appendkv_args fwd_appendkv_args;
            init_args(fwd_appendkv_args);

            return fmha_fwd_appendkv(fwd_appendkv_traits, fwd_appendkv_args, sc);
        }
#endif
        return 0.0f;
    };
    const float appendkv_ave_time = run_appendkv(stream_config);
    if(appendkv_ave_time < 0.0f)
    {
        std::cout << ", not supported yet" << std::flush << std::endl;
        return fwd_result::no_instance;
    }

    auto run_fwd = [&](const ck_tile::stream_config& sc) {
#if CK_TILE_FMHA_FWD_PAGEDKV_API
        if(1 == num_splits && use_kvcache)
        {
            fmha_fwd_pagedkv_traits fmha_pagedkv_traits;
            init_traits(fmha_pagedkv_traits);

            fmha_fwd_pagedkv_args fmha_pagedkv_args;
            init_args(fmha_pagedkv_args);

            const float ave_time = fmha_fwd_pagedkv(fmha_pagedkv_traits, fmha_pagedkv_args, sc);
#if CK_TILE_FMHA_FWD_SPLITKV_API
            // If there is no instance for these args, fallback to fmha_fwd_splitkv
            if(ave_time >= 0.0f)
                return ave_time;
#else
            return ave_time;
#endif
        }
#endif // CK_TILE_FMHA_FWD_PAGEDKV_API
#if CK_TILE_FMHA_FWD_SPLITKV_API
        if(1 < num_splits || use_kvcache)
        {
            fmha_fwd_splitkv_traits fmha_splitkv_traits;
            init_traits(fmha_splitkv_traits);

            fmha_fwd_splitkv_args fmha_splitkv_args;
            init_args(fmha_splitkv_args);

            return fmha_fwd_splitkv(fmha_splitkv_traits, fmha_splitkv_args, sc);
        }
#endif // CK_TILE_FMHA_FWD_SPLITKV_API
        fmha_fwd_traits fmha_traits;
        init_traits(fmha_traits);

        fmha_fwd_args fmha_args;
        init_args(fmha_args);

        return fmha_fwd(fmha_traits, fmha_args, sc);
    };
    const float fwd_ave_time = run_fwd(stream_config);
    if(fwd_ave_time < 0.0f)
    {
        std::cout << ", not supported yet" << std::flush << std::endl;
        return fwd_result::no_instance;
    }

    const float ave_time   = appendkv_ave_time + fwd_ave_time;
    const float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    const float gb_per_sec = num_byte / 1.E6 / ave_time;
    if(stream_config.time_kernel_)
    {
        std::cout << std::fixed << ", " << std::setprecision(3) << ave_time << " ms, "
                  << std::setprecision(2) << tflops << " TFlops, " << std::setprecision(2)
                  << gb_per_sec << " GB/s" << std::flush;
    }

    bool pass = true;
    if(do_validation == 0)
    {
        std::cout << std::flush << std::endl;
    }
    else if(do_validation == 2)
    {
        // NOTE: use gpu to do validation
        ck_tile::naive_attention_fwd_traits naive_t;
        naive_t.q_type     = data_type;
        naive_t.k_type     = data_type;
        naive_t.v_type     = data_type;
        naive_t.o_type     = data_type;
        naive_t.q_layout   = i_perm == 1 ? "bhsd" : "bshd";
        naive_t.k_layout   = i_perm == 1 ? "bhsd" : "bshd";
        naive_t.v_layout   = i_perm == 1 ? "bhsd" : "bshd";
        naive_t.o_layout   = o_perm == 1 ? "bhsd" : "bshd";
        naive_t.variation  = 0; // TODO?
        naive_t.quant_algo = 0;

        ck_tile::DeviceMem o_naive_buf(o_host.get_element_space_size_in_bytes());

        ck_tile::naive_attention_fwd_args naive_a;
        naive_a.q_ptr           = q_buf.GetDeviceBuffer();
        naive_a.k_ptr           = k_buf.GetDeviceBuffer();
        naive_a.v_ptr           = v_buf.GetDeviceBuffer();
        naive_a.o_ptr           = o_naive_buf.GetDeviceBuffer();
        naive_a.scale_s         = scale_s;
        naive_a.context_len_ptr = nullptr; // used when seqlen kv come from a pointer
        naive_a.page_table_ptr =
            nullptr; // [batch, num_blocks] seqlen_kv is in different block(paged attn)
        naive_a.hdim           = hdim_q;
        naive_a.hdim_v         = hdim_v; // could be cross-attn, where V and Q/K hdim are different
        naive_a.batch_q        = batch;
        naive_a.batch_kv       = batch;
        naive_a.batch_ratio_kv = 1; // batch_q / batch_kv
        naive_a.seqlen_q       = seqlen_qs[0];
        naive_a.seqlen_kv = seqlen_ks[0]; // if context_len_ptr is not nullptr, ignore this field
        naive_a.nhead_q   = nhead;
        naive_a.nhead_kv  = nhead_k;
        naive_a.nhead_ratio_kv = naive_a.nhead_q / naive_a.nhead_kv; // nhead_q / nhead_kv
        naive_a.page_size      = 0; // if paged, the seqlen-kv for each block

        ck_tile::stream_config naive_s{};

        naive_attention_fwd(naive_t, naive_a, naive_s);

        auto o_naive_ref = o_naive_buf.ToHost<ODataType>();
        o_buf.FromDevice(o_host.data()); // TODO: ugly

        auto [rtol_, atol_] = get_elimit<DataTypeConfig>(init_method);
        pass                = ck_tile::check_err(
            o_host, o_naive_ref, std::string("OUT Error: Incorrect results!"), rtol_, atol_);
        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }
    else
    {
#if CK_TILE_FMHA_FWD_APPENDKV_API
        // When rotary embedding is used, the appendkv kernel modifies the q tensor (multiple times
        // when time_kernel_ is set). We need to reset the q buffer and rerun all kernels.
        if(0 < rotary_dim && stream_config.time_kernel_)
        {
            const ck_tile::stream_config stream_config2{stream_config.stream_id_, false, 0};
            q_buf.ToDevice(q_host.data());
            run_appendkv(stream_config2);
            run_fwd(stream_config2);
        }
#endif
        o_buf.FromDevice(o_host.data());
        lse_buf.FromDevice(lse_host.data());
        randval_buf.FromDevice(randval_host.data());

        constexpr bool supports_qscale = std::is_same_v<DataTypeConfig, FmhaFwdFp8> ||
                                         std::is_same_v<DataTypeConfig, FmhaFwdFp8Bf16> ||
                                         std::is_same_v<DataTypeConfig, FmhaFwdFp8Fp32>;

        float scale_s_host = scale_s;
        float scale_p_host = 1.0f;
        float scale_o_host = 1.0f;

        if(qscale.type == quant_scale_enum::pertensor)
        {
            scale_s_host = scale_s * q_descale_host(0) * k_descale_host(0);
            scale_p_host = ck_tile::type_convert<float>(ck_tile::numeric<PDataType>::max());
            scale_o_host = v_descale_host(0) / scale_p_host;
        }

        auto p_compute_element_func = [&]() {
            if constexpr(supports_qscale)
                return ck_tile::scales{scale_p_host};
            else
                return ck_tile::identity{};
        }();

        auto oacc_element_func = [&]() {
            if constexpr(std::is_same_v<ODataType, ck_tile::fp8_t> && supports_qscale)
                return ck_tile::composes(ck_tile::saturates<ck_tile::fp8_t>{},
                                         ck_tile::scales{scale_o_host});
            else if constexpr(supports_qscale)
                return ck_tile::scales{scale_o_host};
            else
                return ck_tile::identity{};
        }();

        float p_undrop = 1.0 - p_drop;
        uint8_t p_undrop_in_uint8_t =
            uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
        float rp_undrop = 1.0 / p_undrop;

        for(ck_tile::index_t wb = 0; wb < batch; ++wb)
        {
            ck_tile::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
            ck_tile::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];
            if(mode == mode_enum::batch)
            {
                if(!cuq_cum.empty())
                {
                    real_seqlen_q = cuq_cum[wb + 1] - cuq_cum[wb];
                }
                if(!cukv_cum.empty())
                {
                    real_seqlen_k = cukv_cum[wb + 1] - cukv_cum[wb];
                }
            }

            // adjust matrix index according to the mode
            const ck_tile::index_t b_idx = (mode == mode_enum::batch ? wb : 0);
            const ck_tile::index_t cache_b_idx =
                (use_cache_batch_idx ? cache_batch_idx_host(b_idx) : b_idx);
            // Use physical offset if padding info is valid (not -1) and buffers are available
            const ck_tile::index_t query_offset =
                (mode == mode_enum::batch
                     ? 0
                     : ((seqstart_q_with_padding_host.empty() || seqlen_qpads[0] < 0)
                            ? seqstart_q_host[wb]
                            : seqstart_q_with_padding_host[wb]));
            const ck_tile::index_t key_offset =
                (mode == mode_enum::batch
                     ? 0
                     : ((seqstart_k_with_padding_host.empty() || seqlen_kpads[0] < 0)
                            ? seqstart_k_host[wb]
                            : seqstart_k_with_padding_host[wb]));

            ck_tile::HostTensor<QDataType> q_host_ref({nhead, real_seqlen_q, hdim_q});
            ck_tile::HostTensor<KDataType> k_host_ref({nhead, real_seqlen_k, hdim_q});
            ck_tile::HostTensor<VDataType> v_host_ref({nhead, hdim_v, real_seqlen_k});
            ck_tile::HostTensor<ODataType> o_host_ref({nhead, real_seqlen_q, hdim_v});

            ck_tile::HostTensor<SMPLComputeDataType> s_host_ref(
                {nhead, real_seqlen_q, real_seqlen_k});
            ck_tile::HostTensor<PDataType> p_host_ref({nhead, real_seqlen_q, real_seqlen_k});
            ck_tile::HostTensor<SMPLComputeDataType> lse_host_ref({nhead, real_seqlen_q});

            ck_tile::index_t nr = nhead / nhead_k;

            // clang-format off
            // permute
            if(i_perm) q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b_idx, i[0], i[1] + query_offset, i[2]); });
            else       q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b_idx, i[1] + query_offset, i[0], i[2]); });
                // clang-format on

#if CK_TILE_FMHA_FWD_APPENDKV_API
            // optionally apply RoPE to the q_host_ref
            if(0 < rotary_dim)
            {
                decltype(q_host_ref) q_host_ref_ro(q_host_ref.get_lengths());

                auto [rotary_cos_slice, rotary_sin_slice] = slice_rotary_cos_sin(
                    rotary_cos_host, rotary_sin_host, cache_seqlen_ks[wb], real_seqlen_q);

                ck_tile::reference_batched_rotary_position_embedding(
                    q_host_ref,
                    rotary_cos_slice,
                    rotary_sin_slice,
                    is_rotary_interleaved,
                    q_host_ref_ro,
                    /*use_1_row_sin_cos=*/mask.type == mask_enum::no_mask);

                q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host_ref_ro(i); });
            }
#endif
#if CK_TILE_FMHA_FWD_SPLITKV_API || CK_TILE_FMHA_FWD_PAGEDKV_API
            if(0 < page_block_size)
            {
                // clang-format off
                if(i_perm) k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(block_table_host(wb, i[1] / page_block_size), i[0] / nr, i[1] % page_block_size, i[2]); });
                else       k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(block_table_host(wb, i[1] / page_block_size), i[1] % page_block_size, i[0] / nr, i[2]); });
                // clang-format on
            }
            else
#endif
            {
                // clang-format off
                if(i_perm) k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(cache_b_idx, i[0] / nr, i[1] + key_offset, i[2]); });
                else       k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(cache_b_idx, i[1] + key_offset, i[0] / nr, i[2]); });
                // clang-format on
            }

#if CK_TILE_FMHA_FWD_APPENDKV_API
            // copy Knew to the end of K
            if(0 < seqlen_knew)
            {
                ck_tile::HostTensor<KDataType> knew_host_ref({nhead, seqlen_knew, hdim_q});
                // clang-format off
                if(i_perm) knew_host_ref.ForEach([&](auto& self, auto i) { self(i) = knew_host(wb, i[0] / nr, i[1], i[2]); });
                else       knew_host_ref.ForEach([&](auto& self, auto i) { self(i) = knew_host(wb, i[1], i[0] / nr, i[2]); });
                // clang-format on

                // optionally apply RoPE to the knew_host_ref
                auto* real_knew_host_ref = &knew_host_ref;
                std::optional<decltype(knew_host_ref)> knew_host_ref_ro;
                if(0 < rotary_dim)
                {
                    knew_host_ref_ro.emplace(knew_host_ref.get_lengths());

                    auto [rotary_cos_slice, rotary_sin_slice] = slice_rotary_cos_sin(
                        rotary_cos_host, rotary_sin_host, cache_seqlen_ks[wb], seqlen_knew);

                    ck_tile::reference_batched_rotary_position_embedding(knew_host_ref,
                                                                         rotary_cos_slice,
                                                                         rotary_sin_slice,
                                                                         is_rotary_interleaved,
                                                                         knew_host_ref_ro.value());

                    real_knew_host_ref = &knew_host_ref_ro.value();
                }

                (*real_knew_host_ref).ForEach([&](auto& self, auto i) {
                    k_host_ref(i[0], i[1] + cache_seqlen_ks[wb], i[2]) = self(i);
                });
            }
#endif
#if CK_TILE_FMHA_FWD_SPLITKV_API || CK_TILE_FMHA_FWD_PAGEDKV_API
            if(0 < page_block_size)
            {
                if(is_v_rowmajor)
                {
                    // clang-format off
                    if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(block_table_host(wb, i[2] / page_block_size), i[0] / nr, i[2] % page_block_size, i[1]); });
                    else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(block_table_host(wb, i[2] / page_block_size), i[2] % page_block_size, i[0] / nr, i[1]); });
                    // clang-format on
                }
                else
                {
                    // clang-format off
                    if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(block_table_host(wb, i[2] / page_block_size), i[0] / nr, i[1], i[2] % page_block_size); });
                    else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(block_table_host(wb, i[2] / page_block_size), i[1], i[0] / nr, i[2] % page_block_size); });
                    // clang-format on
                }
            }
            else
#endif
            {
                if(is_v_rowmajor)
                {
                    // clang-format off
                    //                                v_host_ref: [nhead, hdim, seq], v_host: [b, h_k, s, d]
                    if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[0] / nr, i[2] + key_offset, i[1]); });
                    //                                v_host_ref: [nhead, hdim, seq], v_host: [b, s, h_k, d]
                    else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[2] + key_offset, i[0] / nr, i[1]); });
                    // clang-format on
                }
                else
                {
                    // clang-format off
                    if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[0] / nr, i[1], i[2] + key_offset); });
                    else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[1], i[0] / nr, i[2] + key_offset); });
                    // clang-format on
                }
            }

#if CK_TILE_FMHA_FWD_APPENDKV_API
            // copy Vnew to the end of V
            if(0 < seqlen_knew)
            {
                ck_tile::HostTensor<VDataType> vnew_host_ref({nhead, hdim_v, seqlen_knew});
                if(is_v_rowmajor)
                {
                    // clang-format off
                    if(i_perm) vnew_host_ref.ForEach([&](auto& self, auto i) { self(i) = vnew_host(wb, i[0] / nr, i[2], i[1]); });
                    else       vnew_host_ref.ForEach([&](auto& self, auto i) { self(i) = vnew_host(wb, i[2], i[0] / nr, i[1]); });
                    // clang-format on
                }
                else
                {
                    // clang-format off
                    if(i_perm) vnew_host_ref.ForEach([&](auto& self, auto i) { self(i) = vnew_host(wb, i[0] / nr, i[1], i[2]); });
                    else       vnew_host_ref.ForEach([&](auto& self, auto i) { self(i) = vnew_host(wb, i[1], i[0] / nr, i[2]); });
                    // clang-format on
                }

                vnew_host_ref.ForEach([&](auto& self, auto i) {
                    v_host_ref(i[0], i[1], i[2] + cache_seqlen_ks[wb]) = self(i);
                });
            }
#endif

            // reference
            ck_tile::
                reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
                    q_host_ref,
                    k_host_ref,
                    s_host_ref,
                    ck_tile::identity{},
                    ck_tile::identity{},
                    ck_tile::scales(scale_s_host));

            if(0.f < logits_soft_cap)
            {
                ck_tile::reference_unary_elementwise<SaccDataType, SaccDataType, SaccDataType>(
                    s_host_ref, s_host_ref, [logits_soft_cap](SaccDataType logits) {
                        return ck_tile::type_convert<SaccDataType>(
                            logits_soft_cap *
                            std::tanhf(ck_tile::type_convert<float>(logits / logits_soft_cap)));
                    });
            }

            if(bias.type == bias_enum::elementwise_bias)
            {
                // elementwise bias
                ck_tile::HostTensor<BiasDataType> bias_host_ref({1, real_seqlen_q, real_seqlen_k});
                // clang-format off
                if(i_perm) bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, 0, i[1] + query_offset, i[2]); });
                else       bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, i[1] + query_offset, 0, i[2]); });
                // clang-format on

                // broadcast from [1, real_seqlen_q, real_seqlen_k] to [nhead, real_seqlen_q,
                // real_seqlen_k]
                ck_tile::reference_batched_elementwise<SMPLComputeDataType,
                                                       BiasDataType,
                                                       SMPLComputeDataType,
                                                       SMPLComputeDataType>(
                    s_host_ref, bias_host_ref, s_host_ref);
            }
            else if(bias.type == bias_enum::alibi)
            {
                // alibi construct elementwise bias to verify
                auto alibi_host = [&]() {
                    if(mask.type != mask_enum::no_mask)
                    {
                        return ck_tile::make_alibi_from_lr_mask<SaccDataType, true>(
                            0,
                            mask.left,
                            mask.right,
                            real_seqlen_q,
                            real_seqlen_k,
                            static_cast<ck_tile::GenericAttentionMaskEnum>(mask.type));
                    }
                    else
                    {
                        return ck_tile::Alibi<SaccDataType, true>{
                            0, real_seqlen_q, real_seqlen_k, ck_tile::AlibiMode::FROM_BOTTOM_RIGHT};
                    }
                }();

                ck_tile::HostTensor<SaccDataType> alibi_bias_host_ref(
                    {nhead, real_seqlen_q, real_seqlen_k});
                auto i_b_slope = bias.rank_info == 0 ? 0 : wb;
                for(auto i_h = 0; i_h < nhead; i_h++)
                {
                    SaccDataType current_slope = alibi_slope_host(i_b_slope, i_h);
                    alibi_host.slope           = alibi_host.mode == ck_tile::AlibiMode::VERTICAL
                                                     ? current_slope
                                                     : -current_slope;
                    for(auto i_r = 0; i_r < real_seqlen_q; i_r++)
                    {
                        for(auto i_c = 0; i_c < real_seqlen_k; i_c++)
                        {
                            SaccDataType pixel = 0;
                            alibi_host.update(pixel, i_r, i_c);
                            alibi_bias_host_ref(i_h, i_r, i_c) = pixel;
                        }
                    }
                }
                // [nhead, real_seqlen_q, real_seqlen_k]
                ck_tile::reference_batched_elementwise<SMPLComputeDataType,
                                                       SaccDataType,
                                                       SMPLComputeDataType,
                                                       SMPLComputeDataType>(
                    s_host_ref, alibi_bias_host_ref, s_host_ref);
            }

            if(mask.type == mask_enum::no_mask)
            {
                ck_tile::reference_batched_masking<SaccDataType>(
                    s_host_ref, FmhaMasks::NoMask{real_seqlen_q, real_seqlen_k});
            }
            else if(mask.type == mask_enum::window_generic)
            {
                ck_tile::reference_batched_masking<SaccDataType>(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                        mask.left, mask.right, mask.sink, real_seqlen_q, real_seqlen_k));
            }
            else
            {
                // if left window size is negative, means causal
                // else means generic (for current batch)
                if(mask.left < 0)
                    ck_tile::reference_batched_masking<SaccDataType>(
                        s_host_ref,
                        ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::CausalMask>(
                            mask.left,
                            mask.right,
                            mask.sink,
                            real_seqlen_q,
                            real_seqlen_k,
                            mask.type == mask_enum::mask_top_left));
                else
                    ck_tile::reference_batched_masking<SaccDataType>(
                        s_host_ref,
                        ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                            mask.left,
                            mask.right,
                            mask.sink,
                            real_seqlen_q,
                            real_seqlen_k,
                            mask.type == mask_enum::mask_top_left));
            }
            const ck_tile::HostTensor<SaccDataType> masked_s_host_ref = s_host_ref;
            if(lse)
            {
                ck_tile::
                    reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(
                        s_host_ref, p_host_ref, p_compute_element_func, lse_host_ref);
            }
            else
            {
                ck_tile::
                    reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(
                        s_host_ref, p_host_ref, p_compute_element_func);
            }

            if(p_drop > 0)
            {
                ck_tile::HostTensor<RandValOutputDataType> randval_host_ref(
                    {nhead, real_seqlen_q, real_seqlen_k});
                ck_tile::reference_batched_dropout_randval(
                    randval_host_ref, wb, drop_seed, drop_offset);
                ck_tile::reference_batched_dropout(
                    p_host_ref, randval_host_ref, p_undrop_in_uint8_t, rp_undrop);

                ck_tile::HostTensor<RandValOutputDataType> randval_host_result(
                    {nhead, real_seqlen_q, real_seqlen_k});
                randval_host_result.ForEach([&](auto& self, const auto& idx) {
                    self(idx) = randval_host(b_idx, idx[0], idx[1] + query_offset, idx[2]);
                });
                masked_s_host_ref.ForEach([&](const auto& self, const auto& idx) {
                    // Ignore all masked values in validation check
                    if(std::isinf(self(idx)))
                    {
                        randval_host_ref(idx)    = 0;
                        randval_host_result(idx) = 0;
                    }
                });
                bool cur_pass = ck_tile::check_err(randval_host_result,
                                                   randval_host_ref,
                                                   "DROPOUT RANDVAL Error: Incorrect results!");
                pass &= cur_pass;
                if(!cur_pass)
                {
                    break;
                }
            }

            ck_tile::reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(
                p_host_ref,
                v_host_ref,
                o_host_ref,
                ck_tile::identity{},
                ck_tile::identity{},
                oacc_element_func);

            ck_tile::HostTensor<ODataType> o_host_result({nhead, real_seqlen_q, hdim_v});
            // clang-format off
            // permute
            if(o_perm) o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b_idx, idx[0], idx[1] + query_offset, idx[2]); });
            else       o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b_idx, idx[1] + query_offset, idx[0], idx[2]); });
            // clang-format on

            auto [rtol, atol] = get_elimit<DataTypeConfig>(init_method);
            bool cur_pass     = ck_tile::check_err(o_host_result,
                                               o_host_ref,
                                               std::string("OUT Error: Incorrect results!"),
                                               rtol,
                                               atol);
            pass &= cur_pass;
            if(!cur_pass)
            {
                std::cerr << "OUT mismatch found at batch: " << wb << std::endl
                          << "\tseqlen_q: " << real_seqlen_q << std::endl
                          << "\tseqlen_k: " << real_seqlen_k << std::endl
                          << "\tseqstart_q (logical): " << seqstart_q_host << std::endl
                          << "\tseqstart_q (physical): " << seqstart_q_with_padding_host
                          << std::endl
                          << "\tseqstart_k (logical): " << seqstart_k_host << std::endl
                          << "\tseqstart_k (physical): " << seqstart_k_with_padding_host
                          << std::endl
                          << "\tquery_offset used: " << query_offset << std::endl
                          << "\tkey_offset used: " << key_offset << std::endl;

                break;
            }

            if(lse)
            {
                ck_tile::HostTensor<SMPLComputeDataType> lse_host_result({nhead, real_seqlen_q});
                lse_host_result.ForEach([&](auto& self, auto idx) {
                    self(idx) = lse_host(b_idx, idx[0], idx[1] + query_offset);
                });

                cur_pass = ck_tile::check_err(lse_host_result,
                                              lse_host_ref,
                                              "LSE Error: Incorrect results!",
                                              rtol,
                                              atol,
                                              /* allow_infinity_ref = */ true);

                pass &= cur_pass;
                if(!cur_pass)
                {
                    std::cerr << "LSE mismatch found at batch: " << wb << std::endl
                              << "\tseqlen_q: " << real_seqlen_q << std::endl
                              << "\tseqlen_k: " << real_seqlen_k << std::endl
                              << "\tseqstart_q: " << seqstart_q_host << std::endl
                              << "\tseqstart_k: " << seqstart_k_host << std::endl;

                    break;
                }
            }
        }

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    if(json)
    {
        dump_fmha_fwd_json_results(*json,
                                   data_type,
                                   mode == mode_enum::batch ? "batch" : "group",
                                   io_layout(i_perm, o_perm),
                                   batch,
                                   nhead,
                                   nhead_k,
                                   seqlen_qs[0],
                                   seqlen_ks[0],
                                   seqlen_kpads[0],
                                   hdim_q,
                                   hdim_v,
                                   scale_s,
                                   p_drop,
                                   lse,
                                   qscale.type == quant_scale_enum::no_scale ? "no_scale"
                                                                             : "pertensor",
                                   bias.type == bias_enum::elementwise_bias
                                       ? "elementwise_bias"
                                       : (bias.type == bias_enum::alibi ? "alibi" : "no_bias"),
                                   is_v_rowmajor ? "r" : "c",
                                   pass,
                                   ave_time,
                                   tflops,
                                   gb_per_sec);
    }

    return pass ? fwd_result::success : fwd_result::failure;
}
