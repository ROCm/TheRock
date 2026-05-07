// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "fmha_fwd.hpp"
#include "fmha_fwd_runner.hpp"

#include <string>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "0:no validation, 2:cpu validation, 2:gpu validation(experimental)")
        .insert("mode", "0", "kernel mode. 0:batch, 1:group")
        .insert("b", "2", "batch size")
        .insert("h", "8", "num of head, for q")
        .insert("h_k",
                "-1",
                "num of head, for k/v, -1 means equal to h\n"
                "if not equal to h, then this is GQA/MQA case")
        .insert("s",
                "3328",
                "seqlen_q. if group-mode, means the average value of seqlen_q\n"
                "total_seqlen_q = seqlen_q * batch, and seqlen_q per batch may vary\n"
                "also with \"-s=s0,s1,s2...\" comma-separated ints to set seqlen per batch "
                "(group mode)")
        .insert("s_k",
                "-1",
                "seqlen_k (including new key/value), -1 means equal to s\n"
                "also with \"-s_k=s0,s1,s2...\" comma-separated ints to set seqlen per batch "
                "(group mode)")
        .insert("s_knew",
                "0",
                "seqlen_k for new key/value, 0 means not to use this at all; "
                "-1 to choose s_knew in [1, s] randomly.")
        .insert("s_qpad",
                "-1",
                "seqlen_q stride between 2 batches (group-mode optional).\n"
                "Provide positive strides per-batch to simulate physical padding on Q.")
        .insert("s_kpad",
                "-1",
                "seqlen_k stride between 2 batches, currently used in group-mode only\n"
                "for kv-cache case, each batch [1,s,h,d]/[1,h,s,d] can have a stride\n"
                "along seqlen, instead of packed, same as xformer kv_padding,\n"
                "must be greater than or equal to s_k")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("scale_s", "0", "scale factor of S. 0 means equal to 1/sqrt(hdim)")
        .insert("qscale",
                "n",
                "n or 0, no scale\n"
                "pt or 1, per-tensor scale\n")
        .insert("logits_soft_cap", "0", "attention logits soft capping value.")
        .insert("iperm",
                "1",
                "permute input\n"
                "if true, will be b*h*s*d, else b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("bias",
                "n",
                "n or 0, no bias\n"
                "e(lementwise) or 1, elementwise bias with 1*1*s*s. e:1, 1*h*s*s. e:2, b*h*s*s\n"
                "a(libi) or 2, alibi with 1*h. a:1, b*h")
        .insert("prec", "fp16", "data type. fp32/fp16/bf16/fp8/bf8")
        .insert("mask",
                "0",
                "0: no mask, 1: top-left(same as 't'), 2:bottom-right(same as 'b')\n"
                "'t', top-left causal mask, 'b', bottom-r causal mask\n"
                "'t:l,r', top-left sliding window attn(swa) with FA style left right size\n"
                "'b:l,r', bottom-r sliding window attn(swa) with FA style left right size\n"
                "'xt:window_size', xformer style masking from top-left, window_size negative is "
                "causal, positive is swa\n"
                "'xb:window_size', xformer style masking from bottom-r, window_size negative is "
                "causal, positive is swa\n"
                "'g:y,x', generic attention mask coordinate with y/x size (only debug purpose for "
                "now)")
        .insert("vlayout", "r", "r for row-major(seqlen*hdim), c for col-major(hdim*seqlen)")
        .insert("lse", "0", "0 not store lse, 1 store lse")
        .insert("kname", "0", "if set to 1 will print kernel name")
        .insert("init",
                "uf",
                "init method:\n  ui or 0 - uniform random int\n  ni - normalized random int"
                "\n  uf or 1 - uniform random float\n  nf - normalized random float"
                "\n  tf or 2 - trig float"
                "\n  tf or 3 - uniform random float, min max is the max of the type\n")
        .insert("seed",
                "11939",
                "random seed used for initializing input tensors. 0 for "
                "non-deterministic seed")
        .insert("p_drop", "0", "0~1 probability of dropout")
        .insert("drop_seed", "1", "seed for dropout random number generator")
        .insert("drop_offset", "0", "offset for dropout random number generator")
        .insert(
            "drop_prefs",
            "0",
            "whether dropout seed and offset values are present on GPU; 0 - host, 1 - device/GPU")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert(
            "rotary_dim", "0", "RoPE rotary dimension. rotary_dim <= 0 means not apply RoPE at all")
        .insert("rotary_interleaved", "1", "whether to apply interleaved RoPE")
        .insert("num_splits",
                "1",
                "# of splits for key/value. 0 to determine actual number by heuristic")
        .insert("page_block_size", "0", "paged-kvcache block size. 0 means not use paged-kvcahe")
        .insert("cache_batch_idx", "0", "whether to use index map to the kvcache")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "20", "number of iterations to benchmark the kernel")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "fmha_fwd.json", "json file name to dump results")
        .insert("q_eff_lens",
                "",
                "Batch-mode only: per-batch effective seqlen for Q (exclude PAD).\n"
                "Comma-separated list of length 'b'. If empty, no override.")
        .insert("kv_eff_lens",
                "",
                "Batch-mode only: per-batch effective seqlen for KV (exclude PAD).\n"
                "Comma-separated list of length 'b'. If empty, no override.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataTypeConfig>
auto run(const ck_tile::ArgParser& arg_parser)
{
    int do_validation                = arg_parser.get_int("v");
    mode_enum mode                   = static_cast<mode_enum>(arg_parser.get_uint32("mode"));
    ck_tile::index_t batch           = arg_parser.get_int("b");
    ck_tile::index_t nhead           = arg_parser.get_int("h");
    ck_tile::index_t nhead_k         = arg_parser.get_int("h_k");
    auto seqlen_qs                   = arg_parser.get_int_vec("s");
    auto seqlen_ks                   = arg_parser.get_int_vec("s_k");
    ck_tile::index_t hdim_q          = arg_parser.get_int("d");
    ck_tile::index_t hdim_v          = arg_parser.get_int("d_v");
    ck_tile::index_t seqlen_knew     = arg_parser.get_int("s_knew");
    auto seqlen_kpads                = arg_parser.get_int_vec("s_kpad");
    auto seqlen_qpads                = arg_parser.get_int_vec("s_qpad");
    auto q_eff_lens_per_batch        = arg_parser.get_int_vec("q_eff_lens");
    auto kv_eff_lens_per_batch       = arg_parser.get_int_vec("kv_eff_lens");
    ck_tile::index_t rotary_dim      = arg_parser.get_int("rotary_dim");
    bool i_perm                      = arg_parser.get_bool("iperm");
    bool o_perm                      = arg_parser.get_bool("operm");
    float scale_s                    = arg_parser.get_float("scale_s");
    float logits_soft_cap            = arg_parser.get_float("logits_soft_cap");
    bool is_v_rowmajor               = arg_parser.get_str("vlayout") == "r";
    bool lse                         = arg_parser.get_bool("lse");
    ck_tile::index_t page_block_size = arg_parser.get_int("page_block_size");
    bool use_cache_batch_idx         = arg_parser.get_bool("cache_batch_idx");
    std::string bias_str             = arg_parser.get_str("bias");
    std::string qscale_str           = arg_parser.get_str("qscale");
    float p_drop                     = arg_parser.get_float("p_drop");
    uint64_t drop_seed               = arg_parser.get_uint64("drop_seed");
    uint64_t drop_offset             = arg_parser.get_uint64("drop_offset");
    bool drop_prefs                  = arg_parser.get_bool("drop_prefs");
    std::string mask_str             = arg_parser.get_str("mask");
    bool is_rotary_interleaved       = arg_parser.get_bool("rotary_interleaved");
    ck_tile::index_t num_splits      = arg_parser.get_int("num_splits");
    std::string init_method          = arg_parser.get_str("init");
    uint32_t seed                    = arg_parser.get_uint32("seed");

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /* log_level = */ (arg_parser.get_bool("kname") ? 1 : 0),
                                         arg_parser.get_int("warmup"),
                                         arg_parser.get_int("repeat"),
                                         arg_parser.get_str("timer") == std::string("gpu")};

    auto json = arg_parser.get_int("json") == 1
                    ? std::optional<std::string>{arg_parser.get_str("jsonfile")}
                    : std::nullopt;

    return fmha_fwd_run<DataTypeConfig>(mode,
                                        batch,
                                        nhead,
                                        nhead_k,
                                        seqlen_qs,
                                        seqlen_ks,
                                        hdim_q,
                                        hdim_v,
                                        seqlen_knew,
                                        seqlen_qpads,
                                        seqlen_kpads,
                                        q_eff_lens_per_batch,
                                        kv_eff_lens_per_batch,
                                        rotary_dim,
                                        i_perm,
                                        o_perm,
                                        scale_s,
                                        logits_soft_cap,
                                        is_v_rowmajor,
                                        lse,
                                        page_block_size,
                                        use_cache_batch_idx,
                                        bias_str,
                                        p_drop,
                                        drop_seed,
                                        drop_offset,
                                        drop_prefs,
                                        mask_str,
                                        qscale_str,
                                        is_rotary_interleaved,
                                        num_splits,
                                        init_method,
                                        seed,
                                        do_validation,
                                        stream_config,
                                        json);
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, arg_parser] = create_args(argc, argv);
        if(!result)
            return -1;

        const std::string data_type = arg_parser.get_str("prec");
        if(data_type == "fp32")
        {
            return run<FmhaFwdFp32>(arg_parser) == fwd_result::success ? 0 : -2;
        }
        else if(data_type == "fp16")
        {
            return run<FmhaFwdFp16>(arg_parser) == fwd_result::success ? 0 : -2;
        }
        else if(data_type == "bf16")
        {
            return run<FmhaFwdBf16>(arg_parser) == fwd_result::success ? 0 : -2;
        }
        else if(data_type == "fp8bf16")
        {
            return run<FmhaFwdFp8Bf16>(arg_parser) == fwd_result::success ? 0 : -2;
        }
        else if(data_type == "fp8fp32")
        {
            return run<FmhaFwdFp8Fp32>(arg_parser) == fwd_result::success ? 0 : -2;
        }
        std::cerr << "Unsupported precision: " << data_type << std::endl;
        return -1;
    }
    catch(const std::invalid_argument& e)
    {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        return -1;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -2;
    }
}
