// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>
#include <unordered_map>

#include "profiler/data_type_enum.hpp"
#include "profiler/profile_pool2d_fwd_impl.hpp"
#include "profiler_operation_registry.hpp"

using ck::index_t;

struct maxPoolFwdArgParser
{
    std::unordered_map<std::string, std::vector<int>> long_opts = {{"length", {}},
                                                                   {"wsize", {}},
                                                                   {"wstride", {}},
                                                                   {"wdilation", {}},
                                                                   {"pad1", {}},
                                                                   {"pad2", {}}};

    bool parse_opt(int argc, char* argv[], const std::string& key, int i)
    {
        if(std::string("--") + key == argv[i])
        {
            int pos = i;
            while(++i < argc && argv[i][0] != '-') {}
            int end = i;
            for(int j = pos + 1; j < end; j++)
            {
                long_opts[key].push_back(std::stoi(argv[j]));
            }
            return true;
        }
        return false;
    }

    void operator()(int argc, char* argv[])
    {
        for(auto& kv : long_opts)
        {
            for(int i = 1; i < argc; i++)
            {
                if(parse_opt(argc, argv, kv.first, i))
                    break;
            }
        }
    }
};

enum struct PoolDataType
{
    F32 = 0,
    BF16,
    F16,
    INT8,
    F8,
};

void print_help_max_pool2d_fwd()
{
    std::cout << "arg1: data type (0: fp16; 1: fp32; 2: bf16; 3: int8; 4: fp8)\n"
              << "arg2: verification (0: no; 1: yes)\n"
              << "arg3: initialization (0: no init; 1: integer value; 2: decimal value)\n"
              << "arg4: print tensor value (0: no; 1: yes)\n"
              << "arg5: time kernel (0=no, 1=yes)\n"
              << "arg6: return index (0=no, 1=yes)\n"
              << "--length: input tensor length for NCHW(e.g, --length 2 32 30 30) \n"
              << "--wsize: window size for YX (e.g, --wsize 2 2) \n"
              << "--wstride: window stride for HW (e.g, --wstride 2 2) \n"
              << "--wdilation: window dilation for HW (e.g, --wdilation 1 1) \n"
              << "--pad1: left side of padding in HW (e.g, --pad1 1 1) \n"
              << "--pad2: right side of padding in HW (e.g, --pad2 1 1) \n"
              << "eg: ckProfiler max_pool2d_fwd 0 1 2 0 1 0 --length 2 32 30 30 --wsize 2 2"
                 "--wstride 2 2 --wdilation 1 1 --pad1 1 1 --pad2 1 1"
              << std::endl;
}

int profile_max_pool2d_fwd(int argc, char* argv[])
{
    PoolDataType data_type = PoolDataType::F32;
    bool do_verification   = true;
    int init_method        = 0;
    bool do_log            = false;
    bool time_kernel       = true;
    bool return_index      = false;

    std::vector<index_t> in_length = {2, 32, 30, 30};
    std::vector<index_t> wsize     = {2, 2};
    std::vector<index_t> wstride   = {2, 2};
    std::vector<index_t> wdilation = {1, 1};
    std::vector<index_t> pad1      = {1, 1};
    std::vector<index_t> pad2      = {1, 1};

    if(argc != 2 && argc != 28)
    {
        print_help_max_pool2d_fwd();
        return 0;
    }
    else if(argc == 28)
    {
        data_type       = static_cast<PoolDataType>(std::stoi(argv[2]));
        do_verification = std::stoi(argv[3]);
        init_method     = std::stoi(argv[4]);
        do_log          = std::stoi(argv[5]);
        time_kernel     = std::stoi(argv[6]);
        return_index    = std::stoi(argv[7]);

        // parse the long options
        maxPoolFwdArgParser arg_parser;
        arg_parser(argc, argv);
        in_length = arg_parser.long_opts["length"];
        wsize     = arg_parser.long_opts["wsize"];
        wstride   = arg_parser.long_opts["wstride"];
        wdilation = arg_parser.long_opts["wdilation"];
        pad1      = arg_parser.long_opts["pad1"];
        pad2      = arg_parser.long_opts["pad2"];
    }

    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32  = float;
    using I32  = int32_t;
    using F8   = ck::f8_t;
    using I8   = int8_t;
    using NHWC = ck::tensor_layout::convolution::NHWC;

    constexpr auto ReduceOpId = ck::ReduceTensorOp::MAX;

    if(data_type == PoolDataType::F16)
    {
        if(return_index)
        {
            ck::profiler::
                profile_pool2d_fwd_impl<F16, F16, F16, I32, NHWC, NHWC, ReduceOpId, false, true>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    in_length,
                    wsize,
                    wstride,
                    wdilation,
                    pad1,
                    pad2);
        }
        else
        {
            ck::profiler::
                profile_pool2d_fwd_impl<F16, F16, F16, I32, NHWC, NHWC, ReduceOpId, false, false>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    in_length,
                    wsize,
                    wstride,
                    wdilation,
                    pad1,
                    pad2);
        }
    }
    else if(data_type == PoolDataType::BF16)
    {
        if(return_index)
        {
            ck::profiler::
                profile_pool2d_fwd_impl<BF16, BF16, BF16, I32, NHWC, NHWC, ReduceOpId, false, true>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    in_length,
                    wsize,
                    wstride,
                    wdilation,
                    pad1,
                    pad2);
        }
        else
        {
            ck::profiler::profile_pool2d_fwd_impl<BF16,
                                                  BF16,
                                                  BF16,
                                                  I32,
                                                  NHWC,
                                                  NHWC,
                                                  ReduceOpId,
                                                  false,
                                                  false>(do_verification,
                                                         init_method,
                                                         do_log,
                                                         time_kernel,
                                                         in_length,
                                                         wsize,
                                                         wstride,
                                                         wdilation,
                                                         pad1,
                                                         pad2);
        }
    }
    else if(data_type == PoolDataType::F32)
    {
        if(return_index)
        {
            ck::profiler::
                profile_pool2d_fwd_impl<F32, F32, F32, I32, NHWC, NHWC, ReduceOpId, false, true>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    in_length,
                    wsize,
                    wstride,
                    wdilation,
                    pad1,
                    pad2);
        }
        else
        {
            ck::profiler::
                profile_pool2d_fwd_impl<F32, F32, F32, I32, NHWC, NHWC, ReduceOpId, false, false>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    in_length,
                    wsize,
                    wstride,
                    wdilation,
                    pad1,
                    pad2);
        }
    }
    else if(data_type == PoolDataType::INT8)
    {
        if(return_index)
        {
            ck::profiler::
                profile_pool2d_fwd_impl<I8, I8, F32, I32, NHWC, NHWC, ReduceOpId, false, true>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    in_length,
                    wsize,
                    wstride,
                    wdilation,
                    pad1,
                    pad2);
        }
        else
        {
            ck::profiler::
                profile_pool2d_fwd_impl<I8, I8, F32, I32, NHWC, NHWC, ReduceOpId, false, false>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    in_length,
                    wsize,
                    wstride,
                    wdilation,
                    pad1,
                    pad2);
        }
    }
    else if(data_type == PoolDataType::F8)
    {
        if(return_index)
        {
            ck::profiler::
                profile_pool2d_fwd_impl<F8, F8, F32, I32, NHWC, NHWC, ReduceOpId, false, true>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    in_length,
                    wsize,
                    wstride,
                    wdilation,
                    pad1,
                    pad2);
        }
        else
        {
            ck::profiler::
                profile_pool2d_fwd_impl<F8, F8, F32, I32, NHWC, NHWC, ReduceOpId, false, false>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    in_length,
                    wsize,
                    wstride,
                    wdilation,
                    pad1,
                    pad2);
        }
    }
    else
    {
        throw std::runtime_error("not implemented yet");
    }

    return 0;
}

REGISTER_PROFILER_OPERATION("max_pool2d_fwd", "max_pool2d fwd", profile_max_pool2d_fwd);
