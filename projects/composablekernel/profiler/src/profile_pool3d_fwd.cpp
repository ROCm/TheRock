// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>
#include <unordered_map>

#include "profiler/data_type_enum.hpp"
#include "profiler/profile_pool3d_fwd_impl.hpp"
#include "profiler_operation_registry.hpp"

using ck::index_t;

struct poolFwdArgParser
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

void print_help_pool3d_fwd()
{
    std::cout << "arg1: data type (0: fp16; 1: fp32; 3: int8; 5: bf16; 7: fp8)\n"
              << "arg2: verification (0: no; 1: yes)\n"
              << "arg3: initialization (0: no init; 1: integer value; 2: decimal value)\n"
              << "arg4: print tensor value (0: no; 1: yes)\n"
              << "arg5: time kernel (0=no, 1=yes)\n"
              << "arg6: return index (0=no, 1=yes)\n"
              << "arg7: reduce op (0: max; 1: avg)\n"
              << "--length: input tensor length for NCDHW(e.g, --length 2 32 30 30 30) \n"
              << "--wsize: window size for ZYX (e.g, --wsize 2 2 2) \n"
              << "--wstride: window stride for DHW (e.g, --wstride 2 2 2) \n"
              << "--wdilation: window dilation for DHW (e.g, --wdilation 1 1 1) \n"
              << "--pad1: left side of padding in DHW (e.g, --pad1 1 1 1) \n"
              << "--pad2: right side of padding in DHW (e.g, --pad2 1 1 1) \n"
              << "eg: ckProfiler pool3d_fwd 0 1 2 0 1 0 --length 2 32 30 30 30 --wsize 2 2 2 "
                 "--wstride 2 2 2 --wdilation 1 1 1 --pad1 1 1 1 --pad2 1 1 1"
              << std::endl;
}

int profile_pool3d_fwd(int argc, char* argv[])
{
    ck::DataTypeEnum data_type = ck::DataTypeEnum::Half;
    ck::profiler::PoolFwdInputParams in_params{true, 0, false, true, false, 0};
    ck::profiler::PoolFwdKernelParams kernel_params{
        {2, 32, 30, 30, 30}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

    if(argc != 2 && argc != 35)
    {
        print_help_pool3d_fwd();
        return 0;
    }
    else if(argc == 35)
    {
        data_type                 = static_cast<ck::DataTypeEnum>(std::stoi(argv[2]));
        in_params.do_verification = std::stoi(argv[3]);
        in_params.init_method     = std::stoi(argv[4]);
        in_params.do_log          = std::stoi(argv[5]);
        in_params.time_kernel     = std::stoi(argv[6]);
        in_params.return_index    = std::stoi(argv[7]);
        in_params.reduce_op       = std::stoi(argv[8]);

        // parse the long options
        poolFwdArgParser arg_parser;
        arg_parser(argc, argv);
        kernel_params.in_length              = arg_parser.long_opts["length"];
        kernel_params.window_spatial_lengths = arg_parser.long_opts["wsize"];
        kernel_params.window_strides         = arg_parser.long_opts["wstride"];
        kernel_params.window_dilations       = arg_parser.long_opts["wdilation"];
        kernel_params.input_left_pads        = arg_parser.long_opts["pad1"];
        kernel_params.input_right_pads       = arg_parser.long_opts["pad2"];
    }

    using F16   = ck::half_t;
    using BF16  = ck::bhalf_t;
    using F32   = float;
    using I8    = int8_t;
    using I32   = int32_t;
    using F8    = ck::f8_t;
    using NDHWC = ck::tensor_layout::convolution::NDHWC;

    if(data_type == ck::DataTypeEnum::Half)
    {
        if(in_params.reduce_op == 1)
        {
            ck::profiler::profile_pool3d_fwd_impl<F16,
                                                  F16,
                                                  F32,
                                                  I32,
                                                  NDHWC,
                                                  NDHWC,
                                                  ck::ReduceTensorOp::AVG,
                                                  false,
                                                  false>(in_params, kernel_params);
        }
        else
        { // reduce_op == 0
            if(in_params.return_index)
            {
                ck::profiler::profile_pool3d_fwd_impl<F16,
                                                      F16,
                                                      F16,
                                                      I32,
                                                      NDHWC,
                                                      NDHWC,
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      true>(in_params, kernel_params);
            }
            else
            {
                ck::profiler::profile_pool3d_fwd_impl<F16,
                                                      F16,
                                                      F16,
                                                      I32,
                                                      NDHWC,
                                                      NDHWC,
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      false>(in_params, kernel_params);
            }
        }
    }
    else if(data_type == ck::DataTypeEnum::BFloat16)
    {
        if(in_params.reduce_op == 1)
        {
            ck::profiler::profile_pool3d_fwd_impl<BF16,
                                                  BF16,
                                                  F32,
                                                  I32,
                                                  NDHWC,
                                                  NDHWC,
                                                  ck::ReduceTensorOp::AVG,
                                                  false,
                                                  false>(in_params, kernel_params);
        }
        else
        { // reduce_op == 0
            if(in_params.return_index)
            {
                ck::profiler::profile_pool3d_fwd_impl<BF16,
                                                      BF16,
                                                      BF16,
                                                      I32,
                                                      NDHWC,
                                                      NDHWC,
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      true>(in_params, kernel_params);
            }
            else
            {
                ck::profiler::profile_pool3d_fwd_impl<BF16,
                                                      BF16,
                                                      BF16,
                                                      I32,
                                                      NDHWC,
                                                      NDHWC,
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      false>(in_params, kernel_params);
            }
        }
    }
    else if(data_type == ck::DataTypeEnum::Float)
    {
        if(in_params.reduce_op == 1)
        {
            ck::profiler::profile_pool3d_fwd_impl<F32,
                                                  F32,
                                                  F32,
                                                  I32,
                                                  NDHWC,
                                                  NDHWC,
                                                  ck::ReduceTensorOp::AVG,
                                                  false,
                                                  false>(in_params, kernel_params);
        }
        else
        { // reduce_op == 0
            if(in_params.return_index)
            {
                ck::profiler::profile_pool3d_fwd_impl<F32,
                                                      F32,
                                                      F32,
                                                      I32,
                                                      NDHWC,
                                                      NDHWC,
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      true>(in_params, kernel_params);
            }
            else
            {
                ck::profiler::profile_pool3d_fwd_impl<F32,
                                                      F32,
                                                      F32,
                                                      I32,
                                                      NDHWC,
                                                      NDHWC,
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      false>(in_params, kernel_params);
            }
        }
    }
    else if(data_type == ck::DataTypeEnum::Float8)
    {
        if(in_params.reduce_op == 1)
        {
            return ck::profiler::profile_pool3d_fwd_impl<F8,
                                                         F8,
                                                         F32,
                                                         I32,
                                                         NDHWC,
                                                         NDHWC,
                                                         ck::ReduceTensorOp::AVG,
                                                         false,
                                                         false>(in_params, kernel_params);
        }
        else
        { // reduce_op == 0
            if(in_params.return_index)
            {
                return ck::profiler::profile_pool3d_fwd_impl<F8,
                                                             F8,
                                                             F8,
                                                             I32,
                                                             NDHWC,
                                                             NDHWC,
                                                             ck::ReduceTensorOp::MAX,
                                                             false,
                                                             true>(in_params, kernel_params);
            }
            else
            {
                return ck::profiler::profile_pool3d_fwd_impl<F8,
                                                             F8,
                                                             F8,
                                                             I32,
                                                             NDHWC,
                                                             NDHWC,
                                                             ck::ReduceTensorOp::MAX,
                                                             false,
                                                             false>(in_params, kernel_params);
            }
        }
    }
    else if(data_type == ck::DataTypeEnum::Int8)
    {
        if(in_params.reduce_op == 1)
        {
            return ck::profiler::profile_pool3d_fwd_impl<I8,
                                                         I8,
                                                         I32,
                                                         I32,
                                                         NDHWC,
                                                         NDHWC,
                                                         ck::ReduceTensorOp::AVG,
                                                         false,
                                                         false>(in_params, kernel_params);
        }
        else
        { // reduce_op == 0
            if(in_params.return_index)
            {
                return ck::profiler::profile_pool3d_fwd_impl<I8,
                                                             I8,
                                                             I8,
                                                             I32,
                                                             NDHWC,
                                                             NDHWC,
                                                             ck::ReduceTensorOp::MAX,
                                                             false,
                                                             true>(in_params, kernel_params);
            }
            else
            {
                return ck::profiler::profile_pool3d_fwd_impl<I8,
                                                             I8,
                                                             I8,
                                                             I32,
                                                             NDHWC,
                                                             NDHWC,
                                                             ck::ReduceTensorOp::MAX,
                                                             false,
                                                             false>(in_params, kernel_params);
            }
        }
    }
    else
    {
        throw std::runtime_error("not implemented yet");
    }

    return 0;
}

REGISTER_PROFILER_OPERATION("pool3d_fwd", "pool3d fwd", profile_pool3d_fwd);
