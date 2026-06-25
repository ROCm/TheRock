// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>

#include "profiler/profile_contraction_impl.hpp"
#include "profiler/profile_contraction_utils.hpp"
#include "profiler_operation_registry.hpp"

#define OP_NAME "contraction_bilinear"
#define OP_DESC "CONTRACTION+Bilinear"

static void print_helper_msg()
{
    std::cout << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
              << "arg2: data type (0: fp32; 1: f64; 2: f16; 3: bf16)\n"
              << "arg3: compute data type (0: fp32; 1: f64; 2: f16; 3: bf16)\n"
              << "arg4: Number of dimension for M, N and K (one for all)\n"
              << "arg5: matrix layout (0: A[m0, m1, k0, k1] * B[k0, k1, n0, n1] + "
                 "D[m0, m1, n0, n1] = E[m0, m1, n0, n1];\n"
              << "                     1: A[m0, m1, k0, k1] * B[n0, n1, k0, k1] + "
                 "D[m0, m1, n0, n1] = E[m0, m1, n0, n1];\n"
              << "                     2: A[k0, k1, m0, m1] * B[k0, k1, n0, n1] + "
                 "D[m0, m1, n0, n1] = E[m0, m1, n0, n1];\n"
              << "                     3: A[k0, k1, m0, m1] * B[n0, n1, k0, k1] + "
                 "D[m0, m1, n0, n1] = E[m0, m1, n0, n1])\n"
              << "arg6: verification (0: no; 1: yes)\n"
              << "arg7: initialization (0: no init; 1: integer value; 2: decimal " << "value)\n"
              << "arg8: print tensor value (0: no; 1: yes)\n"
              << "arg9: time kernel (0: no, 1: yes)\n"
              << "arg10: alpha\n"
              << "arg11: beta\n"
              << "arg12 to 17/29: M0, M1, N0, N1, K0, K1\n"
              << "arg18/30 to 33/77: Strides for A, B, D and E (skip for default)\n"
              << std::endl;
}

int profile_contraction_bilinear(int argc, char* argv[])
{
    const bool default_strides = argc == 18 || 30;

    if(argc != 34 && argc != 78 && !default_strides)
    {
        print_helper_msg();
        exit(1);
    }

    const auto data_type          = static_cast<ContractionDataType>(std::stoi(argv[2]));
    const auto compute_data_type  = static_cast<ContractionComputeDataType>(std::stoi(argv[3]));
    const ck::index_t NumDimMNK   = std::stoi(argv[4]);
    const auto layout             = static_cast<ContractionMatrixLayout>(std::stoi(argv[5]));
    const bool do_verification    = std::stoi(argv[6]);
    const ck::index_t init_method = std::stoi(argv[7]);
    const bool do_log             = std::stoi(argv[8]);
    const bool time_kernel        = std::stoi(argv[9]);
    const float alpha             = std::stof(argv[10]);
    const float beta              = std::stof(argv[11]);

    std::vector<ck::index_t> M;
    std::vector<ck::index_t> N;
    std::vector<ck::index_t> K;
    const ck::index_t dims_arg_num = 12;
    collect_index_params(argv, M, dims_arg_num, NumDimMNK);
    collect_index_params(argv, N, dims_arg_num + NumDimMNK, NumDimMNK);
    collect_index_params(argv, K, dims_arg_num + NumDimMNK * 2, NumDimMNK);

    std::vector<ck::index_t> StridesA(NumDimMNK * 2);
    std::vector<ck::index_t> StridesB(NumDimMNK * 2);
    std::vector<ck::index_t> StridesE(NumDimMNK * 2);
    std::vector<ck::index_t> StridesD(NumDimMNK * 2);
    if(!default_strides)
    {
        collect_index_params(argv, StridesA, dims_arg_num + NumDimMNK * 3, NumDimMNK * 2);
        collect_index_params(argv, StridesB, dims_arg_num + NumDimMNK * 5, NumDimMNK * 2);
        collect_index_params(argv, StridesE, dims_arg_num + NumDimMNK * 7, NumDimMNK * 2);
        collect_index_params(argv, StridesD, dims_arg_num + NumDimMNK * 9, NumDimMNK * 2);
    }

    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32  = float;
    using F64  = double;

    auto profile =
        [&](auto a_layout, auto b_layout, auto cde_layout, auto type, auto compute_type) {
            using ALayout   = decltype(a_layout);
            using BLayout   = decltype(b_layout);
            using CDELayout = decltype(cde_layout);

            using DataType        = decltype(type);
            using ComputeDataType = decltype(compute_type);

            if(default_strides)
            {
                auto merge_dims = [](const std::vector<ck::index_t>& dims01,
                                     const std::vector<ck::index_t>& dims23) {
                    std::vector<ck::index_t> dims_szt(dims01.begin(), dims01.end());
                    dims_szt.insert(dims_szt.end(), dims23.begin(), dims23.end());
                    return dims_szt;
                };

                assign_default_strides(a_layout, StridesA, merge_dims(M, K));
                assign_default_strides(b_layout, StridesB, merge_dims(N, K));
                assign_default_strides(cde_layout, StridesE, merge_dims(M, N));
                assign_default_strides(cde_layout, StridesD, merge_dims(M, N));
            }
            if(NumDimMNK == 2)
            {
                bool pass = ck::profiler::profile_contraction_impl<2,
                                                                   ALayout,
                                                                   BLayout,
                                                                   CDELayout,
                                                                   DataType,
                                                                   ComputeDataType,
                                                                   ck::Tuple<DataType>,
                                                                   Bilinear>(do_verification,
                                                                             init_method,
                                                                             do_log,
                                                                             time_kernel,
                                                                             Bilinear{alpha, beta},
                                                                             M,
                                                                             N,
                                                                             K,
                                                                             StridesA,
                                                                             StridesB,
                                                                             StridesE,
                                                                             StridesD);

                return pass;
            }
            else if(NumDimMNK == 6)
            {
                bool pass = ck::profiler::profile_contraction_impl<6,
                                                                   ALayout,
                                                                   BLayout,
                                                                   CDELayout,
                                                                   DataType,
                                                                   ComputeDataType,
                                                                   ck::Tuple<DataType>,
                                                                   Bilinear>(do_verification,
                                                                             init_method,
                                                                             do_log,
                                                                             time_kernel,
                                                                             Bilinear{alpha, beta},
                                                                             M,
                                                                             N,
                                                                             K,
                                                                             StridesA,
                                                                             StridesB,
                                                                             StridesE,
                                                                             StridesD);

                return pass;
            }
            else
            {
                throw std::runtime_error("Not supported NumDimMNK");
                return false;
            }
        };

    auto run_profile_for_datatype = [&](auto type, auto compute_type) {
        if(layout == ContractionMatrixLayout::MK_KN_MN_MN)
        {
            return profile(Row{}, Row{}, Row{}, type, compute_type);
        }
        else if(layout == ContractionMatrixLayout::MK_NK_MN_MN)
        {
            return profile(Row{}, Col{}, Row{}, type, compute_type);
        }
        else if(layout == ContractionMatrixLayout::KM_KN_MN_MN)
        {
            return profile(Col{}, Row{}, Row{}, type, compute_type);
        }
        else if(layout == ContractionMatrixLayout::KM_NK_MN_MN)
        {
            return profile(Col{}, Col{}, Row{}, type, compute_type);
        }
        return false;
    };

    if(data_type == ContractionDataType::F32_F32_F32_F32)
    {
        if(compute_data_type == ContractionComputeDataType::F32)
        {
            return run_profile_for_datatype(F32{}, F32{});
        }
        else if(compute_data_type == ContractionComputeDataType::F16)
        {
            return run_profile_for_datatype(F32{}, F16{});
        }
        else if(compute_data_type == ContractionComputeDataType::BF16)
        {
            return run_profile_for_datatype(F32{}, BF16{});
        }
        else
        {
            std::cout << "Incorrect combination of data type and compute data type." << std::endl;
            return 1;
        }
    }
    else if(data_type == ContractionDataType::F64_F64_F64_F64)
    {
        if(compute_data_type == ContractionComputeDataType::F64)
        {
            return run_profile_for_datatype(F64{}, F64{});
        }
        else if(compute_data_type == ContractionComputeDataType::F32)
        {
            return run_profile_for_datatype(F64{}, F32{});
        }
        else
        {
            std::cout << "Incorrect combination of data type and compute data type." << std::endl;
            return 1;
        }
    }
    else if(data_type == ContractionDataType::F16_F16_F16_F16)
    {
        if(compute_data_type == ContractionComputeDataType::F32)
        {
            return run_profile_for_datatype(F16{}, F32{});
        }
        else
        {
            std::cout << "Incorrect combination of data type and compute data type." << std::endl;
            return 1;
        }
    }
    else if(data_type == ContractionDataType::BF16_BF16_BF16_BF16)
    {
        if(compute_data_type == ContractionComputeDataType::F32)
        {
            return run_profile_for_datatype(BF16{}, F32{});
        }
        else
        {
            std::cout << "Incorrect combination of data type and compute data type." << std::endl;
            return 1;
        }
    }
    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_contraction_bilinear);
