// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "profiler/profile_gemm_reduce_impl.hpp"
static ck::index_t instance_index = -1;
int main(int argc, char** argv)
{
    if(argc == 1) {}
    else if(argc == 2)
    {
        instance_index = atoi(argv[1]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1: instance_index(-1 means all)" << std::endl;
    }

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    int M = 512;
    int N = 256;
    int K = 128;

    bool pass = true;

    pass = pass &&
           ck::profiler::
               profile_gemm_reduce_impl<ck::half_t, ck::half_t, ck::half_t, float, Row, Row, Row>(
                   true, 1, false, false, M, N, K, K, N, N, instance_index);

    pass = pass &&
           ck::profiler::
               profile_gemm_reduce_impl<ck::half_t, ck::half_t, ck::half_t, float, Row, Col, Row>(
                   true, 1, false, false, M, N, K, K, K, N, instance_index);

    pass = pass &&
           ck::profiler::
               profile_gemm_reduce_impl<ck::half_t, ck::half_t, ck::half_t, float, Col, Row, Row>(
                   true, 1, false, false, M, N, K, M, N, N, instance_index);

    pass = pass &&
           ck::profiler::
               profile_gemm_reduce_impl<ck::half_t, ck::half_t, ck::half_t, float, Col, Col, Row>(
                   true, 1, false, false, M, N, K, M, K, N, instance_index);

    if(pass)
    {
        std::cout << "test GEMM+Reduce fp16: Pass" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "test GEMM+Reduce fp16: Fail" << std::endl;
        return -1;
    }
}
