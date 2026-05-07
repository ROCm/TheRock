// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "test/smfmac_op/smfmac_op_util.hpp"

template <typename Src1Type,
          ck::index_t Src1VecSize,
          typename Src2Type,
          ck::index_t Src2VecSize,
          typename DstType,
          ck::index_t AccVecSize,
          typename GPUAccType,
          typename CPUAccType,
          ck::index_t M,
          ck::index_t N,
          ck::index_t K>
bool run_test()
{
    using Row         = ck::tensor_layout::gemm::RowMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    bool pass         = true;

    const auto matmul_default = ck::smfmac_op_util::matmul<Src1Type,
                                                           Src1VecSize,
                                                           Src2Type,
                                                           Src2VecSize,
                                                           GPUAccType,
                                                           AccVecSize,
                                                           DstType,
                                                           M,
                                                           N,
                                                           K>;

    const auto smfmac_kernel_container = std::make_tuple(matmul_default);

    ck::static_for<0, 1, 1>{}([&](auto i) {
        pass &=
            ck::smfmac_op_util::TestSmfmac<decltype(std::get<ck::Number<i>{}>(
                                               smfmac_kernel_container)),
                                           Src1Type,
                                           Src2Type,
                                           DstType,
                                           GPUAccType,
                                           CPUAccType,
                                           decltype(Row{}),
                                           decltype(Row{}),
                                           decltype(Row{}),
                                           PassThrough,
                                           PassThrough,
                                           PassThrough,
                                           AccVecSize,
                                           M,
                                           N,
                                           K>{}(std::get<ck::Number<i>{}>(smfmac_kernel_container));
    });

    return pass;
}
int main(int, char*[])
{
    bool pass = true;
    // clang-format off
    //              |   Src1Type| Src1VecSize|    Src2Type| Src2VecSize| DstType| DstVecSize|  GPUAccType| CPUAccType| M| N| K|
    pass &= run_test< ck::half_t,           4,  ck::half_t,           8,   float,          4,       float,      float,16,16,32>();
    pass &= run_test<ck::bhalf_t,           4, ck::bhalf_t,           8,   float,          4,       float,      float,16,16,32>();
    pass &= run_test< ck::half_t,           4,  ck::half_t,           8,   float,         16,       float,      float,32,32,16>();
    pass &= run_test<ck::bhalf_t,           4, ck::bhalf_t,           8,   float,         16,       float,      float,32,32,16>();
    // clang-format on

    std::cout << "TestGemm ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass;
}
