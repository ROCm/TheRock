// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_fastgelu.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "profile_grouped_gemm_impl.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
bool profile_grouped_gemm_fastgelu_impl(int do_verification,
                                        int init_method,
                                        bool do_log,
                                        bool time_kernel,
                                        const std::vector<int>& Ms,
                                        const std::vector<int>& Ns,
                                        const std::vector<int>& Ks,
                                        const std::vector<int>& StrideAs,
                                        const std::vector<int>& StrideBs,
                                        const std::vector<int>& StrideCs)
{
    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::FastGelu;

    return profile_grouped_gemm_impl<ADataType,
                                     BDataType,
                                     CDataType,
                                     AccDataType,
                                     ALayout,
                                     BLayout,
                                     CLayout,
                                     AElementOp,
                                     BElementOp,
                                     CElementOp>(do_verification,
                                                 init_method,
                                                 do_log,
                                                 time_kernel,
                                                 Ms,
                                                 Ns,
                                                 Ks,
                                                 StrideAs,
                                                 StrideBs,
                                                 StrideCs,
                                                 {1});
}

} // namespace profiler
} // namespace ck
