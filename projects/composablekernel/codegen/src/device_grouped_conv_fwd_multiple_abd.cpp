// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_problem.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_op.hpp"
#include "ck/host/utils.hpp"
#include <algorithm>
#include <iostream>

namespace ck {
namespace host {
namespace conv {

// return the relevant device op file based on the operation
// NOTE: this is a modified version of the original CK file that calls the kernel from a device
// function and makes the Argument class accessible on the device
std::string Problem_Conv_Fwd::GetIncludeHeader() const
{
    return "ck/tensor_operation/gpu/device/impl/"
           "codegen_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp";
}

// return vector of forward convolution instances when provided with a problem instance
std::vector<Solution> Problem_Conv_Fwd::GetSolutions(const std::string& arch,
                                                     const std::string& prologue,
                                                     const std::string& epilogue) const
{
    if(get_xdlop_archs().count(arch) == 0)
        return {};
    auto ops = ck::host::conv::Operation_Conv_Fwd_Xdl_Cshuffle::CreateOperations(
        *this, prologue, epilogue);
    std::vector<Solution> result;
    std::transform(ops.begin(), ops.end(), std::back_inserter(result), [&](const auto& op) {
        return op.ToSolution();
    });
    return result;
}

} // namespace conv
} // namespace host
} // namespace ck
