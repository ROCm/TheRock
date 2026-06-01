// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_gemm_multiple_d/problem.hpp"
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/utils.hpp"
#include <algorithm>

namespace ck {
namespace host {
namespace device_gemm_multiple_d {

// return the relevant device op file based on the operation
std::string Problem::GetIncludeHeader() const
{
    return "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp";
}

// returns templated instances when provided with a problem specification
std::vector<Solution> Problem::GetSolutions(const std::string& arch,
                                            const std::string& prologue,
                                            const std::string& epilogue) const
{
    if(get_xdlop_archs().count(arch) == 0)
        return {};
    auto ops = ck::host::device_gemm_multiple_d::Operation_Xdl_CShuffle::CreateOperations(
        *this, prologue, epilogue); // obtains vector of instances
    std::vector<Solution> result;
    std::transform(ops.begin(), ops.end(), std::back_inserter(result), [&](const auto& op) {
        return op.ToSolution(); // template instance with correct values
    });
    return result;
}

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
