// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct ConvolutionBackwardDataSpecialization
{
    Default,
    Filter1x1Stride1Pad0,
};

inline std::string
getConvBackwardDataSpecializationString(const ConvolutionBackwardDataSpecialization& s)
{
    switch(s)
    {
    case ConvolutionBackwardDataSpecialization::Default: return "Default";
    case ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0: return "Filter1x1Stride1Pad0";
    default: return "Unrecognized specialization!";
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
