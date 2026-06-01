// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct TensorSpecialization
{
    Default,
    Packed
};

inline std::string getTensorSpecializationString(const TensorSpecialization& s)
{
    switch(s)
    {
    case TensorSpecialization::Default: return "Default";
    case TensorSpecialization::Packed: return "Packed";
    default: return "Unrecognized specialization!";
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
