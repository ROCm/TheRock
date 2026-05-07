// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct GemmSpecialization
{
    // Gemm
    Default,
    MPadding,
    NPadding,
    KPadding,
    MNPadding,
    MKPadding,
    NKPadding,
    MNKPadding,
    // Gemm + Gemm
    OPadding,
    MOPadding,
    NOPadding,
    KOPadding,
    MNOPadding,
    MKOPadding,
    NKOPadding,
    MNKOPadding,
};
#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
inline std::string getGemmSpecializationString(const GemmSpecialization& s)
{
    switch(s)
    {
    case GemmSpecialization::Default: return "Default";
    case GemmSpecialization::MPadding: return "MPadding";
    case GemmSpecialization::NPadding: return "NPadding";
    case GemmSpecialization::KPadding: return "KPadding";
    case GemmSpecialization::MNPadding: return "MNPadding";
    case GemmSpecialization::MKPadding: return "MKPadding";
    case GemmSpecialization::NKPadding: return "NKPadding";
    case GemmSpecialization::MNKPadding: return "MNKPadding";
    case GemmSpecialization::OPadding: return "OPadding";
    case GemmSpecialization::MOPadding: return "MOPadding";
    case GemmSpecialization::NOPadding: return "NOPadding";
    case GemmSpecialization::KOPadding: return "KOPadding";
    case GemmSpecialization::MNOPadding: return "MNOPadding";
    case GemmSpecialization::MKOPadding: return "MKOPadding";
    case GemmSpecialization::NKOPadding: return "NKOPadding";
    case GemmSpecialization::MNKOPadding: return "MNKOPadding";
    default: return "Unrecognized specialization!";
    }
}
#endif

} // namespace device
} // namespace tensor_operation
} // namespace ck
