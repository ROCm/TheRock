// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#include <ostream>
#endif

namespace ck {

/// @brief Pipeline version enumeration for GEMM kernels
/// @details Defines different pipeline strategies for data movement and computation overlap
/// in GEMM kernels. This is a lightweight header containing only the enum definition,
/// extracted from gridwise_gemm_pipeline_selector.hpp to minimize dependencies.
enum struct PipelineVersion
{
    v1, ///< Version 1 pipeline
    v2, ///< Version 2 pipeline
    // v3 is only used in the Stream-K implementation.
    v4,          ///< Version 4 pipeline
    weight_only, ///< Weight-only specialized pipeline
};

} // namespace ck

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
inline std::ostream& operator<<(std::ostream& os, const ck::PipelineVersion& p)
{
    switch(p)
    {
    case ck::PipelineVersion::v1: os << "PipelineVersion::v1"; break;
    case ck::PipelineVersion::v2: os << "PipelineVersion::v2"; break;
    case ck::PipelineVersion::v4: os << "PipelineVersion::v4"; break;
    case ck::PipelineVersion::weight_only: os << "PipelineVersion::weight_only"; break;
    default: os << "";
    }
    return os;
}
#endif
