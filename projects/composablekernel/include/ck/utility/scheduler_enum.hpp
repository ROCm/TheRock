// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#include <ostream>
#endif

namespace ck {

/// @brief Block GEMM pipeline version enumeration
/// @details Defines different block GEMM pipeline strategies.
/// This is a lightweight header containing only enum definitions,
/// extracted from blkgemmpipe_scheduler.hpp to minimize dependencies.
enum struct BlockGemmPipelineVersion
{
    // For GEMM
    v1, ///< Naive pipeline
    v2, ///< Memory-optimized pipeline
    v3, ///< Compute-optimized pipeline
    v4, ///< Compute-optimized with double LDS buffer
    v5, ///< Compute-optimized with double global prefetch register buffer

    // For GEMM with preshuffled weight
    // v1, single lds buffer
    // v2, double lds buffer
};

/// @brief Block GEMM pipeline scheduler enumeration
/// @details Defines scheduling strategies for block GEMM pipelines.
enum struct BlockGemmPipelineScheduler
{
    Intrawave, ///< Schedule within a single wavefront
    Interwave, ///< Schedule across multiple wavefronts
};

/// @brief Loop scheduler enumeration
/// @details Defines scheduling strategies for computational loops.
enum struct LoopScheduler
{
    Default,   ///< Default scheduling strategy
    Interwave, ///< Cross-wavefront scheduling
};

/// @brief Tail number enumeration for pipeline buffering
/// @details Defines the number of tail iterations in pipelined loops.
enum struct TailNumber
{
    // Single / Double buffer pipeline
    Odd,  ///< Odd number of iterations
    Even, ///< Even number of iterations

    // Long prefetch pipeline, up to 8
    One,   ///< One tail iteration
    Two,   ///< Two tail iterations
    Three, ///< Three tail iterations
    Four,  ///< Four tail iterations
    Five,  ///< Five tail iterations
    Six,   ///< Six tail iterations
    Seven, ///< Seven tail iterations

    // Unroll stages > Prefetch stages, number of loop is multiple of unroll stages
    Empty, ///< No tail iterations
           // Unroll stages <= Prefetch stages, number of loop is multiple of unroll stages add
           // prefetchstages
    Full,  ///< Full tail iterations
};

} // namespace ck

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
inline std::ostream& operator<<(std::ostream& os, const ck::LoopScheduler& s)
{
    switch(s)
    {
    case ck::LoopScheduler::Default: os << "Default"; break;
    case ck::LoopScheduler::Interwave: os << "Interwave"; break;
    default: os << "";
    }
    return os;
}
#endif
