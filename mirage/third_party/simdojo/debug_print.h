// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#ifndef SIMDOJO_DEBUG_PRINT_H_
#define SIMDOJO_DEBUG_PRINT_H_

#include <iostream>
#include <ostream>

namespace simdojo {
namespace debug {

#ifndef NDEBUG
inline constexpr bool DEBUG_ENABLE = true;
#else
inline constexpr bool DEBUG_ENABLE = false;
#endif

template <typename... Args> static void print(Args &&...args) {
  if constexpr (DEBUG_ENABLE) {
    std::ostream &trace_stream(std::cerr);
    (trace_stream << ... << args);
    trace_stream << std::endl;
    trace_stream.flush();
  }
}

} // namespace debug
} // namespace simdojo

#endif // SIMDOJO_DEBUG_PRINT_H_
