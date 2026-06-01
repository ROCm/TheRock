// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_TEST_HOST_RTC_FILESYSTEM_HPP
#define GUARD_TEST_HOST_RTC_FILESYSTEM_HPP

#include <string>
#include <string_view>

// clang-format off
#if defined(CPPCHECK)
  #define RTC_HAS_FILESYSTEM 1
  #define RTC_HAS_FILESYSTEM_TS 1
#elif defined(_WIN32)
  #if _MSC_VER >= 1920
    #define RTC_HAS_FILESYSTEM 1
    #define RTC_HAS_FILESYSTEM_TS 0
  #elif _MSC_VER >= 1900
    #define RTC_HAS_FILESYSTEM 0
    #define RTC_HAS_FILESYSTEM_TS 1
  #else
    #define RTC_HAS_FILESYSTEM 0
    #define RTC_HAS_FILESYSTEM_TS 0
  #endif
#elif defined(__has_include)
  #if __has_include(<filesystem>) && __cplusplus >= 201703L
    #define RTC_HAS_FILESYSTEM 1
  #else
    #define RTC_HAS_FILESYSTEM 0
  #endif
  #if __has_include(<experimental/filesystem>) && __cplusplus >= 201103L
    #define RTC_HAS_FILESYSTEM_TS 1
  #else
    #define RTC_HAS_FILESYSTEM_TS 0
  #endif
#else
  #define RTC_HAS_FILESYSTEM 0
  #define RTC_HAS_FILESYSTEM_TS 0
#endif
// clang-format on

#if RTC_HAS_FILESYSTEM
#include <filesystem>
#elif RTC_HAS_FILESYSTEM_TS
#include <experimental/filesystem>
#else
#error "No filesystem include available"
#endif

namespace rtc {

#if RTC_HAS_FILESYSTEM
namespace fs = ::std::filesystem;
#elif RTC_HAS_FILESYSTEM_TS
namespace fs = ::std::experimental::filesystem;
#endif

} // namespace rtc

#endif // GUARD_RTC_FILESYSTEM_HPP_
