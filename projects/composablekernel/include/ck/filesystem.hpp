// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_CK_FILESYSTEM_HPP_
#define GUARD_CK_FILESYSTEM_HPP_

#include <string>
#include <string_view>

// clang-format off
#if defined(CPPCHECK)
  #define CK_HAS_FILESYSTEM 1
  #define CK_HAS_FILESYSTEM_TS 1
#elif defined(_WIN32)
  #if _MSC_VER >= 1920
    #define CK_HAS_FILESYSTEM 1
    #define CK_HAS_FILESYSTEM_TS 0
  #elif _MSC_VER >= 1900
    #define CK_HAS_FILESYSTEM 0
    #define CK_HAS_FILESYSTEM_TS 1
  #else
    #define CK_HAS_FILESYSTEM 0
    #define CK_HAS_FILESYSTEM_TS 0
  #endif
#elif defined(__has_include)
  #if __has_include(<filesystem>) && __cplusplus >= 201703L
    #define CK_HAS_FILESYSTEM 1
  #else
    #define CK_HAS_FILESYSTEM 0
  #endif
  #if __has_include(<experimental/filesystem>) && __cplusplus >= 201103L
    #define CK_HAS_FILESYSTEM_TS 1
  #else
    #define CK_HAS_FILESYSTEM_TS 0
  #endif
#else
  #define CK_HAS_FILESYSTEM 0
  #define CK_HAS_FILESYSTEM_TS 0
#endif
// clang-format on

#if CK_HAS_FILESYSTEM
#include <filesystem>
#elif CK_HAS_FILESYSTEM_TS
#include <experimental/filesystem>
#else
#error "No filesystem include available"
#endif

namespace CK {

#if CK_HAS_FILESYSTEM
namespace fs = ::std::filesystem;
#elif CK_HAS_FILESYSTEM_TS
namespace fs = ::std::experimental::filesystem;
#endif

} // namespace CK

inline std::string operator+(const std::string_view s, const CK::fs::path& path)
{
    return path.string().insert(0, s);
}

inline std::string operator+(const CK::fs::path& path, const std::string_view s)
{
    return path.string().append(s);
}

#define FS_ENUM_PERMS_ALL fs::perms::all

#if CK_HAS_FILESYSTEM_TS
#ifdef __linux__
#include <linux/limits.h>
namespace CK {
inline fs::path weakly_canonical(const fs::path& path)
{
    std::string result(PATH_MAX, '\0');
    std::string p{path.is_relative() ? (fs::current_path() / path).string() : path.string()};
    char* retval = realpath(p.c_str(), &result[0]);
    return (retval == nullptr) ? path : fs::path{result};
}
} // namespace CK
#else
#error "Not implmeneted!"
#endif
#else
namespace CK {
inline fs::path weakly_canonical(const fs::path& path) { return fs::weakly_canonical(path); }
} // namespace CK
#endif

namespace CK {

#ifdef _WIN32
constexpr std::string_view executable_postfix{".exe"};
constexpr std::string_view library_prefix{""};
constexpr std::string_view dynamic_library_postfix{".dll"};
constexpr std::string_view static_library_postfix{".lib"};
constexpr std::string_view object_file_postfix{".obj"};
#else
constexpr std::string_view executable_postfix{""};
constexpr std::string_view library_prefix{"lib"};
constexpr std::string_view dynamic_library_postfix{".so"};
constexpr std::string_view static_library_postfix{".a"};
constexpr std::string_view object_file_postfix{".o"};
#endif

inline fs::path make_executable_name(const fs::path& path)
{
    return path.parent_path() / (path.filename() + executable_postfix);
}

inline fs::path make_dynamic_library_name(const fs::path& path)
{
    return path.parent_path() / (library_prefix + path.filename() + dynamic_library_postfix);
}

inline fs::path make_object_file_name(const fs::path& path)
{
    return path.parent_path() / (path.filename() + object_file_postfix);
}

inline fs::path make_static_library_name(const fs::path& path)
{
    return path.parent_path() / (library_prefix + path.filename() + static_library_postfix);
}

struct FsPathHash
{
    std::size_t operator()(const fs::path& path) const { return fs::hash_value(path); }
};
} // namespace CK

#endif // GUARD_CK_FILESYSTEM_HPP_
