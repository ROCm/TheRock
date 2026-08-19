# Porting AMD TheRock to GCC 15, Ubuntu 26.04 LTS, and AMD Strix Halo (gfx1151)

## Overview

This document summarizes the modifications made to [ROCm/TheRock](https://github.com/ROCm/TheRock) to enable complete compatibility with **Ubuntu 26.04 LTS (Resolute Raccoon)**, **GCC 15.2.0**, **CMake 4.x**, and the **AMD Strix Halo APU (`gfx1151` / Radeon 8060S / 8050S)**.

### Verified Reference Hardware Platform

| Specification Item | Value |
| :--- | :--- |
| **System / Model** | **GMKtec NucBox EVO-X2** (SKU: EVO-X2-001) |
| **APU / Processor** | **AMD Ryzen™ AI MAX+ 395** (16 Cores, 32 Threads, Strix Halo) |
| **Integrated GPU** | **AMD Radeon™ 8060S Graphics** (40 CUs / 2560 SPs, RDNA 3.5, ISA: `gfx1151`) |
| **System Memory** | **128 GB LPDDR5X** Unified High-Speed Memory |
| **Operating System** | **Ubuntu 26.04 LTS (Resolute Raccoon)** |
| **Kernel Version** | `Linux 7.0.0-29-generic` (x86_64) |
| **Host Toolchain** | GCC 15.2.0 (`gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0`) / G++ 15.2.0 |
| **Build Tools** | CMake 4.2.3, Ninja 1.12.1, Python 3.14 |

---

## 1. Summary of Porting Modifications

### 1.1 CMake 4.x Compatibility in Superproject Dependency Provider
* **File**: `cmake/therock_primlibs_benchmark_deps.cmake`
* **Problem**: In CMake 4.x, `CMAKE_PROJECT_TOP_LEVEL_INCLUDES` is evaluated at the start of `project()` before language initialization (`enable_language`) is complete. Calling `find_package(amd_smi REQUIRED CONFIG)` immediately caused CMake to evaluate `add_library(amd_smi SHARED IMPORTED)` without active dynamic linking platform support, triggering the error:
  `ADD_LIBRARY called with SHARED option but the target platform does not support dynamic linking.`
* **Solution**: Deferred the `find_package` evaluation using CMake's native `cmake_language(DEFER CALL ...)` mechanism so it executes after platform and language initialization.

```diff
--- a/cmake/therock_primlibs_benchmark_deps.cmake
+++ b/cmake/therock_primlibs_benchmark_deps.cmake
@@ -14,3 +14,3 @@
 if(NOT TARGET amd_smi)
-  find_package(amd_smi REQUIRED CONFIG)
+  cmake_language(DEFER CALL find_package amd_smi REQUIRED CONFIG)
 endif()
```

---

### 1.2 GCC 15 / libstdc++ Standard Header Conformance (C++20 `<ciso646>` Removal)
* **File**: `rocm-libraries/projects/miopen/src/include/miopen/serializable.hpp`
* **Problem**: ISO C++ deprecated and removed `<ciso646>` in C++20 in favor of `<version>`. In GCC 15 / libstdc++, including `<ciso646>` emits a `#warning`. Because MIOpen compiles with `-Werror`, this triggered a compilation failure (`-Werror,-W#warnings`).
* **Solution**: Replaced the unconditional `#include <ciso646>` with standard feature testing via `#if __has_include(<version>)`.

```diff
--- a/projects/miopen/src/include/miopen/serializable.hpp
+++ b/projects/miopen/src/include/miopen/serializable.hpp
@@ -29,3 +29,7 @@
- #include <ciso646>
+ #if __has_include(<version>)
+ #include <version>
+ #else
+ #include <ciso646>
+ #endif
```

---

### 1.3 Hermetic `sqlite3` Include Directory Propagation in `rocprofiler-sdk`
* **File**: `rocm-systems/projects/rocprofiler-sdk/source/lib/output/CMakeLists.txt`
* **Problem**: When using TheRock's bundled `therock-sqlite3` sysdep, the include directory containing `sqlite3.h` was not exported to `rocprofiler-sdk-output-library` and downstream targets (such as Python rocpd bindings), causing `'sqlite3.h' file not found`.
* **Solution**: Linked `rocprofiler-sdk::rocprofiler-sdk-sqlite3` as `PUBLIC` in `rocprofiler-sdk-output-library` to transitively propagate `INTERFACE_INCLUDE_DIRECTORIES` to all consumers.

```diff
--- a/projects/rocprofiler-sdk/source/lib/output/CMakeLists.txt
+++ b/projects/rocprofiler-sdk/source/lib/output/CMakeLists.txt
@@ -66,2 +66,3 @@ target_link_libraries(
            rocprofiler-sdk::rocprofiler-sdk-aqlprofile
+           rocprofiler-sdk::rocprofiler-sdk-sqlite3
     PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
```

---

### 1.4 Linux Sysdeps Symbol Versioning & Meson Linking for GCC 15 / Modern `ld`
* **Files**:
  - `third-party/sysdeps/linux/amd-mesa/CMakeLists.txt` & `patch_source.sh`
  - `third-party/sysdeps/linux/libdrm/CMakeLists.txt` & `patch_source.sh`
  - `third-party/sysdeps/linux/libpciaccess/CMakeLists.txt`
* **Problem**: In GCC 15 and modern binutils `ld`, injecting `-Wl,--version-script=...` through global `LDFLAGS` environment variables caused duplicate symbol version node definitions and build failures in Meson subprojects.
* **Solution**: Removed `--version-script` from the external `LDFLAGS` environment and added it cleanly using Meson's native `add_project_link_arguments('-Wl,--version-script=...', language : ['c', 'cpp'])` within each package's `patch_source.sh`.

---

### 1.5 Host Warning Suppression for Stricter GCC 15 Diagnostics
* **File**: `CMakeLists.txt` (Root)
* **Problem**: GCC 15 enforces much stricter static diagnostics (such as `-Wtemplate-body`, `-Wmaybe-uninitialized`, `-Wdangling-reference`, `-Wstringop-overread`) by default.
* **Solution**: Configured top-level compiler warning suppressions to prevent new warnings from turning into build errors during host sysdep compilation.

```cmake
add_compile_options(
  -Wno-error=maybe-uninitialized
  -Wno-error=template-body
  -Wno-error=dangling-reference
  -Wno-error=stringop-overread
  -Wno-maybe-uninitialized
)
```

---

## 2. Backward Compatibility Assessment

All changes made are **fully backward compatible** with GCC versions below 15 (e.g., GCC 11, 12, 13, 14) and older Ubuntu releases (e.g., 22.04 LTS, 24.04 LTS):

1. **`cmake_language(DEFER ...)`**: Supported since CMake 3.19 (TheRock requires CMake 3.25+).
2. **`#if __has_include(<version>)`**: Standard C++ feature test macro supported across all modern C++14/C++17/C++20 compilers.
3. **`PUBLIC` CMake target linking**: Standard CMake functionality across all supported versions.
4. **Meson `add_project_link_arguments`**: Officially supported Meson syntax across all versions.
5. **`-Wno-error=...` flags**: GCC safely ignores unknown `-Wno-...` options on older compiler versions without error.
