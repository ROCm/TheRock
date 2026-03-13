# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-hook for rocFFT to handle build-time tools with sanitizers.
#
# rocFFT uses rocfft_aot_helper during build to generate kernel cache databases.
# When building with ASAN, this tool needs the ASan runtime preloaded via LD_PRELOAD
# to avoid the "ASan runtime does not come first in initial library list" error.
#
# CMAKE_CROSSCOMPILING_EMULATOR is the standard CMake mechanism for prefixing
# executable invocations, which rocFFT's custom commands will respect.

if(THEROCK_SANITIZER_LAUNCHER)
  message(STATUS "rocFFT: Configuring sanitizer launcher for build-time tools")
  # CMAKE_CROSSCOMPILING_EMULATOR must be a single string with semicolon-separated list items
  # THEROCK_SANITIZER_LAUNCHER is already a list, so we can pass it directly
  list(APPEND _cmake_args "-DCMAKE_CROSSCOMPILING_EMULATOR=${THEROCK_SANITIZER_LAUNCHER}")
endif()
