# Project wide compiler configuration.

# On win32 only support embedded debug databases project wide.
# This improves compatibility with ccache and sccache:
# https://github.com/ccache/ccache/issues/1040
if(WIN32 AND NOT DEFINED CMAKE_MSVC_DEBUG_INFORMATION_FORMAT)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif()

if(WIN32 AND NOT MSVC AND NOT THEROCK_DISABLE_MSVC_CHECK)
  message(FATAL_ERROR
      "Bootstrap compiler was expected to be MSVC (cl.exe). Instead found:\n"
      "  Detected CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}\n"
      "  Detected CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}\n"
      "To use MSVC, try one of these:\n"
      "  * Use one of the MSVC command prompts like `x64 Native Tools Command Prompt for VS 2022`\n"
      "  * In your existing terminal, run `vcvars64.bat` from e.g. `C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build`\n"
      "  * Set `-DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe`\n"
      "Set THEROCK_DISABLE_MSVC_CHECK to bypass this check.")
endif()

if(MSVC AND MSVC_VERSION LESS_EQUAL 1941 AND NOT THEROCK_DISABLE_MSVC_VERSION_CHECK)
  # See https://github.com/ROCm/TheRock/issues/711. We can probably remove this
  # check after adding openmp to LLVM_ENABLE_RUNTIMES in pre_hook_amd-llvm.cmake.
  #
  # For the compiler version mapping, see
  # https://learn.microsoft.com/en-us/cpp/overview/compiler-versions
  message(FATAL_ERROR
      "Detected MSVC_VERSION is too old and may encounter OpenMP linker errors.\n"
      "  Detected MSVC_VERSION: ${MSVC_VERSION}\n"
      "  Detected MSVC_TOOLSET_VERSION: ${MSVC_TOOLSET_VERSION}\n"
      "  Detected CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}\n"
      "We recommend at least Visual Studio [Build Tools] version 17.12.*, with:\n"
      "  * MSVC_VERSION 1942\n"
      "  * CMAKE_CXX_COMPILER_VERSION version 19.42.*\n"
      "Set THEROCK_DISABLE_MSVC_VERSION_CHECK to bypass this check.")
endif()

if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
  message(FATAL_ERROR
    "Cannot build 32-bit ROCm with MSVC. You must enable the Windows x64 "
    "development tools and rebuild.\n"
    "  Detected CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
endif()


if(UNIX AND NOT APPLE AND NOT THEROCK_DISABLE_GNU_CHECK)
  # Linux-specific check.
  if(THEROCK_ENABLE_ROCPROFSYS AND (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_C_COMPILER_ID STREQUAL "GNU"))
   # Currently, the Dyninst dependency in rocprofiler-systems requires GNU compilers (gcc/g++).
    message(FATAL_ERROR
        "Currently, rocprofiler-systems requires GNU compilers (gcc/g++).\n"
        "  Detected CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}\n"
        "  Detected CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}\n"
        "  Detected CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}\n"
        "  Detected CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}\n"
        "To use GNU compilers, set:\n"
        "  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++\n"
        "To disable rocprofiler-systems, set:\n"
        "  -DTHEROCK_ENABLE_ROCPROFSYS=OFF\n"
        "Set THEROCK_DISABLE_GNU_CHECK to bypass this check.")
  endif()
endif()
