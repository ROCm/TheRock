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

if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
  message(FATAL_ERROR
    "Cannot build 32-bit ROCm with MSVC. You must enable the Windows x64 "
    "development tools and rebuild.\n"
    "  Detected CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
endif()
