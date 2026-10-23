# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Lives in lib/rocprofiler-sdk
set_target_properties(rocprofiler-sdk-tool PROPERTIES THEROCK_INSTALL_RPATH_ORIGIN
  lib/rocprofiler-sdk
)
set_target_properties(rocprofiler-sdk-tool-kokkosp PROPERTIES THEROCK_INSTALL_RPATH_ORIGIN
  lib/rocprofiler-sdk
)
set_target_properties(rocprofv3-list-avail PROPERTIES THEROCK_INSTALL_RPATH_ORIGIN
  lib/rocprofiler-sdk
)

# rocprofiler-sdk component CI compiles its libraries with -Werror
# (ROCPROFILER_BUILD_DEVELOPER, which ROCPROFILER_BUILD_CI turns on by
# default). TheRock leaves that option off: it also makes clang-format,
# cmake-format, and black required at configure time, and it adds -Wextra,
# -Wshadow, -Wvla, and -Wdouble-promotion. The libraries already compile with
# -Wall and link this interface, so -Werror here is what fails the build on a
# diagnostic such as an unhandled enumerator. Tests and samples already get
# -Werror from ROCPROFILER_BUILD_CI.
target_compile_options(rocprofiler-sdk-build-flags INTERFACE -Werror)
