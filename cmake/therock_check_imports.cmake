# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_check_imports.cmake
# Check that a Windows executable or DLL imports only the DLLs it is allowed to.
#
# Run as a script, typically as a ctest:
#
#   cmake -DFILE=<exe or dll>
#         -DREADOBJ=<path to llvm-readobj>
#         -DALLOW=<regular expression>
#         -P cmake/therock_check_imports.cmake
#
# Every DLL that FILE imports, delay-loaded ones included, must match ALLOW,
# which is matched against the whole name, lowercased, so write it in lowercase.
# The script fails naming every DLL that does not match.
#
# A DLL the build machine has but a user's machine may not, such as one from
# MSYS2's ucrt64/bin, loads fine where the binary was built and only fails to
# start elsewhere, so nothing else would notice it.  llvm-readobj comes with the
# compiler TheRock builds, so the check needs no MSYS2 and no Visual Studio.

foreach(_var FILE READOBJ ALLOW)
  if(NOT ${_var})
    message(FATAL_ERROR "FILE, READOBJ and ALLOW are all required")
  endif()
endforeach()
if(NOT EXISTS "${READOBJ}")
  message(FATAL_ERROR "llvm-readobj not found: ${READOBJ}")
endif()

execute_process(
  COMMAND "${READOBJ}" --coff-imports "${FILE}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _output
  ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "${READOBJ} --coff-imports ${FILE} failed: ${_error}")
endif()

# Each Import and DelayImport record has one "Name: <dll>" line.  The symbols
# under it are "Symbol: ..." lines, so they do not match.
string(REGEX MATCHALL "(^|\n)[ \t]*Name: [^\r\n]+" _lines "${_output}")
set(_bad)
set(_count 0)
foreach(_line IN LISTS _lines)
  string(REGEX REPLACE "^[\r\n \t]*Name: " "" _dll "${_line}")
  string(TOLOWER "${_dll}" _dll)
  math(EXPR _count "${_count} + 1")
  if(NOT _dll MATCHES "^(${ALLOW})$")
    list(APPEND _bad "${_dll}")
  endif()
endforeach()

# Every Windows binary imports something, so an empty list means the output
# was not what this script expects, not that the binary is clean.
if(_count EQUAL 0)
  message(FATAL_ERROR
    "Found no imports in ${FILE}, llvm-readobj's output being:\n${_output}")
endif()
if(_bad)
  list(JOIN _bad ", " _bad)
  message(FATAL_ERROR "${FILE} imports DLLs outside the allowed set: ${_bad}")
endif()
message(STATUS "${FILE}: all ${_count} imported DLLs allowed")
