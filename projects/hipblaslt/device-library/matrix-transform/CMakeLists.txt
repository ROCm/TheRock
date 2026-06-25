# ########################################################################
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Need to support Release/Debug

string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${GPU_TARGETS}")
list(REMOVE_DUPLICATES archs)

# Create separate --offload-arch flags for each architecture
set(arch_flags "")
foreach(arch IN LISTS archs)
    list(APPEND arch_flags "--offload-arch=${arch}")
endforeach()

set(output_library "${CMAKE_CURRENT_BINARY_DIR}/hipblasltTransform.hsaco")
set(matrix_transform_cpp "${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform.cpp")

add_custom_command(
    COMMAND ${CMAKE_CXX_COMPILER} -x hip ${matrix_transform_cpp} ${arch_flags} -c --offload-device-only -Xoffload-linker --build-id=sha1 -O3 -o ${output_library}
    OUTPUT ${output_library}
    COMMENT "Building hipblasltTransform.hsaco"
)

add_custom_target(matrix-transform-cp ALL
    DEPENDS ${output_library}
    COMMAND ${CMAKE_COMMAND} -E copy "${output_library}" "${HIPBLASLT_TENSILE_LIBPATH}/library"
)
