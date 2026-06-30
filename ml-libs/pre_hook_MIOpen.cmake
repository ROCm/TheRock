# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# TODO: hipblas does not appear to be setting its include directory when used
# in an isolated directory via find_package (this has probably been masked when
# in a directory with everything else). So we just include it here.
# Fix upstream and remove this.
# Use SYSTEM so these third-party headers are treated as system includes and do
# not trigger warnings from MIOpen's own -Wall/-Wextra/-Werror flags.
include_directories(SYSTEM "${THEROCK_BINARY_DIR}/math-libs/BLAS/hipBLAS/stage/include")
include_directories(SYSTEM "${THEROCK_BINARY_DIR}/math-libs/BLAS/hipBLAS-common/stage/include")
include_directories(SYSTEM "${THEROCK_BINARY_DIR}/math-libs/rocRAND/stage/include")

# In HOST_ASAN mode, -fsanitize=address is passed globally but is not supported
# for GPU device targets without xnack+. Clang emits -Woption-ignored for this,
# which MIOpen's -Werror promotes to a hard error. Suppress it here since the
# behavior (host instrumented, device not) is intentional in HOST_ASAN mode.
if(THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  add_compile_options(-Wno-option-ignored)
endif()
