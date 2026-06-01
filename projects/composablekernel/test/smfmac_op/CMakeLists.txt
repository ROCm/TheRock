# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_smfmac_op smfmac_op_xdl.cpp)
target_link_libraries(test_smfmac_op PRIVATE utility)
