# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_example_executable(example_convnd_bwd_data_xdl_fp16 convnd_bwd_data_xdl_fp16.cpp)
if(result EQUAL 0)
    target_link_libraries(example_convnd_bwd_data_xdl_fp16 PRIVATE utility)
endif()

add_example_executable(example_convnd_bwd_data_dl_fp16 convnd_bwd_data_dl_fp16.cpp)
if(result EQUAL 0)
  target_link_libraries(example_convnd_bwd_data_dl_fp16 PRIVATE utility)
endif()
