# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_groupnorm_bwd_data groupnorm_bwd_data.cpp)
target_link_libraries(client_groupnorm_bwd_data PRIVATE composable_kernel::device_other_operations)

add_executable(client_groupnorm_bwd_gamma_beta groupnorm_bwd_gamma_beta.cpp)
target_link_libraries(client_groupnorm_bwd_gamma_beta PRIVATE composable_kernel::device_other_operations)

add_executable(client_groupnorm_swish_fwd groupnorm_swish_fwd.cpp)
target_link_libraries(client_groupnorm_swish_fwd PRIVATE composable_kernel::device_other_operations)
