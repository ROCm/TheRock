# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_batchnorm_fwd_instance_id batchnorm_fwd_instance_id.cpp)
target_link_libraries(client_batchnorm_fwd_instance_id PRIVATE composable_kernel::device_other_operations)
