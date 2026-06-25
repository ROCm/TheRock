// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_util.hpp"

ck_tile::index_t get_cu_count()
{
    hipDeviceProp_t dev_prop;
    hipDevice_t dev;
    ck_tile::hip_check_error(hipGetDevice(&dev));
    ck_tile::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
    return dev_prop.multiProcessorCount;
}
