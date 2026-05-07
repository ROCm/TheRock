// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <hip/hip_runtime.h>

namespace ck {

// To be removed, which really does not tell the location of failed HIP functional call
inline void hip_check_error(hipError_t x)
{
    if(x != hipSuccess)
    {
        std::ostringstream ss;
        ss << "HIP runtime error: " << hipGetErrorString(x) << ". " << "hip_check_error.hpp" << ": "
           << __LINE__ << "in function: " << __func__;
        throw std::runtime_error(ss.str());
    }
}

#define HIP_CHECK_ERROR(retval_or_funcall)                                         \
    do                                                                             \
    {                                                                              \
        hipError_t _tmpVal = retval_or_funcall;                                    \
        if(_tmpVal != hipSuccess)                                                  \
        {                                                                          \
            std::ostringstream ostr;                                               \
            ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ << ") " \
                 << hipGetErrorString(_tmpVal);                                    \
            throw std::runtime_error(ostr.str());                                  \
        }                                                                          \
    } while(0)

} // namespace ck
