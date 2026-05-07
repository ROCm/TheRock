// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <sstream>
#include <vector>

#include "ck/ck.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/**
 * Calculates the maximum number of subsequent elements of the fast changing dimension
 * that are consecutive in memory.
 *
 * Example:
 *   NumDimM = 2, NumDimK = 3
 *   A shape =   [  2,   3,  4, 5, 6]
 *   A strides = [360, 120, 30, 6, 1]
 *                |  M   |  |   K  |
 *   It follows from strides that K is FCD and all the subsequent elements of K are consecutive
 *   in memory.
 *   But if strides were [360, 120, 6, 24, 1], then only 6 subsequent elements of K would be
 *   consecutive in memory.
 *
 * Assumes that the dimensions are split into two groups of `NumDim1` and `NumDim2` dimensions.
 */
template <index_t NumDim1, index_t NumDim2>
auto CalculateMaxRead(const std::vector<index_t>& lengths, const std::vector<index_t>& strides)
{
    if(lengths.size() != NumDim1 + NumDim2)
    {
        std::ostringstream err;
        err << "Incorrect number of lengths in " << "device_contraction_utils.hpp" << ":"
            << __LINE__ << ", in function: " << __func__;
        throw std::runtime_error(err.str());
    }
    if(strides.size() != NumDim1 + NumDim2)
    {
        std::ostringstream err;
        err << "Incorrect number of strides in " << "device_contraction_utils.hpp" << ":"
            << __LINE__ << ", in function: " << __func__;
        throw std::runtime_error(err.str());
    }

    // Determine the beginning and end idx of the group representing the FCD.
    index_t begin_idx, end_idx, continous_dim, consecutive_stride = 1;
    if(strides[NumDim1 - 1] == 1 && strides[NumDim1 + NumDim2 - 1] == 1)
    {
        // MZ or KZ are ones
        bool dims1_are_ones = true;
        for(index_t dim_idx = 0; dim_idx < NumDim1; dim_idx++)
        {
            if(lengths[dim_idx] != 1)
            {
                dims1_are_ones = false;
            }
        }

        if(dims1_are_ones)
        {
            begin_idx     = NumDim1;
            end_idx       = NumDim1 + NumDim2 - 1;
            continous_dim = 1;
        }
        else
        {
            begin_idx     = 0;
            end_idx       = NumDim1 - 1;
            continous_dim = 0;
        }
    }
    else if(strides[NumDim1 - 1] == 1)
    {
        begin_idx     = 0;
        end_idx       = NumDim1 - 1;
        continous_dim = 0;
    }
    else if(strides[NumDim1 + NumDim2 - 1] == 1)
    {
        begin_idx     = NumDim1;
        end_idx       = NumDim1 + NumDim2 - 1;
        continous_dim = 1;
    }
    else
    {
        // The dimension consecutive in memory is not the last dimension of any group, so only
        // one element can be read/written at once.
        consecutive_stride = 1;
        continous_dim      = 0;
        return make_tuple(continous_dim, consecutive_stride);
    }

    for(index_t dim_idx = end_idx; dim_idx >= begin_idx; --dim_idx)
    {
        if(strides[dim_idx] == consecutive_stride)
        {
            consecutive_stride *= lengths[dim_idx];
        }
        else
        {
            break;
        }
    }
    const index_t max_subsequent_elems = consecutive_stride;
    return make_tuple(continous_dim, max_subsequent_elems);
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
