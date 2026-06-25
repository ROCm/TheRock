// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include <hip/hip_runtime.h>
#include <vector>

namespace ck {
namespace ref {

// RAII wrapper for device memory to prevent leaks
struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&p_mem_), mem_size));
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

// Helper function to map layout dimension character to index in lengths array
// lengths array structure: [G, N/K, C/K, spatial...]
inline int map_dim_char_to_index(char dim_char, index_t ndim_spatial, bool is_weight)
{
    // G dimension
    if(dim_char == 'G')
        return 0;

    // Batch/output channels dimension (N for input/output, K for weight's first non-G dim)
    if(dim_char == 'N')
        return 1;
    if(dim_char == 'K' && is_weight)
        return 1;

    // Channel dimension (C for input/weight, K for output)
    if(dim_char == 'C')
        return 2;
    if(dim_char == 'K' && !is_weight)
        return 2;

    // Spatial dimensions - map based on ndim_spatial
    // Input/Output use: D/H/W, Weight uses: Z/Y/X
    if(ndim_spatial == 1)
    {
        if(dim_char == 'W' || dim_char == 'X')
            return 3;
    }
    else if(ndim_spatial == 2)
    {
        if(dim_char == 'H' || dim_char == 'Y')
            return 3;
        if(dim_char == 'W' || dim_char == 'X')
            return 4;
    }
    else if(ndim_spatial == 3)
    {
        if(dim_char == 'D' || dim_char == 'Z')
            return 3;
        if(dim_char == 'H' || dim_char == 'Y')
            return 4;
        if(dim_char == 'W' || dim_char == 'X')
            return 5;
    }

    // Should not reach here
    return -1;
}

// Template function to compute layout-aware strides based on layout name
// The layout name directly encodes memory ordering from left to right
template <typename Layout>
inline std::vector<index_t> compute_conv_tensor_strides(const std::vector<index_t>& lengths,
                                                        index_t ndim_spatial)
{
    constexpr const char* layout_name = Layout::name;
    const int num_dims                = static_cast<int>(lengths.size());
    std::vector<index_t> strides(num_dims, 0);

    // Determine if this is a weight tensor (has 'K' but not 'N')
    bool has_k = false;
    bool has_n = false;
    for(const char* p = layout_name; *p != '\0'; ++p)
    {
        if(*p == 'K')
            has_k = true;
        if(*p == 'N')
            has_n = true;
    }
    bool is_weight = has_k && !has_n;

    // Build dimension ordering from layout name (parse string)
    std::vector<char> dim_order;
    const char dim_chars[] = {'G', 'N', 'K', 'C', 'D', 'H', 'W', 'X', 'Y', 'Z'};
    for(const char* p = layout_name; *p != '\0'; ++p)
    {
        char c = *p;
        // Skip underscores (strided layouts)
        if(c == '_')
            continue;
        // Valid dimension characters
        if(std::find(std::begin(dim_chars), std::end(dim_chars), c) != std::end(dim_chars))
        {
            dim_order.push_back(c);
        }
    }

    // Compute strides: process from right to left (innermost to outermost)
    index_t stride = 1;
    for(int i = static_cast<int>(dim_order.size()) - 1; i >= 0; --i)
    {
        char dim_char  = dim_order[i];
        int length_idx = map_dim_char_to_index(dim_char, ndim_spatial, is_weight);

        if(length_idx >= 0 && length_idx < num_dims)
        {
            strides[length_idx] = stride;
            stride *= lengths[length_idx];
        }
    }

    return strides;
}

// Unified kernel for strided tensor copy operations
// IsUnpack=false: Pack strided -> contiguous
// IsUnpack=true:  Unpack contiguous -> strided
template <typename DataType, bool IsUnpack>
__global__ void strided_copy_kernel(const DataType* __restrict__ src,
                                    DataType* __restrict__ dst,
                                    const index_t* tensor_lengths,
                                    const index_t* strided_strides,
                                    int num_dims,
                                    long_index_t total_elements)
{
    const long_index_t tid         = blockIdx.x * blockDim.x + threadIdx.x;
    const long_index_t num_threads = blockDim.x * gridDim.x;

    for(long_index_t linear_idx = tid; linear_idx < total_elements; linear_idx += num_threads)
    {
        // Compute strided index from linear index
        long_index_t remaining   = linear_idx;
        long_index_t strided_idx = 0;

        for(int dim = num_dims - 1; dim >= 0; --dim)
        {
            index_t coord = remaining % tensor_lengths[dim];
            remaining /= tensor_lengths[dim];
            strided_idx += coord * strided_strides[dim];
        }

        // Direction determines which is src and which is dst
        if constexpr(IsUnpack)
        {
            // Unpack: src is contiguous (linear_idx), dst is strided (strided_idx)
            dst[strided_idx] = src[linear_idx];
        }
        else
        {
            // Pack: src is strided (strided_idx), dst is contiguous (linear_idx)
            dst[linear_idx] = src[strided_idx];
        }
    }
}

} // namespace ref
} // namespace ck
