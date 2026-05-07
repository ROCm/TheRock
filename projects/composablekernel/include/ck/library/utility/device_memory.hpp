// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <math.h>
#include "ck/library/utility/device_tensor_generator.hpp"

namespace ck {

template <typename T>
__global__ void set_buffer_value(T* p, T x, uint64_t buffer_element_size)
{
    for(uint64_t i = threadIdx.x; i < buffer_element_size; i += blockDim.x)
    {
        p[i] = x;
    }
}

/**
 * @brief Container for storing data in GPU device memory
 *
 */
struct DeviceMem
{
    DeviceMem() : mpDeviceBuf(nullptr), mMemSize(0) {}
    DeviceMem(std::size_t mem_size);
    void Realloc(std::size_t mem_size);
    void* GetDeviceBuffer() const;
    std::size_t GetBufferSize() const;
    void ToDevice(const void* p) const;
    void ToDevice(const void* p, const std::size_t cpySize) const;
    void FromDevice(void* p) const;
    void FromDevice(void* p, const std::size_t cpySize) const;
    void SetZero() const;
    template <typename T>
    void SetValue(T x) const;
    template <typename T>
    void FillUniformRandInteger(int min_value, int max_value);
    template <typename T>
    void FillUniformRandFp(float min_value, float max_value);
    template <typename T>
    void FillNormalRandFp(float sigma, float mean);
    ~DeviceMem();

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

template <typename T>
void DeviceMem::SetValue(T x) const
{
    if(mMemSize % sizeof(T) != 0)
    {
        throw std::runtime_error("wrong! not entire DeviceMem will be set");
    }

    set_buffer_value<T><<<1, 1024>>>(static_cast<T*>(mpDeviceBuf), x, mMemSize / sizeof(T));
}

template <typename T>
void DeviceMem::FillUniformRandInteger(int min_value, int max_value)
{
    if(mMemSize % sizeof(T) != 0)
    {
        throw std::runtime_error("wrong! not entire DeviceMem will be filled");
    }
    if(max_value - min_value <= 1)
    {
        throw std::runtime_error("Error while filling device tensor with random integer data: max "
                                 "value must be at least 2 greater than min value, otherwise "
                                 "tensor will be filled by a constant value (end is exclusive)");
    }
    if(max_value - 1 == min_value || max_value - 1 == max_value)
    {
        throw std::runtime_error("Error while filling device tensor with random integer data: "
                                 "insufficient precision in specified range");
    }

    size_t packed_size = packed_size_v<T>;
    fill_tensor_uniform_rand_int_values<<<256, 256>>>(
        static_cast<T*>(mpDeviceBuf), min_value, max_value, (mMemSize * packed_size) / sizeof(T));
}

template <typename T>
void DeviceMem::FillUniformRandFp(float min_value, float max_value)
{
    if(mMemSize % sizeof(T) != 0)
    {
        throw std::runtime_error("wrong! not entire DeviceMem will be filled");
    }

    size_t packed_size = packed_size_v<T>;
    fill_tensor_uniform_rand_fp_values<<<256, 256>>>(
        static_cast<T*>(mpDeviceBuf), min_value, max_value, (mMemSize * packed_size) / sizeof(T));
}

template <typename T>
void DeviceMem::FillNormalRandFp(float sigma, float mean)
{

    fill_tensor_norm_rand_fp_values<<<256, 256>>>(
        static_cast<T*>(mpDeviceBuf), sigma, mean, mMemSize / sizeof(T));
}
} // namespace ck
