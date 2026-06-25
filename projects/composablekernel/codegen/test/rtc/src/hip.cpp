// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rtc/hip.hpp>
#include <rtc/manage_ptr.hpp>
#include <stdexcept>
#include <cassert>
#include <iostream>

namespace rtc {

using hip_ptr = RTC_MANAGE_PTR(void, hipFree);

std::string hip_error(int error) { return hipGetErrorString(static_cast<hipError_t>(error)); }

int get_device_id()
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        throw std::runtime_error("No device");
    return device;
}

std::string get_device_name()
{
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, get_device_id());
    if(status != hipSuccess)
        throw std::runtime_error("Failed to get device properties");
    return props.gcnArchName;
}

bool is_device_ptr(const void* ptr)
{
    hipPointerAttribute_t attr;
    auto status = hipPointerGetAttributes(&attr, ptr);
    if(status != hipSuccess)
        return false;
    return attr.type == hipMemoryTypeDevice;
}

void gpu_sync()
{
    auto status = hipDeviceSynchronize();
    if(status != hipSuccess)
        throw std::runtime_error("hip device synchronization failed: " + hip_error(status));
}

std::size_t get_available_gpu_memory()
{
    size_t free;
    size_t total;
    auto status = hipMemGetInfo(&free, &total);
    if(status != hipSuccess)
    {
        std::cerr << "Failed getting available memory: " + hip_error(status) << std::endl;
        return (8ull * 1024ull * 1024ull * 1024ull);
    }
    return free;
}

std::shared_ptr<void> allocate_gpu(std::size_t sz, bool host)
{
    if(sz > get_available_gpu_memory())
        throw std::runtime_error("Memory not available to allocate buffer: " + std::to_string(sz));
    void* alloc_ptr = nullptr;
    auto status     = host ? hipHostMalloc(&alloc_ptr, sz) : hipMalloc(&alloc_ptr, sz);
    if(status != hipSuccess)
    {
        if(host)
            throw std::runtime_error("Gpu allocation failed: " + hip_error(status));
        else
            return allocate_gpu(sz, true);
    }
    assert(alloc_ptr != nullptr);
    std::shared_ptr<void> result = share(hip_ptr{alloc_ptr});
    return result;
}

std::shared_ptr<void> write_to_gpu(const void* x, std::size_t sz, bool host)
{
    gpu_sync();
    auto result = allocate_gpu(sz, host);
    assert(is_device_ptr(result.get()));
    assert(not is_device_ptr(x));
    auto status = hipMemcpy(result.get(), x, sz, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        throw std::runtime_error("Copy to gpu failed: " + hip_error(status));
    return result;
}

std::shared_ptr<void> read_from_gpu(const void* x, std::size_t sz)
{
    gpu_sync();
    std::shared_ptr<char> result(new char[sz]);
    assert(not is_device_ptr(result.get()));
    if(not is_device_ptr(x))
    {
        throw std::runtime_error(
            "read_from_gpu() requires Src buffer to be on the GPU, Copy from gpu failed\n");
    }
    auto status = hipMemcpy(result.get(), x, sz, hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        throw std::runtime_error("Copy from gpu failed: " + hip_error(status)); // NOLINT
    return std::static_pointer_cast<void>(result);
}

} // namespace rtc
