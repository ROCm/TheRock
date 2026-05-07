// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_HIP
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_HIP

#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <stdexcept>

namespace rtc {

template <class T>
struct buffer
{
    buffer() : ptr(), n(0) {}
    buffer(std::shared_ptr<T> p, std::size_t sz) : ptr(p), n(sz) {}
    buffer(std::shared_ptr<void> p, std::size_t sz)
        : ptr(std::reinterpret_pointer_cast<T>(p)), n(sz)
    {
    }
    explicit buffer(std::size_t sz) : ptr(new T[sz]), n(sz) {}
    T* begin() { return data(); }
    T* end() { return data() + size(); }
    const T* begin() const { return data(); }
    const T* end() const { return data() + size(); }

    T& front() { return data()[0]; }
    T& back() { return data()[size() - 1]; }
    T& operator[](std::size_t i) { return data()[i]; }
    T& at(std::size_t i)
    {
        if(i >= size())
            throw std::runtime_error("Out of bounds");
        return data()[i];
    }

    const T& front() const { return data()[0]; }
    const T& back() const { return data()[size() - 1]; }
    const T& operator[](std::size_t i) const { return data()[i]; }
    const T& at(std::size_t i) const
    {
        if(i >= size())
            throw std::runtime_error("Out of bounds");
        return data()[i];
    }
    const T* data() const { return ptr.get(); }
    T* data() { return ptr.get(); }

    std::size_t size() const { return n; }
    std::size_t bytes() const { return size() * sizeof(T); }

    bool empty() const { return size() == 0; }

    private:
    std::shared_ptr<T> ptr;
    std::size_t n;
};

std::string get_device_name();
std::string hip_error(int error);

std::shared_ptr<void> allocate_gpu(std::size_t sz, bool host = false);
std::shared_ptr<void> write_to_gpu(const void* x, std::size_t sz, bool host = false);
std::shared_ptr<void> read_from_gpu(const void* x, std::size_t sz);

template <class T>
buffer<T> to_gpu(const buffer<T>& input)
{
    return {write_to_gpu(input.data(), input.bytes()), input.size()};
}

template <class T>
buffer<T> from_gpu(const buffer<T>& input)
{
    return {read_from_gpu(input.data(), input.bytes()), input.size()};
}

} // namespace rtc

#endif
