// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_KERNEL
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_KERNEL

#include <hip/hip_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace rtc {

struct kernel_argument
{
    template <class T,
              class U = std::remove_reference_t<T>,
              class   = std::enable_if_t<not std::is_base_of<kernel_argument, T>{}>>
    kernel_argument(T&& x) : size(sizeof(U)), align(alignof(U)), data(&x) // NOLINT
    {
    }
    std::size_t size;
    std::size_t align;
    void* data;
};

std::vector<char> pack_args(const std::vector<kernel_argument>& args);

struct kernel_impl;

struct kernel
{
    kernel() = default;
    kernel(const char* image, const std::string& name);
    template <class T>
    kernel(const std::vector<T>& image, const std::string& name)
        : kernel(reinterpret_cast<const char*>(image.data()), name)
    {
        static_assert(sizeof(T) == 1, "Only byte types");
    }

    void launch(hipStream_t stream,
                std::size_t global,
                std::size_t local,
                const std::vector<kernel_argument>& args) const;

    void launch(hipStream_t stream,
                std::size_t global,
                std::size_t local,
                std::vector<void*> args) const;

    template <class... Ts>
    auto launch(hipStream_t stream, std::size_t global, std::size_t local, Ts... zs) const
    {
        return [=, this](auto&&... xs) {
            launch(stream, global, local, std::vector<kernel_argument>{xs...}, zs...);
        };
    }

    private:
    std::shared_ptr<kernel_impl> impl;
};
} // namespace rtc

#endif
