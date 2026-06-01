// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_MANAGE_POINTER
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_MANAGE_POINTER

#include <type_traits>
#include <memory>

namespace rtc {
template <class F, F f>
struct manage_deleter
{
    template <class T>
    void operator()(T* x) const
    {
        if(x != nullptr)
        {
            (void)f(x);
        }
    }
};

struct null_deleter
{
    template <class T>
    void operator()(T*) const
    {
    }
};

template <class T, class F, F f>
using manage_ptr = std::unique_ptr<T, manage_deleter<F, f>>;

template <class T>
struct element_type
{
    using type = typename T::element_type;
};

template <class T>
using remove_ptr = typename std::
    conditional_t<std::is_pointer<T>{}, std::remove_pointer<T>, element_type<T>>::type;

template <class T>
using shared = std::shared_ptr<remove_ptr<T>>;

template <class T>
shared<T> share(T p)
{
    return shared<T>{std::move(p)};
}

#define RTC_MANAGE_PTR(T, F) rtc::manage_ptr<std::remove_pointer_t<T>, decltype(&F), &F>

} // namespace rtc

#endif
