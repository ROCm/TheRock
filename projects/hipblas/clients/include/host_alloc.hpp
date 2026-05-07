/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#include <new>

//!
//! @brief Host free memory w/o swap.  Returns kB or -1 if unknown.
//!
ptrdiff_t host_bytes_available();

//!
//! @brief Return rough estimate of memory used via host_ helper APIs only.
//!
size_t host_bytes_allocated();

//!
//! @brief Count external memory towards limits
//!
void alloc_ptr_use(void* ptr, size_t size);

//!
//! @brief Release counted external memory
//!
void free_ptr_use(void* ptr, bool call_free = false);

//!
//! @brief Allocates memory which can be freed with free.  Returns nullptr if swap required.
//!
void* host_malloc(size_t size);

//!
//! @brief Allocates memory which can be freed with free.  Throws exception if swap required.
//!
inline void* host_malloc_throw(size_t nmemb, size_t size)
{
    void* ptr = host_malloc(nmemb * size);
    if(!ptr)
    {
        throw std::bad_alloc{};
    }
    return ptr;
}

//!
//! @brief Allocates cleared memory which can be freed with free.  Returns nullptr if swap required.
//!
void* host_calloc(size_t nmemb, size_t size);

//!
//! @brief Allocates cleared memory which can be freed with free.  Throws exception if swap required.
//!
inline void* host_calloc_throw(size_t nmemb, size_t size)
{
    void* ptr = host_calloc(nmemb, size);
    if(!ptr)
    {
        throw std::bad_alloc{};
    }
    return ptr;
}

//!
//! @brief Release memory allocated with host_ prefixed allocators
//!
void host_free(void* ptr);

//!
//! @brief  Allocator which allocates with host_malloc_throw
//!
template <class T>
struct host_memory_allocator
{
    using value_type = T;

    host_memory_allocator() = default;

    template <class U>
    host_memory_allocator(const host_memory_allocator<U>&)
    {
    }

    T* allocate(std::size_t n)
    {
        return (T*)host_malloc_throw(n, sizeof(T));
    }

    void deallocate(T* ptr, std::size_t n)
    {
        host_free(ptr);
    }
};

template <class T, class U>
constexpr bool operator==(const host_memory_allocator<T>&, const host_memory_allocator<U>&)
{
    return true;
}

template <class T, class U>
constexpr bool operator!=(const host_memory_allocator<T>&, const host_memory_allocator<U>&)
{
    return false;
}
