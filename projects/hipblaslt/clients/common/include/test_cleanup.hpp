/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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
 *******************************************************************************/

#pragma once

#include <functional>
#include <stack>
#include <utility>

/*! \brief  Test cleanup handler. Frees memory or performs other cleanup at
specified points in program. */

class test_cleanup
{
    static auto& stack()
    {
        // Placed inside function to avoid dependency on initialization order
        static std::stack<std::function<void()>> stack;
        return stack;
    }

public:
    // Run all cleanup handlers pushed so far, in LIFO order
    static void cleanup()
    {
        while(!stack().empty())
        {
            stack().top()();
            stack().pop();
        }
    }

    // Create an object and register a cleanup handler
    template <typename T, typename... Args>
    static T* allocate(T** ptr, Args&&... args)
    {
        *ptr = nullptr;
        stack().push([=] {
            delete *ptr;
            *ptr = nullptr;
        });
        return new T(std::forward<Args>(args)...);
    }
};
