/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef WIN32
typedef long long ssize_t; /* x64 only supported */
#endif

#include "argument_model.hpp"
#include "hipblas.h"
#include "hipblas_arguments.hpp"
#include "test_cleanup.hpp"

#ifdef GOOGLE_TEST

// Extra macro so that macro arguments get expanded before calling Google Test
#define CHECK_HIP_ERROR2(ERROR) ASSERT_EQ(ERROR, hipSuccess)
#define CHECK_HIP_ERROR(ERROR) CHECK_HIP_ERROR2(ERROR)

#define CHECK_DEVICE_ALLOCATION(ERROR)                   \
    do                                                   \
    {                                                    \
        /* Use error__ in case ERROR contains "error" */ \
        hipError_t error__ = (ERROR);                    \
        if(error__ != hipSuccess)                        \
        {                                                \
            if(error__ == hipErrorOutOfMemory)           \
                GTEST_SKIP() << LIMITED_VRAM_STRING;     \
            else                                         \
                FAIL() << hipGetErrorString(error__);    \
            return;                                      \
        }                                                \
    } while(0)

// This wraps the hipBLAS call with catch_signals_and_exceptions_as_failures().
// By placing it at the hipBLAS call site, memory resources are less likely to
// be leaked in the event of a caught signal.
#define EXPECT_HIPBLAS_STATUS(STATUS, EXPECT)                 \
    do                                                        \
    {                                                         \
        volatile bool signal_or_exception = true;             \
        /* Use status__ in case STATUS contains "status" */   \
        hipblasStatus_t status__;                             \
        catch_signals_and_exceptions_as_failures([&] {        \
            status__            = (STATUS);                   \
            signal_or_exception = false;                      \
        });                                                   \
        if(signal_or_exception)                               \
            return;                                           \
        { /* localize status for ASSERT_EQ message */         \
            hipblasStatus_t status_ = status__;               \
            ASSERT_EQ(status_, EXPECT); /* prints "status" */ \
        }                                                     \
    } while(0)

#else // GOOGLE_TEST

inline void hipblas_expect_status(hipblasStatus_t status, hipblasStatus_t expect)
{
    if(status != expect)
    {
        std::cerr << "hipBLAS status error: Expected " << hipblasStatusToString(expect)
                  << ", received " << hipblasStatusToString(status) << std::endl;
        if(expect == HIPBLAS_STATUS_SUCCESS)
            exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP_ERROR(ERROR)                                                      \
    do                                                                              \
    {                                                                               \
        /* Use error__ in case ERROR contains "error" */                            \
        hipError_t error__ = (ERROR);                                               \
        if(error__ != hipSuccess)                                                   \
        {                                                                           \
            std::cerr << "error: " << hipGetErrorString(error__) << " (" << error__ \
                      << ") at " __FILE__ ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while(0)

#define CHECK_DEVICE_ALLOCATION CHECK_HIP_ERROR

#define EXPECT_HIPBLAS_STATUS hipblas_expect_status

#endif // GOOGLE_TEST

#define CHECK_HIPBLAS_ERROR2(STATUS) EXPECT_HIPBLAS_STATUS(STATUS, HIPBLAS_STATUS_SUCCESS)
#define CHECK_HIPBLAS_ERROR(STATUS) CHECK_HIPBLAS_ERROR2(STATUS)

#ifdef GOOGLE_TEST

// The tests are instantiated by filtering through the HipBLAS_Data stream
// The filter is by category and by the type_filter() and function_filter()
// functions in the testclass
#define INSTANTIATE_TEST_CATEGORY(testclass, category)                                            \
    INSTANTIATE_TEST_SUITE_P(category,                                                            \
                             testclass,                                                           \
                             testing::ValuesIn(HipBLAS_TestData::begin([](const Arguments& arg) { \
                                                   return testclass::function_filter(arg)         \
                                                          && testclass::type_filter(arg);         \
                                               }),                                                \
                                               HipBLAS_TestData::end()),                          \
                             testclass::PrintToStringParamName());

#if defined(GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST)
#define HIPBLAS_ALLOW_UNINSTANTIATED_GTEST(testclass) \
    GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(testclass);
#else
#define HIPBLAS_ALLOW_UNINSTANTIATED_GTEST(testclass)
#endif

// Instantiate all test categories
#define INSTANTIATE_TEST_CATEGORIES(testclass)    \
    HIPBLAS_ALLOW_UNINSTANTIATED_GTEST(testclass) \
    INSTANTIATE_TEST_CATEGORY(testclass, _)

// Category based instantiation requires pass of large yaml data for each category
// Using single '_' named category and category name is moved to test name prefix
// gtest_filter should be able to select same test subsets
// INSTANTIATE_TEST_CATEGORY(testclass, quick)       \
// INSTANTIATE_TEST_CATEGORY(testclass, pre_checkin) \
// INSTANTIATE_TEST_CATEGORY(testclass, nightly)     \
// INSTANTIATE_TEST_CATEGORY(testclass, multi_gpu)   \
// INSTANTIATE_TEST_CATEGORY(testclass, HMM)         \
// INSTANTIATE_TEST_CATEGORY(testclass, known_bug)

// Function to catch signals and exceptions as failures
void catch_signals_and_exceptions_as_failures(std::function<void()> test, bool set_alarm = false);

// Macro to call catch_signals_and_exceptions_as_failures() with a lambda expression
#define CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(test) \
    catch_signals_and_exceptions_as_failures([&] { test; }, true)

/* ============================================================================================ */
/*! \brief  Normalized test name to conform to Google Tests */
// The template parameter is only used to generate multiple instantiations with distinct static local variables
template <typename>
class HipBLAS_TestName
{
    std::ostringstream m_str;

public:
    explicit HipBLAS_TestName(const char* name)
    {
        m_str << name << '_';
    }

    // Convert stream to normalized Google Test name
    // rvalue reference qualified so that it can only be called once
    // The name should only be generated once before the stream is destroyed
    operator std::string() &&
    {
        // This table is private to each instantiation of HipBLAS_TestName
        // Placed inside function to avoid dependency on initialization order
        static std::unordered_map<std::string, size_t>* table = test_cleanup::allocate(&table);
        std::string HipBLAS_TestName_to_string(std::unordered_map<std::string, size_t>&,
                                               const std::ostringstream&);
        return HipBLAS_TestName_to_string(*table, m_str);
    }

    // Stream output operations
    template <typename U> // Lvalue LHS
    friend HipBLAS_TestName& operator<<(HipBLAS_TestName& name, U&& obj)
    {
        name.m_str << std::forward<U>(obj);
        return name;
    }

    template <typename U> // Rvalue LHS
    friend HipBLAS_TestName&& operator<<(HipBLAS_TestName&& name, U&& obj)
    {
        name.m_str << std::forward<U>(obj);
        return std::move(name);
    }

    HipBLAS_TestName()                        = default;
    HipBLAS_TestName(const HipBLAS_TestName&) = delete;
    HipBLAS_TestName& operator=(const HipBLAS_TestName&) = delete;
};

bool hipblas_client_global_filters(const Arguments& args);

// ----------------------------------------------------------------------------
// Hipblas_Test base class. All non-legacy hipblas Google tests derive from it.
// It defines a type_filter_functor() and a PrintToStringParamName class
// which calls name_suffix() in the derived class to form the test name suffix.
// ----------------------------------------------------------------------------
template <typename TEST, template <typename...> class FILTER>
class HipBLAS_Test : public testing::TestWithParam<Arguments>
{
protected:
    // This template functor returns true if the type arguments are valid.
    // It converts a FILTER specialization to bool to test type matching.
    template <typename... T>
    struct type_filter_functor
    {
        bool operator()(const Arguments& args)
        {
            // additional global filters applied first
            if(!hipblas_client_global_filters(args))
                return false;

            // type filters
            return static_cast<bool>(FILTER<T...>{});
        }
    };

public:
    // Wrapper functor class which calls name_suffix()
    struct PrintToStringParamName
    {
        std::string operator()(const testing::TestParamInfo<Arguments>& info) const
        {
            std::string name(info.param.category);
            name += TEST::name_suffix(info.param);
            return name;
        }
    };
};

// Function to set up signal handlers
void hipblas_test_sigaction();

#endif // GOOGLE_TEST

// ----------------------------------------------------------------------------
// Normal tests which return true when converted to bool
// ----------------------------------------------------------------------------
struct hipblas_test_valid
{
    // Return true to indicate the type combination is valid, for filtering
    virtual explicit operator bool() final
    {
        return true;
    }

    // Require derived class to define functor which takes (const Arguments &)
    virtual void operator()(const Arguments&) = 0;

    virtual ~hipblas_test_valid() = default;
};

// ----------------------------------------------------------------------------
// Error case which returns false when converted to bool. A void specialization
// of the FILTER class template above, should be derived from this class, in
// order to indicate that the type combination is invalid.
// ----------------------------------------------------------------------------
struct hipblas_test_invalid
{
    // Return false to indicate the type combination is invalid, for filtering
    virtual explicit operator bool() final
    {
        return false;
    }

    // If this specialization is actually called, print fatal error message
    virtual void operator()(const Arguments&) final
    {
        static constexpr char msg[] = "Internal error: Test called with invalid types";

#ifdef GOOGLE_TEST
        FAIL() << msg;
#else
        std::cerr << msg << std::endl;
        abort();
#endif
    }

    virtual ~hipblas_test_invalid() = default;
};
