/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipblas_test.hpp"
#include "utility.h"

#include <cerrno>
#include <csetjmp>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <regex>
#ifdef WIN32
#include <windows.h>
#define strcasecmp(A, B) _stricmp(A, B)
#else
#include <pthread.h>
#include <unistd.h>
#endif

/*********************************************
 * callback function
 *********************************************/
thread_local std::unique_ptr<std::function<void(hipblasHandle_t)>> t_set_stream_callback;

/*********************************************
 * Signal-handling for detecting test faults *
 *********************************************/

// State of the signal handler
static thread_local struct
{
    // Whether sigjmp_buf is set and catching signals is enabled
    volatile sig_atomic_t enabled = false;

    // sigjmp_buf describing stack frame to go back to
#ifndef WIN32
    sigjmp_buf sigjmp;
#else
    jmp_buf sigjmp;
#endif

    // The signal which was received
    volatile sig_atomic_t signal;
} t_handler;

// Signal handler (must have external "C" linkage)
extern "C" void hipblas_test_signal_handler(int sig)
{
    int saved_errno = errno; // Save errno

    // If the signal handler is disabled, because we're not in the middle of
    // running a hipBLAS test, restore this signal's disposition to default,
    // and reraise the signal
    if(!t_handler.enabled)
    {
        signal(sig, SIG_DFL);
        errno = saved_errno;
        raise(sig);
        return;
    }

#ifndef WIN32
    // If this is an alarm timeout, we abort
    if(sig == SIGALRM)
    {
        static constexpr char msg[]
            = "\nAborting tests due to an alarm timeout.\n\n"
              "This could be due to a deadlock caused by mutexes being left locked\n"
              "after a previous test's signal was caught and partially recovered from.\n";
        // We must use write() because it's async-signal-safe and other IO might be blocked
        auto n = write(STDERR_FILENO, msg, sizeof(msg) - 1);
        abort();
    }
#endif

    // Jump back to the handler code after setting handler.signal
    // Note: This bypasses stack unwinding, and may lead to memory leaks, but
    // it is better than crashing.
    t_handler.signal = sig;
    errno            = saved_errno;
#ifndef WIN32
    siglongjmp(t_handler.sigjmp, true);
#else
    longjmp(t_handler.sigjmp, true);
#endif
}

// Set up signal handlers
void hipblas_test_sigaction()
{
#ifndef WIN32
    struct sigaction act;
    act.sa_flags = 0;
    sigfillset(&act.sa_mask);
    act.sa_handler = hipblas_test_signal_handler;

    // Catch SIGALRM and synchronous signals
    for(int sig : {SIGALRM, SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV})
        sigaction(sig, &act, nullptr);
#else
    for(int sig : {SIGABRT, SIGFPE, SIGILL, SIGINT, SIGSEGV, SIGTERM})
        signal(sig, hipblas_test_signal_handler);
#endif
}

static const unsigned test_timeout = [] {
    // Number of seconds each test is allowed to take before all testing is killed.
    constexpr unsigned TEST_TIMEOUT = 600;
    unsigned           timeout;
    const char*        env = getenv("HIPBLAS_TEST_TIMEOUT");
    return env && sscanf(env, "%u", &timeout) == 1 ? timeout : TEST_TIMEOUT;
}();

// Lambda wrapper which detects signals and exceptions in an invokable function
void catch_signals_and_exceptions_as_failures(std::function<void()> test, bool set_alarm)
{
    // Save the current handler (to allow nested calls to this function)
    auto old_handler = t_handler;

#ifndef WIN32
    // Set up the return point, and handle siglongjmp returning back to here
    if(sigsetjmp(t_handler.sigjmp, true))
    {
#if(__GLIBC__ < 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 32)
        FAIL() << "Received " << sys_siglist[t_handler.signal] << " signal";
#else
        FAIL() << "Received " << sigdescr_np(t_handler.signal) << " signal";
#endif
    }
#else
    if(setjmp(t_handler.sigjmp))
    {
        FAIL() << "Received signal";
    }
#endif
    else
    {
#ifndef WIN32
        // Alarm to detect deadlocks or hangs
        if(set_alarm)
            alarm(test_timeout);
#endif
        // Enable the signal handler
        t_handler.enabled = true;

        // Run the test function, catching signals and exceptions
        try
        {
            test();
        }
        catch(const std::bad_alloc& e)
        {
            GTEST_SKIP() << LIMITED_RAM_STRING;
        }
        catch(const std::exception& e)
        {
            FAIL() << "Received uncaught exception: " << e.what();
        }
        catch(...)
        {
            FAIL() << "Received uncaught exception";
        }
    }

#ifndef WIN32
    // Cancel the alarm if it was set
    if(set_alarm)
        alarm(0);
#endif
    // Restore the previous handler
    t_handler = old_handler;

    if(hipPeekAtLastError() != hipSuccess)
    {
        std::cerr << "hipGetLastError at end of test: "
                  << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
        (void)hipblas_internal_convert_hip_to_hipblas_status_and_log(
            hipGetLastError()); // clear last error
    }
}

// Convert stream to normalized Google Test name
std::string HipBLAS_TestName_to_string(std::unordered_map<std::string, size_t>& table,
                                       const std::ostringstream&                str)
{
    std::string name{str.str()};

    // Remove trailing underscore
    if(!name.empty() && name.back() == '_')
        name.pop_back();

    // If name is empty, make it 1
    if(name.empty())
        name = "1";

    // Warn about unset letter parameters
    if(name.find('*') != name.npos)
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                     "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                     "Warning: Character * found in name."
                     " This means a required letter parameter\n"
                     "(e.g., transA, diag, etc.) has not been set in the YAML file."
                     " Check the YAML file.\n"
                     "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                     "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                  << std::endl;

    // Replace non-alphanumeric characters with letters
    std::replace(name.begin(), name.end(), '-', 'n'); // minus
    std::replace(name.begin(), name.end(), '.', 'p'); // decimal point

    // Complex (A,B) is replaced with ArBi
    name.erase(std::remove(name.begin(), name.end(), '('), name.end());
    std::replace(name.begin(), name.end(), ',', 'r');
    std::replace(name.begin(), name.end(), ')', 'i');

    // If parameters are repeated, append an incrementing suffix
    auto p = table.find(name);
    if(p != table.end())
        name += "_t" + std::to_string(++p->second);
    else
        table[name] = 1;

    return name;
}

static const char* const validCategories[]
    = {"quick", "pre_checkin", "nightly", "multi_gpu", "HMM", "known_bug", NULL};

static bool valid_category(const char* category)
{
    int i = 0;
    while(validCategories[i])
    {
        if(!strcmp(category, validCategories[i++]))
            return true;
    }
    return false;
}

bool hipblas_client_global_filters(const Arguments& args)
{
    static std::string gpu_arch = getArchString();

#ifdef WIN32
    static constexpr hipblas_client_os os = hipblas_client_os::WINDOWS;
#else
    static constexpr hipblas_client_os os      = hipblas_client_os::LINUX;
#endif

#ifdef __HIP_PLATFORM_NVCC__
    static constexpr hipblas_backend backend = hipblas_backend::NVIDIA;
#else
    static constexpr hipblas_backend   backend = hipblas_backend::AMD;
#endif

#if defined(__HIP_PLATFORM_NVCC__) && CUBLAS_VER_MAJOR < 12
    if(args.api == hipblas_client_api::FORTRAN_64 || args.api == hipblas_client_api::C_64)
        return false;
#endif

    if(!(args.os_flags & os))
        return false;

    if(!(args.backend_flags & backend))
        return false;

    if(args.gpu_arch[0] && !gpu_arch_match(gpu_arch, args.gpu_arch))
        return false;

    return true;
}
