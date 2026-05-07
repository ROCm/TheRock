/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Debug.hpp>

#include <mutex>

#ifndef DEBUG_SM
#define DEBUG_SM 0
#endif

#ifndef DEBUG_SM2
#define DEBUG_SM2 0
#endif

namespace rocblaslt
{
    bool Debug::preload() const
    {
        return m_preloadAllKernels;
    }

    bool Debug::printLogAsMarker() const
    {
        return m_printLogAsMarker;
    }

    bool Debug::benchPrintCommand() const
    {
        return m_benchPrintCommand;
    }

    void Debug::setBenchPrint(bool p)
    {
        m_benchPrintCommand = p;
    }

    Debug::Debug()
        : m_value(DEBUG_SM)
        , m_value2(DEBUG_SM2)
    {
        const char* hipblaslt_marker = std::getenv("HIPBLASLT_ENABLE_MARKER");
        if(hipblaslt_marker)
        {
            m_printMarker      = strtol(hipblaslt_marker, nullptr, 0) != 0;
            m_printLogAsMarker = strtol(hipblaslt_marker, nullptr, 0) == 2;
#ifndef HIPBLASLT_ENABLE_MARKER
            if(m_printMarker)
                printf("HIPBLASLT_ENABLE_MARKER is not defined. Please rebuild with -DHIPBLASLT_ENABLE_MARKER=ON\n");
#endif
        }

        const char* hipblaslt_bench = std::getenv("HIPBLASLT_BENCH_PRINT_COMMAND");
        if(hipblaslt_bench)
        {
            m_benchPrintCommand = strtol(hipblaslt_bench, nullptr, 0) != 0;
        }

        const char *hipblaslt_preload = std::getenv("HIPBLASLT_PRELOAD_KERNELS");
        m_preloadAllKernels = hipblaslt_preload && strtol(hipblaslt_preload, nullptr, 0) != 0;
    }

} // namespace rocblaslt
