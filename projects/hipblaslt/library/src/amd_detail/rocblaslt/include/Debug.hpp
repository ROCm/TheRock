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

#pragma once

#include <cstdlib>
#include <string>
#ifdef HIPBLASLT_ENABLE_MARKER
#include <roctracer/roctx.h>
#endif

namespace rocblaslt
{
    template <typename Class>
    class LazySingleton
    {
    public:
        static Class& Instance()
        {
            static Class rocInstance;

            return rocInstance;
        }

    private:
    };

    /**
 * @brief Common place for defining flags which enable debug behaviour.
 */
    class Debug : public LazySingleton<Debug>
    {
    public:
        __attribute__((always_inline)) inline void markerStart(const char* name) const
        {
#ifdef HIPBLASLT_ENABLE_MARKER
            if(m_printMarker)
            {
                roctxRangePush(name);
            }
#endif
        }

        __attribute__((always_inline)) inline void markerStop() const
        {
#ifdef HIPBLASLT_ENABLE_MARKER
            if(m_printMarker)
            {
                roctxRangePop();
            }
#endif
        }

        __attribute__((always_inline)) inline void logMarkerStart(const char* name) const
        {
#ifdef HIPBLASLT_ENABLE_MARKER
            if(m_printLogAsMarker)
            {
                roctxRangePush(name);
            }
#endif
        }

        __attribute__((always_inline)) inline void logMarkerStop() const
        {
#ifdef HIPBLASLT_ENABLE_MARKER
            if(m_printLogAsMarker)
            {
                roctxRangePop();
            }
#endif
        }

        bool printLogAsMarker() const;

        bool benchPrintCommand() const;
        void setBenchPrint(bool p);

        bool preload() const;

    private:
        friend LazySingleton<Debug>;

        int         m_value;
        int         m_value2;
        bool        m_printMarker       = false;
        bool        m_printLogAsMarker  = false;
        bool        m_preloadAllKernels = false;
        bool        m_benchPrintCommand = false;

        Debug();
    };
} // namespace rocblaslt
