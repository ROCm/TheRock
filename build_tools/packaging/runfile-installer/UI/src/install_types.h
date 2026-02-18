/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _INSTALL_TYPES_H
#define _INSTALL_TYPES_H

#include <ncurses.h>
#include <menu.h>
#include <form.h>

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include <fcntl.h>
#include <sys/wait.h>

#ifndef BUILD_INTERNAL
#define BUILD_INTERNAL  0
#endif


#define DEFAULT_CHAR_SIZE      256
#define LARGE_CHAR_SIZE        1024
#define SMALL_CHAR_SIZE        32

// max rocm paths
#define MAX_PATHS 50


// Distro/OS type
typedef enum _DISTRO_TYPE
{
    eDISTRO_TYPE_DEB,       // Deb-based
    eDISTRO_TYPE_EL,        // RHEL-based
    eDISTRO_TYPE_SLE,       // Suse-based
}DISTRO_TYPE;

// ROCm/Driver install types
typedef enum _INTSTALL_TYPE
{
    eINSTALL_NONE,           // Not install
    eINSTALL_PACKAGE,        // Package-based install
    eINSTALL_RUNFILE,        // Runfile-based install
    eINSTALL_NODKMS          // dkms not installed
}INTSTALL_TYPE;



#endif // _INSTALL_TYPES_H
