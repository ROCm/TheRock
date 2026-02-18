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
#ifndef _DEBUG_UI_H
#define _DEBUG_UI_H

#define ENABLE_MENU_DEBUG   0                   // enable/disable debug messages


// Debug/Error region coordinates

#define DEBUG_ERR_START_Y   WIN_NUM_LINES - 2   // debug message row
#define DEBUG_ERR_START_X   1                   // debug message col

// Warning message region coordinates
#define WARN_ERR_START_Y   WIN_NUM_LINES - 3    // warning message row
#define WARN_ERR_START_X   1                    // warning message col


#if ENABLE_MENU_DEBUG
#define DEBUG_UI_MSG(pMenuData, ...)            \
{                                               \
    print_menu_dbg_msg(pMenuData, __VA_ARGS__); \
}
#else
#define DEBUG_UI_MSG(pMenuData, ...)
#endif // ENABLE_MENU_DEBUG


#endif // _DEBUG_UI_H

