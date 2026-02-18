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
#ifndef _PRE_MENU_H
#define _PRE_MENU_H

#include "menu_data.h"

#define PRE_MENU_ITEM_START_Y    9   // minimum starting y/row
#define PRE_MENU_ITEM_START_X    2   // minimum starting x/col

#define PRE_HELP_MENU_DESC_STARTX    28
#define PRE_HELP_MENU_DESC_STARTY    5

// menu item indices

// config menu indices
#define PRE_MENU_ITEM_DEPS_ROCM_INDEX        0
#define PRE_MENU_ITEM_DEPS_DRIVER_INDEX      1
#define PRE_MENU_ITEM_DEPS_LIST_INDEX        3
#define PRE_MENU_ITEM_DEPS_VALIDATE_INDEX    4
#define PRE_MENU_ITEM_DEPS_INSTALL_INDEX     5

// Dependency settings
#define DEPS_OUT_FILE       "deps_list.txt"
#define LOG_FILE_DIR        "logs"
#define DEP_FILE_Y          7
#define DEP_FILE_X          5


void create_pre_menu_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig);
void destroy_pre_menu_window();
void do_pre_menu();


#endif // _PRE_MENU_H

