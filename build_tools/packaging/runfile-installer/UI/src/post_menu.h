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
#ifndef _POST_MENU_H
#define _POST_MENU_H

#include "menu_data.h"

#define POST_MENU_ITEM_START_Y    6   // minimum starting y/row
#define POST_MENU_ITEM_START_X    1   // minimum starting x/col

#define POST_MENU_FORM_ROW        9       // starting row for post menu form
#define POST_MENU_FORM_COL        32      // staring column for post menu form

#define POST_HELP_MENU_DESC_STARTX  27
#define POST_HELP_MENU_DESC_STARTY  5

// menu item indicies
#define POST_MENU_ITEM_CUR_USER_INDEX       0
#define POST_MENU_ITEM_ALL_USER_INDEX       1
#define POST_MENU_ITEM_POST_ROCM_INDEX      3

// menu item rows
#define POST_MENU_ITEM_CUR_USER_ROW         6
#define POST_MENU_ITEM_ALL_USER_ROW         7
#define POST_MENU_ITEM_POST_ROCM_ROW        9


void create_post_menu_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig);
void destroy_post_menu_window();
void do_post_menu();


#endif // _POST_MENU_H

