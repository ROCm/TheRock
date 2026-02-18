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
#ifndef _DRIVER_MENU_H
#define _DRIVER_MENU_H

#include "menu_data.h"

#define DRIVER_MENU_ITEM_START_Y    5   // minimum starting y/row
#define DRIVER_MENU_ITEM_START_X    1   // minimum starting x/col

#define DRIVER_MENU_FORM_ROW        DRIVER_MENU_ITEM_START_Y  // starting row for driver menu form
#define DRIVER_MENU_FORM_COL        29                        // staring column for driver menu form

#define DRIVER_HELP_MENU_DESC_STARTX    22
#define DRIVER_HELP_MENU_DESC_STARTY    5

// menu item indicies
#define DRIVER_MENU_ITEM_INSTALL_DRIVER_INDEX   0
#define DRIVER_MENU_ITEM_START_DRIVER_INDEX     1
#define DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX 3

// menu item rows
#define DRIVER_MENU_DRIVER_STATUS_ROW           5
#define DRIVER_MENU_DRIVER_STATUS_COL           DRIVER_MENU_FORM_COL + 6
#define DRIVER_MENU_DRIVER_STATUS_INFO_ROW      DRIVER_MENU_FORM_ROW + DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX
#define DRIVER_MENU_DRIVER_STATUS_INFO_COL      DRIVER_MENU_FORM_COL

#define DRIVER_MENU_ITEM_INSTALL_DRIVER_ROW     DRIVER_MENU_ITEM_START_Y
#define DRIVER_MENU_ITEM_START_DRIVER_ROW       DRIVER_MENU_ITEM_START_Y + 1

// DMKS defines

#define DRIVER_DKMS_PATH    "/var/lib/dkms/amdgpu/"


void create_driver_menu_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig);
void destroy_driver_menu_window();
void do_driver_menu();


#endif // _DRIVER_MENU_H

