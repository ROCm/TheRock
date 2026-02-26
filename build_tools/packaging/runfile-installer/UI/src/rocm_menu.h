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
#ifndef _ROCM_MENU_H
#define _ROCM_MENU_H

#include "menu_data.h"

// ROCm menu draw loc
#define ROCM_MENU_ITEM_START_Y              5   // minimum starting y/row
#define ROCM_MENU_ITEM_START_X              1   // minimum starting x/col

// ROCM status menu draw loc
#define ROCM_STATUS_MENU_ITEM_START_Y       23              
#define ROCM_STATUS_MENU_ITEM_START_X       1

// ROCm menu indices
#define ROCM_MENU_ITEM_INSTALL_ROCM_INDEX   0
#define ROCM_MENU_ITEM_DEVICE_INDEX         1
#define ROCM_MENU_ITEM_COMPO_INDEX          2
#define ROCM_MENU_ITEM_ROCM_PATH_INDEX      3
#define ROCM_MENU_ITEM_UNINSTALL_ROCM_INDEX 5

// ROCm menu setup
#define ROCM_MENU_ITEM_INSTALL_ROCM_ROW     ROCM_MENU_ITEM_START_Y + ROCM_MENU_ITEM_INSTALL_ROCM_INDEX
#define ROCM_MENU_HELP_TITLE                "ROCm Options Help"
#define ROCM_MENU_HELP_FILE                 "./UI/rocm_menus/rocm_help"
#define ROCM_MENU_ITEM_DEVICE_ROW           ROCM_MENU_ITEM_START_Y + ROCM_MENU_ITEM_DEVICE_INDEX
#define ROCM_MENU_ITEM_DEVICE_COL           29
#define ROCM_MENU_ITEM_COMPO_ROW            ROCM_MENU_ITEM_START_Y + ROCM_MENU_ITEM_COMPO_INDEX
#define ROCM_MENU_ITEM_COMPO_COL            ROCM_MENU_ITEM_DEVICE_COL

// ROCm device menu setup
#define ROCM_MENU_DEVICE_ITEMS_FILE         "./UI/rocm_menus/rocm_device_items"
#define ROCM_MENU_DEVICE_ITEMDECS_FILE      "./UI/rocm_menus/rocm_device_items_desc"
#define ROCM_MENU_DEVICE_HELP_TITLE         "ROCm Device Help"
#define ROCM_MENU_DEVICE_HELP_FILE          "./UI/rocm_menus/rocm_device_help"

// ROCm component menu setup
#define ROCM_MENU_COMPO_ITEMS_FILE          "./UI/rocm_menus/rocm_compo_items"
#define ROCM_MENU_COMPO_ITEMDECS_FILE       "./UI/rocm_menus/rocm_compo_items_desc"
#define ROCM_MENU_COMPO_HELP_TITLE          "ROCm Component Help"
#define ROCM_MENU_COMPO_HELP_FILE           "./UI/rocm_menus/rocm_compo_help"

// Form setup
#define ROCM_MENU_NUM_FORM_FIELDS           1
#define ROCM_MENU_FORM_FIELD_WIDTH          50
#define ROCM_MENU_FORM_FIELD_HEIGHT         1       // one line

#define ROCM_MENU_FORM_ROW                  ROCM_MENU_ITEM_INSTALL_ROCM_ROW + 3 // starting row for rocm menu form
#define ROCM_MENU_FORM_COL                  29                                  // staring column for rocm menu form

#define ROCM_MENU_DEFAULT_INSTALL_PATH      "/opt"

// ROCm status/uninstall
#define ROCM_MENU_ROCM_STATUS_ROW           5
#define ROCM_MENU_ROCM_STATUS_COL           ROCM_MENU_FORM_COL + 6

void create_rocm_menu_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig);
void destroy_rocm_menu_window();
void do_rocm_menu();

// ROCm Device Menu
void create_rocm_menu_device_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig);
void do_rocm_menu_device();
void destroy_rocm_menu_device_window();

// ROCm Component Menu
void create_rocm_menu_compo_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig);
void do_rocm_menu_compo();
void destroy_rocm_menu_compo_window();


#endif // _ROCM_MENU_H

