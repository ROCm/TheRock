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
#include "menu_data.h"
#define HELP_MENU_DESC_STARTX   28
#define HELP_MENU_DESC_STARTY   5
#define HELP_MENU_DESC_WIDTH    55

#define HELP_MENU_OP_STARTX     4
#define HELP_MENU_OP_WIDTH      22

#define HELP_MENU_ITEM_START_Y        5   // minimum starting y/row
#define HELP_MENU_ITEM_START_X        3   // minimum starting x/col

void create_help_form(MENU_DATA *pMenuData, WINDOW *pMenuWin, int menuDescStartx, int manuDescStarty, int menuDescWidth, int menuOpStartx, int menuOptWidth, char *helpMenuOps[], char *helpMenuDesc[]);

void do_help_menu(MENU_DATA *pMenuData);
void help_menu_draw(MENU_DATA *pMenuData);
void destroy_help_menu(MENU_DATA *pMenuData);
