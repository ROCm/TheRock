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
#include <string.h>
#include <stdlib.h>
#include "help_menu.h"
#include "utils.h"

int create_help_menu_window(MENU_DATA *pMenuData, char *helpTile, char *helpFile)
{
    MENU_DATA *pHelpMenuData;
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    
    pMenuData->pHelpMenu = calloc(1, sizeof(MENU_DATA));
    if (NULL ==  pMenuData->pHelpMenu)
    {
        return -1;
    }

    pHelpMenuData = pMenuData->pHelpMenu;

    // setup the help menu properties
    strcpy(pHelpMenuData->helpMenuFile, helpFile);

    MENU_PROP helpMenuProps  = {
        .pMenuTitle = helpTile,
        .pMenuControlMsg = DEFAULT_SCROLLABLE_VERBOSE_HELP_CONTROL_MSG,
        .numLines = 0,
        .numCols = MAX_MENU_ITEM_COLS,
        .starty = HELP_MENU_ITEM_START_Y,
        .startx = HELP_MENU_ITEM_START_X,
        .numItems = 0
    };

    ITEMLIST_PARAMS helpMenuItems = {
        .numItems           = 0,
        .pItemListTitle     = "",
        .pItemListChoices   = NULL,
        .pItemListDesp      = NULL
    };

    // Create menu window w/ border and title
    create_menu(pHelpMenuData, pMenuWindow, &helpMenuProps, &helpMenuItems, NULL);
    menu_opts_off(pHelpMenuData->pMenu, O_SHOWDESC);

    return 0;
}

void destroy_help_menu(MENU_DATA *pMenuData)
{
    if (pMenuData->pHelpMenu)
    {
        destroy_menu(pMenuData->pHelpMenu);
        free(pMenuData->pHelpMenu);

        pMenuData->pHelpMenu = NULL;
    }
}

void help_draw(MENU_DATA *pHelpMenuData)
{
    char *pHelpFile = pHelpMenuData->helpMenuFile;

    // clear and draw the base help menu window
    wclear(pHelpMenuData->pMenuWindow);
    menu_draw(pHelpMenuData);
    
    // draw/scroll help content
    int ret = display_help_scroll_window(pHelpMenuData, pHelpFile);
    if (ret == -1)
    {
        wgetch(pHelpMenuData->pMenuWindow);
    }

    wclear(pHelpMenuData->pMenuWindow);
    menu_draw(pHelpMenuData);
}

void do_help_menu(MENU_DATA *pMenuData)
{
    MENU *pMenu = pMenuData->pMenu;
    MENU_DATA *pHelpMenuData = pMenuData->pHelpMenu;

    // unpost the menu and clear the window
    unpost_menu(pMenu);

    // draw the help menu for the provided source menu
    help_draw(pHelpMenuData);

    unpost_menu(pMenu);
}
