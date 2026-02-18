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
#include <form.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "help_menu.h"
#include "utils.h"

void create_help_form(MENU_DATA *pMenuData, WINDOW *pMenuWin, int menuDescStartx, int manuDescStarty, int menuDescWidth, int menuOpStartx, int menuOptWidth, char *helpMenuOps[], char *helpMenuDesc[]) 
{
    int rows, cols, prevDescFieldHeight, prevDescFieldFirstRow;
    int numOfOps = get_char_array_size(helpMenuOps);
    int descHeight = calculate_text_height(helpMenuDesc[0], menuDescWidth);

    // items in pMenuData->pFormList.field from index 0 to field_size - 1 are fields for helpMenuDesc
    pMenuData->pFormList.field[0] = new_field(descHeight, menuDescWidth, manuDescStarty, menuDescStartx, 0,0);
    field_opts_off(pMenuData->pFormList.field[0], O_ACTIVE);
    set_field_buffer(pMenuData->pFormList.field[0], 0, helpMenuDesc[0]);

    // items in pMenuData->pFormList.field from index field_size to (2*field_size - 1) are fields for helpMenuOps
    pMenuData->pFormList.field[numOfOps] = new_field(descHeight, menuOptWidth, manuDescStarty, menuOpStartx, 0,0);
    field_opts_off(pMenuData->pFormList.field[numOfOps], O_ACTIVE);
    set_field_buffer(pMenuData->pFormList.field[numOfOps], 0, helpMenuOps[0]);
    
    for(int i = 1; i < numOfOps; ++i) 
    {
        FIELD *prevDescField = pMenuData->pFormList.field[i-1];
        field_info(prevDescField, &prevDescFieldHeight, NULL, &prevDescFieldFirstRow, NULL, NULL, NULL);
        descHeight = calculate_text_height(helpMenuDesc[i], menuDescWidth);
        int frow = (prevDescFieldFirstRow + prevDescFieldHeight) + 1;

        // create field for menu option description
        pMenuData->pFormList.field[i] = new_field(descHeight, menuDescWidth, frow, menuDescStartx, 0, 0);
        set_field_buffer(pMenuData->pFormList.field[i], 0, helpMenuDesc[i]);

        // create field for menu option
        int name_field_height = calculate_text_height(helpMenuOps[i], menuOptWidth);
        pMenuData->pFormList.field[i+numOfOps] = new_field(name_field_height, menuOptWidth, frow,menuOpStartx , 0, 0);
        set_field_buffer(pMenuData->pFormList.field[i+numOfOps], 0, helpMenuOps[i]);
    }

    // set field to static
    for (int i = 0; i < (2*numOfOps); ++i)
    {
        field_opts_off(pMenuData->pFormList.field[i], O_ACTIVE);
    }
    
    // TODO: Might want to remove this line in the future.
    pMenuData->pFormList.field[2*numOfOps] = NULL;    
    pMenuData->pFormList.pForm = new_form(pMenuData->pFormList.field);
    pMenuData->pFormList.numFields = 2*numOfOps;
    
    scale_form(pMenuData->pFormList.pForm, &rows, &cols);
    set_form_win(pMenuData->pFormList.pForm, pMenuWin);

    // cols and rows of sub window must be < cols and rows of its parent window
    // If either of them are > than parent window, then it won't display the form on the screen correctly.
    // In this case, parent window is pMenuWin which is the main menu window whose rows and cols are defined by the variables WIN_NUM_LINES
    // and WIN_WIDTH_COLS respectively.
    set_form_sub(pMenuData->pFormList.pForm, derwin(pMenuWin, rows, cols, 0, 0));
}

void help_menu_draw(MENU_DATA *pMenuData)
{
    
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    MENU *pMenu = pMenuData->pMenu;

    int curItemListIndex = pMenuData->curItemListIndex;

    resize_and_reposition_window_and_subwindow(pMenuData, WIN_NUM_LINES, WIN_WIDTH_COLS);

    print_menu_title(pMenuData, MENU_TITLE_Y, MENU_TITLE_X, WIN_WIDTH_COLS, pMenuData->menuTitle, COLOR_PAIR(2));
    print_menu_item_title(pMenuData,  ITEM_TITLE_Y, ITEM_TITLE_X, pMenuData->itemList[curItemListIndex].itemTitle, COLOR_PAIR(9));

    remove_menu_item_selection_description(pMenuData, MENU_SEL_START_Y, MENU_SEL_START_X);

    print_menu_control_msg(pMenuData);

    box(pMenuWindow, 0, 0);

    post_menu(pMenu);
}

void resize_help_menu(MENU_DATA *pMenuData)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    int c = wgetch(pMenuWindow);

    while (c == KEY_RESIZE) // Terminal window resize
    {
        if (should_window_be_resized(pMenuWindow, WIN_NUM_LINES,WIN_WIDTH_COLS))
        {   
            unpost_form(pMenuData->pFormList.pForm);

            reset_window_before_resizing(pMenuData);

            post_form(pMenuData->pFormList.pForm);
            
            help_menu_draw(pMenuData);
        }
        c = wgetch(pMenuWindow);
    }
}

void do_help_menu(MENU_DATA *pMenuData)
{
    MENU *pMenu = pMenuData->pMenu;
    

    post_form(pMenuData->pFormList.pForm);

    // draw menu title, borders, and removes item selection description at the bottom.
    // if you draw this before post_form, then part of window gets covered by the form
    help_menu_draw(pMenuData);

    resize_help_menu(pMenuData);
    
    unpost_menu(pMenu);
    unpost_form(pMenuData->pFormList.pForm);
}

void destroy_help_menu(MENU_DATA *pMenuData)
{
    if (pMenuData)
    {
        destroy_form(pMenuData);
        destroy_menu(pMenuData);
        free(pMenuData);
    }
    
}
