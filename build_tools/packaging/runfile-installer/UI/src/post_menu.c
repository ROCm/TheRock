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
#include "post_menu.h"
#include "help_menu.h"


// Post Install Menu Setup
char *postMenuOps[] = {
    "    Add video,render group",
    "    Add udev rule",
    SKIPPABLE_MENU_ITEM,
    "Post ROCm setup",
    SKIPPABLE_MENU_ITEM,
    "<HELP>",
    "<DONE>",
    (char*)NULL,
};

char *postMenuDesc[] = {
    "Add current user to the video,render group for GPU access.",
    "Add all users for GPU access.",
    " ",
    "Apply ROCm post-install settings.",
    " ",
    DEFAULT_VERBOSE_HELP_WINDOW_MSG,
    "Exit to Main Menu",
    (char*)NULL,
};


MENU_PROP postMenuProps = {
    .pMenuTitle = "Post-Install Configuration",
    .pMenuControlMsg = "<DONE> to exit : Enter key to toggle selection",
    .numLines = ARRAY_SIZE(postMenuOps) - 1,
    .numCols = MAX_MENU_ITEM_COLS,
    .starty = POST_MENU_ITEM_START_Y,
    .startx = POST_MENU_ITEM_START_X,
    .numItems = ARRAY_SIZE(postMenuOps)
};

ITEMLIST_PARAMS postMenuItems = {
    .numItems           = (ARRAY_SIZE(postMenuOps)),
    .pItemListTitle     = "Settings:",
    .pItemListChoices   = postMenuOps,
    .pItemListDesp      = postMenuDesc
};

void process_post_menu();

// menu draw
void post_menu_draw();

MENU_DATA menuPost = {0};

// Global config pointers (defined in rocm_ui.c)
extern OFFLINE_INSTALL_CONFIG *g_pConfig;
extern POST_MENU_CONFIG *g_pPostConfig;


/**************** Post-install MENU **********************************************************************************/

void create_post_menu_window(WINDOW *pMenuWindow)
{
    // Create the post install options menu
    create_menu(&menuPost, pMenuWindow, &postMenuProps, &postMenuItems, g_pConfig);

    // Create help menu
    create_help_menu_window(&menuPost, POST_MENU_HELP_TITLE, POST_MENU_HELP_FILE);

    // Set pointer to draw menu function when window is resized
    menuPost.drawMenuFunc = post_menu_draw;

    // Set user pointers for 'ENTER' events
    set_menu_userptr(menuPost.pMenu, process_post_menu);

    // Set default: post ROCm installation is enabled by default (matches rocm-installer.sh)
    g_pPostConfig->rocm_post = true;

    // set items to non-selectable
    set_menu_grey(menuPost.pMenu, BLUE);
    menu_set_item_select(&menuPost, menuPost.itemList[0].numItems - 4, false);    // space before help
}

void destroy_post_menu_window()
{
    destroy_help_menu(&menuPost);
    destroy_menu(&menuPost);
}

void post_menu_draw()
{
    WINDOW *pWin = menuPost.pMenuWindow;

    wattron(pWin, WHITE | A_BOLD);
    mvwprintw(pWin, 5, 4, "%s", "Set GPU access permissions");
    wattroff(pWin, WHITE | A_BOLD);

    menu_draw(&menuPost);

    menu_info_draw_bool(&menuPost, POST_MENU_ITEM_CUR_USER_ROW, POST_MENU_FORM_COL, g_pPostConfig->current_user_grp);
    menu_info_draw_bool(&menuPost, POST_MENU_ITEM_ALL_USER_ROW, POST_MENU_FORM_COL, g_pPostConfig->all_user_grp);
    menu_info_draw_bool(&menuPost, POST_MENU_ITEM_POST_ROCM_ROW, POST_MENU_FORM_COL, g_pPostConfig->rocm_post);
}

void do_post_menu()
{
    MENU *pMenu = menuPost.pMenu;

    wclear(menuPost.pMenuWindow);

    // draw the post install menu contents
    post_menu_draw(&menuPost);

    // post install menu loop
    menu_loop(&menuPost);

    unpost_menu(pMenu);
}

// process "ENTER" key events from the Extra packages main menu
void process_post_menu()
{
    MENU *pMenu = menuPost.pMenu;
    ITEM *pCurrentItem = current_item(pMenu);

    int index = item_index(pCurrentItem);

    DEBUG_UI_MSG(&menuPost, "post menu: item %d", index);

    bool isSelectable = item_opts(pCurrentItem) == O_SELECTABLE;

    if (isSelectable)
    {
        if (index == POST_MENU_ITEM_CUR_USER_INDEX)
        {
            g_pPostConfig->current_user_grp = !g_pPostConfig->current_user_grp;
            menu_info_draw_bool(&menuPost, POST_MENU_ITEM_CUR_USER_ROW, POST_MENU_FORM_COL, g_pPostConfig->current_user_grp);
        
            if (g_pPostConfig->current_user_grp)
            {
                // disable udev and set to false
                g_pPostConfig->all_user_grp = false;
                menu_info_draw_bool(&menuPost, POST_MENU_ITEM_ALL_USER_ROW, POST_MENU_FORM_COL, g_pPostConfig->all_user_grp);
                menu_set_item_select(&menuPost, POST_MENU_ITEM_ALL_USER_INDEX, false);
            }
            else
            {
                // enable udev
                menu_set_item_select(&menuPost, POST_MENU_ITEM_ALL_USER_INDEX, true);
            }
        
        }
        else if (index == POST_MENU_ITEM_ALL_USER_INDEX)
        {
            g_pPostConfig->all_user_grp = !g_pPostConfig->all_user_grp;
            menu_info_draw_bool(&menuPost, POST_MENU_ITEM_ALL_USER_ROW, POST_MENU_FORM_COL, g_pPostConfig->all_user_grp);

            if (g_pPostConfig->all_user_grp)
            {
                // disable user and set to false
                g_pPostConfig->current_user_grp = false;
                menu_info_draw_bool(&menuPost, POST_MENU_ITEM_CUR_USER_ROW, POST_MENU_FORM_COL, g_pPostConfig->current_user_grp);
                menu_set_item_select(&menuPost, POST_MENU_ITEM_CUR_USER_INDEX, false);
            }
            else
            {
                // enable user
                menu_set_item_select(&menuPost, POST_MENU_ITEM_CUR_USER_INDEX, true);
            }
        }
        else if (index == POST_MENU_ITEM_POST_ROCM_INDEX)
        {
            g_pPostConfig->rocm_post = !g_pPostConfig->rocm_post;
            menu_info_draw_bool(&menuPost, POST_MENU_ITEM_POST_ROCM_ROW, POST_MENU_FORM_COL, g_pPostConfig->rocm_post);
        }
    }

    post_menu_draw(&menuPost);
}
