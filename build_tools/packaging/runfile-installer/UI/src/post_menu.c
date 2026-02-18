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

// verbose help menu variables
// Spaces added/deleted from HelpOps and HelpDesc to ensure whole words aren't
// cut off between lines when displaying help menu.
char *postMenuHelpOps[] = {
    "GPU Access Permissions",
    "    Current User",
    "    All Users"
    SKIPPABLE_MENU_ITEM,
    SKIPPABLE_MENU_ITEM,
    "Post ROCm Install",
    (char*)NULL,
};

char *postMenuHelpDesc[] = {
    " ",
    "Add the current user to the video,render group         for access to GPU resources.",
    "Grant GPU access to all users on the system via        udev rules.",
    "Note, this is an option for system admins.",
    "Execute all ROCm install scripts setting up ROCm       symbolic links, etc.",
    (char*)NULL,
};

MENU_PROP postMenuHelpProps = {
    .pMenuTitle = "Post-Install Configuration Help",
    .pMenuControlMsg = DEFAULT_VERBOSE_HELP_CONTROL_MSG,
    .numLines = 0,
    .numCols = MAX_MENU_ITEM_COLS,
    .starty = POST_MENU_ITEM_START_Y,
    .startx = POST_MENU_ITEM_START_X, 
    .numItems = 0
};

ITEMLIST_PARAMS postMenuHelpItems = {
    .numItems           = 0,
    .pItemListTitle     = "Post-Install Settings Description:",
    .pItemListChoices   = 0,
    .pItemListDesp      = 0
};


void process_post_menu();

// sub-menus
void create_post_help_menu_window();

// menu draw
void post_menu_draw();

MENU_DATA menuPost = {0};


/**************** Post-install MENU **********************************************************************************/

void create_post_menu_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig)
{
    // Create the post install options menu
    create_menu(&menuPost, pMenuWindow, &postMenuProps, &postMenuItems, pConfig);

    // create verbose help menu
    menuPost.pHelpMenu = calloc(1, sizeof(MENU_DATA));
    if (menuPost.pHelpMenu)
    {
        create_post_help_menu_window();
    }

    // Set pointer to draw menu function when window is resized
    menuPost.drawMenuFunc = post_menu_draw;

    // Set user pointers for 'ENTER' events
    set_menu_userptr(menuPost.pMenu, process_post_menu);

    // set items to non-selectable
    set_menu_grey(menuPost.pMenu, COLOR_PAIR(5));
    menu_set_item_select(&menuPost, menuPost.itemList[0].numItems - 4, false);    // space before help
}

void destroy_post_menu_window()
{
    destroy_help_menu(menuPost.pHelpMenu);
    destroy_menu(&menuPost);
}

void post_menu_draw()
{
    WINDOW *pWin = menuPost.pMenuWindow;
    POST_MENU_CONFIG *pPostConfig = &(menuPost.pConfig)->post_config;

    wattron(pWin, COLOR_PAIR(3) | A_BOLD);
    mvwprintw(pWin, 5, 4, "%s", "Set GPU access permissions");
    wattroff(pWin, COLOR_PAIR(3) | A_BOLD);

    menu_draw(&menuPost);

    menu_info_draw_bool(&menuPost, POST_MENU_ITEM_CUR_USER_ROW, POST_MENU_FORM_COL, pPostConfig->current_user_grp);
    menu_info_draw_bool(&menuPost, POST_MENU_ITEM_ALL_USER_ROW, POST_MENU_FORM_COL, pPostConfig->all_user_grp);
    menu_info_draw_bool(&menuPost, POST_MENU_ITEM_POST_ROCM_ROW, POST_MENU_FORM_COL, pPostConfig->rocm_post);
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
    POST_MENU_CONFIG *pPostConfig = &(menuPost.pConfig)->post_config;
    
    ITEM *pCurrentItem = current_item(pMenu);

    int index = item_index(pCurrentItem);

    DEBUG_UI_MSG(&menuPost, "post menu: item %d", index);

    bool isSelectable = item_opts(pCurrentItem) == O_SELECTABLE;

    if (isSelectable)
    {
        if (index == POST_MENU_ITEM_CUR_USER_INDEX)
        {
            pPostConfig->current_user_grp = !pPostConfig->current_user_grp;
            menu_info_draw_bool(&menuPost, POST_MENU_ITEM_CUR_USER_ROW, POST_MENU_FORM_COL, pPostConfig->current_user_grp);
        
            if (pPostConfig->current_user_grp)
            {
                // disable udev and set to false
                pPostConfig->all_user_grp = false;
                menu_info_draw_bool(&menuPost, POST_MENU_ITEM_ALL_USER_ROW, POST_MENU_FORM_COL, pPostConfig->all_user_grp);
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
            pPostConfig->all_user_grp = !pPostConfig->all_user_grp;
            menu_info_draw_bool(&menuPost, POST_MENU_ITEM_ALL_USER_ROW, POST_MENU_FORM_COL, pPostConfig->all_user_grp);

            if (pPostConfig->all_user_grp)
            {
                // disable user and set to false
                pPostConfig->current_user_grp = false;
                menu_info_draw_bool(&menuPost, POST_MENU_ITEM_CUR_USER_ROW, POST_MENU_FORM_COL, pPostConfig->current_user_grp);
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
            pPostConfig->rocm_post = !pPostConfig->rocm_post;
            menu_info_draw_bool(&menuPost, POST_MENU_ITEM_POST_ROCM_ROW, POST_MENU_FORM_COL, pPostConfig->rocm_post);
        }
    }

    post_menu_draw(&menuPost);
}

void create_post_help_menu_window()
{
    MENU_DATA *pMenuData = menuPost.pHelpMenu;
    WINDOW *pMenuWindow = menuPost.pMenuWindow;
    
    // Create menu window w/ border and title
    create_menu(pMenuData, pMenuWindow, &postMenuHelpProps, &postMenuHelpItems, NULL);

    menu_opts_off(pMenuData->pMenu, O_SHOWDESC);

    // create form that displays verbose help menu
    create_help_form(pMenuData, pMenuWindow, POST_HELP_MENU_DESC_STARTX, POST_HELP_MENU_DESC_STARTY, HELP_MENU_DESC_WIDTH, HELP_MENU_OP_STARTX, HELP_MENU_OP_WIDTH, postMenuHelpOps, postMenuHelpDesc); 
}
