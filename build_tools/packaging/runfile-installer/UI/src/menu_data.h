
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
#ifndef _MENU_DATA_H
#define _MENU_DATA_H

#include "config.h"
#include "debug_ui.h"


// Menu Window Settings
#define WIN_WIDTH_COLS          85
#define WIN_NUM_LINES           30
#define WIN_START_X             4   // relative to stdscr
#define WIN_START_Y             4   // relative to stdscr

// Menu Settings
#define MENU_TITLE_Y            1
#define MENU_TITLE_X            0

// Menu Item Settings
#define MAX_NUM_ITEM_LIST       5   // max number of items list per menu
#define MAX_MENU_ITEM_COLS      80
#define MAX_MENU_ITEMS          20

// Menu Item/list region coordinates
#define ITEM_TITLE_Y            3   // item list title row
#define ITEM_TITLE_X            3   // item list title col

// Item Selection region coordinates
#define MENU_SEL_START_Y        WIN_NUM_LINES - 4
#define MENU_SEL_START_X        1

// Menu Form Settings
#define MAX_NUM_FORM_FIELDS          20   // maximum number of fields in a form for a menu
#define MAX_FORM_FIELD_WIDTH         100  // maximum width for dynamic fields in a form

#define DEFAULT_FORM_CONTROL_MSG                            "<ENTER> to exit : Ctrl+D to delete field."
#define DEFAULT_VERBOSE_HELP_WINDOW_MSG                     "A help menu with more verbose descriptions for each field."
#define DEFAULT_VERBOSE_HELP_CONTROL_MSG                    "Press any key to exit this menu."
#define DEFAULT_SCROLLABLE_VERBOSE_HELP_CONTROL_MSG         "Press the Up/Down arrows keys to scroll : Press any key to exit this menu."
#define SKIPPABLE_MENU_ITEM                                 " "

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))
#define CTRL(x) ((x) & 0x1F)
#define UNUSED(x) (void)(x)

// Scroll window
#define DISPLAY_SCROLL_WINDOW_QUERY_SIZE    16

// Progress window
#define PROGRESS_BAR_WIDTH      (WIN_WIDTH_COLS - 5 - 2)


// Structure defining propertie for a menu
typedef struct _MENU_PROP
{
    char    *pMenuTitle;
    char    *pMenuControlMsg;

    int     numLines;
    int     numCols;

    int     starty;     // starting row
    int     startx;     // starting column

    int     numItems;
}MENU_PROP;

typedef struct _ITEMLIST_PARAMS
{
    int     numItems;
    char    *pItemListTitle;
    char    **pItemListChoices;
    char    **pItemListDesp;
}ITEMLIST_PARAMS;

typedef struct _ITEM_DATA
{
    char    itemTitle[MAX_MENU_ITEM_COLS];
    ITEM    **items;
    int     numItems;

    int     doneItemIndex;
    int     helpItemIndex;
}ITEM_DATA;

typedef struct _FORM_DATA
{
    FORM        *pForm;
    char        formControlMsg[MAX_MENU_ITEM_COLS];
    int         numFields;
    FIELD       *field[MAX_NUM_FORM_FIELDS];
}FORM_DATA;

typedef struct _MENU_DATA
{
    WINDOW      *pMenuWindow;   // window associated with the menu
    
    MENU        *pMenu;         // menu object
    MENU_PROP   *pMenuProps;    // menu items in pMenu
    char        menuTitle[MAX_MENU_ITEM_COLS];
    char        menuControlMsg[MAX_MENU_ITEM_COLS];

    // items
    int         curItemListIndex;
    ITEM_DATA   itemList[MAX_NUM_ITEM_LIST];
    uint32_t    itemSelections;
    int         curItemSelection;

    // forms (one for now)
    FORM_DATA   pFormList;

    bool enableMultiSelection;
    
    // For menus where user can select/deselect items
    bool isMenuItemsSelectable; 

    // global configuration state for the installer
    OFFLINE_INSTALL_CONFIG *pConfig;
    void *pHelpMenu;

    // pointer to menu main draw function
    void *drawMenuFunc;

    WINDOW      *pMenuSubWindow;

    // Only clear the debug/err message area after KEY_UP or KEY_DOWN
    // events in menu_loop iff this is set to true.
    bool clearErrMsgAfterUpOrDownKeyPress; 
}MENU_DATA;


// Menu handling
int create_menu(MENU_DATA *pMenuData,  WINDOW *pMenuWin, MENU_PROP *pProperties, ITEMLIST_PARAMS *pItemListParams, OFFLINE_INSTALL_CONFIG *pConfig);
void destroy_menu(MENU_DATA *pMenuData);

int add_menu_items(MENU_DATA *pMenuData, int itemListIndex, ITEMLIST_PARAMS *pItemListParams);

void menu_loop(MENU_DATA *pMenuData);
void menu_draw(MENU_DATA *pMenuData);
void menu_info_draw_bool(MENU_DATA *pMenuData, int starty, int startx, bool val);
void menu_set_item_select(MENU_DATA *pMenuData, int itemIndex, bool enable_select);

void print_menu_title(MENU_DATA *pMenuData, int starty, int startx, int width, char *string, chtype color);
void print_menu_item_title(MENU_DATA *pMenuData, int starty, int startx, char *string, chtype color);
void print_menu_item_selection(MENU_DATA *pMenuData, int starty, int startx);
void print_menu_item_selection_opt(MENU_DATA *pMenuData, int starty, int startx, const char *description);
void print_version(MENU_DATA *pMenuData);

// Menu message handling
void print_menu_msg(MENU_DATA *pMenuData, chtype color, const char *fmt, ...);
void print_menu_warning_msg(MENU_DATA *pMenuData, const char *fmt, ...);
void print_menu_err_msg(MENU_DATA *pMenuData, const char *fmt, ...);
void clear_menu_msg(MENU_DATA *pMenuData);
void print_menu_dbg_msg(MENU_DATA *pMenuData, const char *fmt, ...);
void print_menu_control_msg(MENU_DATA *pMenuData);
void remove_menu_item_selection_description(MENU_DATA *pMenuData, int starty, int startx);

void skip_menu_item_down_if_skippable(MENU *pMenu);
void skip_menu_item_up_if_skippable(MENU *pMenu);

// Functions to add/delete selection mark that gives user instant feedback
// if a menu item has been selected/deselected
void add_menu_item_selection_mark(MENU_DATA *pMenuData, ITEM *pCurrentItem);
void delete_menu_item_selection_mark(MENU_DATA *pMenuData, ITEM *pCurrentItem);

// Menu Form handling
int create_form(MENU_DATA *pMenuData, WINDOW *pMenuWin, int numFields, int width, int height, int starty, int startx);
void destroy_form(MENU_DATA *pMenuData);

void form_loop(MENU_DATA *pMenuData, bool enableNextField);
void print_form_control_msg(MENU_DATA *pMenuData);

// menu resizing
void reset_window_before_resizing(MENU_DATA *pMenuDatax);
bool should_window_be_resized(WINDOW *pMenuWindow, int y, int x);
void resize_and_reposition_window_and_subwindow(MENU_DATA *pMenuData, int y, int x);

// scroll menus
int display_help_scroll_window(MENU_DATA *pMenuData, char *filename);
int display_scroll_window(char *windowTitle, char *listTitle, char *filename, int *pNumLines);

// progress window/bar
int wait_with_progress_bar(pid_t pid, int time, int show);

#endif // _MENU_DATA_H

