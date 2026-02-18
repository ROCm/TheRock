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
#include "pre_menu.h"
#include "help_menu.h"
#include "utils.h"


// Pre Install Menu Setup
char *preMenuOps[] = {
    "ROCm   [ ]",
    "Driver [ ]",
    SKIPPABLE_MENU_ITEM,
    "    Display  Dependencies",
    "    Validate Dependencies",
    "    Install  Dependencies",
    SKIPPABLE_MENU_ITEM,
    "<HELP>",
    "<DONE>",
    (char*)NULL,
};

char *preMenuDesc[] = {
    "Select for ROCm dependencies.",
    "Select for amdgpu dependencies.",
    " ",
    "Display a list of all required dependencies.",
    "Validate required dependencies.",
    "Install required dependencies.",
    " ",
    DEFAULT_VERBOSE_HELP_WINDOW_MSG,
    "Exit to Main Menu",
    (char*)NULL,
};


MENU_PROP preMenuProps = {
    .pMenuTitle = "Pre-Install Configuration",
    .pMenuControlMsg = "<DONE> to exit : Enter key to select or toggle",
    .numLines = ARRAY_SIZE(preMenuOps) - 1,
    .numCols = MAX_MENU_ITEM_COLS,
    .starty = PRE_MENU_ITEM_START_Y,
    .startx = PRE_MENU_ITEM_START_X, 
    .numItems = ARRAY_SIZE(preMenuOps)
};

ITEMLIST_PARAMS preMenuItems = {
    .numItems           = (ARRAY_SIZE(preMenuOps)),
    .pItemListTitle     = "Settings:",
    .pItemListChoices   = preMenuOps,
    .pItemListDesp      = preMenuDesc
};

// verbose help menu variables
// Spaces added/deleted from HelpOps and HelpDesc to ensure whole words aren't
// cut off between lines when displaying help menu.
char *preMenuHelpOps[] = {
    "ROCm",
    "Driver",
    " ",
    "Display  Dependencies",
    "Validate Dependencies",
    "Install Dependencies",
    (char*)NULL,
};

char *preMenuHelpDesc[] = {
    "Enabled/Disable ROCm required dependencies             for display/validate/install.",
    "Enabled/Disable amdgpu Driver required                 dependencies for display/validate/install.",
    " ",
    "Display a list of all 1st-level dependencies           required for ROCm/amdgpu functionality.",
    "Validate the currently installed 1st-level             dependencies for ROCm/amdgpu on the system.",
    "Install 1st-level dependencies required                for ROCm/amdgpu on the system.",
    (char*)NULL,
};

MENU_PROP preMenuHelpProps = {
    .pMenuTitle = "Pre-Install Configuration Help",
    .pMenuControlMsg = DEFAULT_VERBOSE_HELP_CONTROL_MSG,
    .numLines = 0,
    .numCols = MAX_MENU_ITEM_COLS,
    .starty = PRE_MENU_ITEM_START_Y,
    .startx = PRE_MENU_ITEM_START_X, 
    .numItems = 0
};

ITEMLIST_PARAMS preMenuHelpItems = {
    .numItems           = 0,
    .pItemListTitle     = "Pre-install Settings Description:",
    .pItemListChoices   = 0,
    .pItemListDesp      = 0
};


void process_pre_menu();
void process_item();

// sub-menus
void create_pre_help_menu_window();

// menu draw
void pre_menu_toggle_grey_items(bool enable);
void pre_menu_draw();
void draw_deps_selections();

MENU_DATA menuPre = {0};


/**************** Pre-install MENU **********************************************************************************/

void create_pre_menu_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig)
{
    // Create the pre install options menu
    create_menu(&menuPre, pMenuWindow, &preMenuProps, &preMenuItems, pConfig);

    // create verbose help menu
    menuPre.pHelpMenu = calloc(1, sizeof(MENU_DATA));
    if (menuPre.pHelpMenu)
    {
        create_pre_help_menu_window();
    }

    // Set pointer to draw menu function when window is resized
    menuPre.drawMenuFunc = pre_menu_draw;

    // Set user pointers for "item" event
    for (int i = 0; i < menuPre.itemList[0].numItems - 1; ++i) 
    {
        set_item_userptr(menuPre.itemList[0].items[i], process_item);
    }

    // Set user pointers for 'ENTER' events
    set_menu_userptr(menuPre.pMenu, process_pre_menu);

    // set items to non-selectable
    set_menu_grey(menuPre.pMenu, COLOR_PAIR(5));
    menu_set_item_select(&menuPre, menuPre.itemList[0].numItems - 4, false);    // space before help
    pre_menu_toggle_grey_items(false);
}

void destroy_pre_menu_window()
{
    destroy_help_menu(menuPre.pHelpMenu);
    destroy_menu(&menuPre);
}

void pre_menu_toggle_grey_items(bool enable)
{
    if (enable)
    {
        // enable all deps fields
        menu_set_item_select(&menuPre, PRE_MENU_ITEM_DEPS_LIST_INDEX, true);
        menu_set_item_select(&menuPre, PRE_MENU_ITEM_DEPS_VALIDATE_INDEX, true);
        menu_set_item_select(&menuPre, PRE_MENU_ITEM_DEPS_INSTALL_INDEX, true);
    }
    else
    {
        // disable all deps fields
        menu_set_item_select(&menuPre, PRE_MENU_ITEM_DEPS_LIST_INDEX, false);
        menu_set_item_select(&menuPre, PRE_MENU_ITEM_DEPS_VALIDATE_INDEX, false);
        menu_set_item_select(&menuPre, PRE_MENU_ITEM_DEPS_INSTALL_INDEX, false);
    }

    // clear the msg on toggle
    clear_menu_msg(&menuPre);
}

void draw_deps_selections()
{
    PRE_MENU_CONFIG *pPreConfig = &(menuPre.pConfig)->pre_config;
    WINDOW *pWin = menuPre.pMenuWindow;

    if (pPreConfig->rocm_deps)
    {
        mvwprintw(pWin, PRE_MENU_ITEM_START_Y+0, PRE_MENU_ITEM_START_X+11, "%s", "X");
    }
    else
    {
        mvwprintw(pWin, PRE_MENU_ITEM_START_Y+0, PRE_MENU_ITEM_START_X+11, "%s", " ");
    }

    if (pPreConfig->driver_deps)
    {
        mvwprintw(pWin, PRE_MENU_ITEM_START_Y+1, PRE_MENU_ITEM_START_X+11, "%s", "X");
    }
    else
    {
        mvwprintw(pWin, PRE_MENU_ITEM_START_Y+1, PRE_MENU_ITEM_START_X+11, "%s", " ");
    }
}

void pre_menu_draw()
{
    WINDOW *pWin = menuPre.pMenuWindow;

    char depsPath[LARGE_CHAR_SIZE];
    char cwd[512];
    
    getcwd(cwd, sizeof(cwd));
    sprintf(depsPath, "%s/%s", cwd, DEPS_OUT_FILE);

    wattron(pWin, COLOR_PAIR(3) | A_BOLD);
    mvwprintw(pWin, 5, 3, "%s", "Dependencies");
    wattroff(pWin, COLOR_PAIR(3) | A_BOLD);

    wattron(pWin, A_BOLD);
    
    if(access(depsPath, F_OK) != -1)
    {
        mvwprintw(pWin, DEP_FILE_Y, DEP_FILE_X, "File: %s", depsPath);
    }
    else
    {
        mvwprintw(pWin, DEP_FILE_Y, DEP_FILE_X, "File: ");
    }
    
    wattroff(pWin, A_BOLD);

    menu_draw(&menuPre);
    
    draw_deps_selections();
}

void do_pre_menu()
{
    MENU *pMenu = menuPre.pMenu;

    // clear the content of the window
    wclear(menuPre.pMenuWindow);

    // draw the pre install menu contents
    pre_menu_draw();

    // pre install menu loop
    menu_loop(&menuPre);

    unpost_menu(pMenu);
}

void draw_logs_path(WINDOW *pWin)
{
    char cwd[512];
    getcwd(cwd, sizeof(cwd));
    
    mvwprintw(pWin, DEP_FILE_Y, DEP_FILE_X, "Install logs:   ");
    wattron(pWin, A_BOLD);
    mvwprintw(pWin, DEP_FILE_Y+1, DEP_FILE_X, "%s/%s", cwd, LOG_FILE_DIR);
    wattroff(pWin, A_BOLD);
}

void draw_window_title(WINDOW *pWin, char *pTitle)
{
    float temp = (WIN_WIDTH_COLS - strlen(pTitle))/ 2;

    wclear(pWin);

    box(pWin, 0, 0);
    
    wattron(pWin, COLOR_PAIR(2) | A_BOLD);
    mvwprintw(pWin, 1, (int)temp, "%s", pTitle);
    wattroff(pWin, COLOR_PAIR(2) | A_BOLD);
    mvwhline(pWin, 2, 2, ACS_HLINE, WIN_WIDTH_COLS - 4);

    wrefresh(pWin);
}

int execute_cmd_with_progress(const char *script, const char *arg1, const char *arg2, const char *arg3)
{
    int height = 3; 
    int width = PROGRESS_BAR_WIDTH + 5;
    int start_y = WIN_NUM_LINES;
    int start_x = WIN_START_X + 1;
    
    int status;
    int fd = -1;
    
    WINDOW *progress_win = newwin(height, width, start_y, start_x);
    wrefresh(progress_win);

    pid_t pid = fork();
    if (pid == 0) 
    {
        // Child
        fd = open("/dev/null", O_WRONLY);
        if (fd == -1)
        {
            exit(1);
        }
        
        dup2(fd, 1);

        execl("/bin/bash", "bash", script, arg1, arg2, arg3, NULL);
        
        exit(1); // exit if execl fails
    } 
    else if (pid > 0) 
    {
        // Parent
        status = wait_with_progress_bar(pid, 5000, 0);
    } 
    else 
    {
        // Fork failed
        endwin();
        perror("fork");

        exit(1);
    }

    // close any open file descriptors
    if (fd >= 0)
    {
        close(fd);
    }

    delwin(progress_win);

    return (WEXITSTATUS(status));
}

void wait_for_user_input(WINDOW *pWin, int y, int x, char *output)
{
    wattron(pWin, COLOR_PAIR(4) | A_BOLD);
    mvwprintw(pWin, y, x, "%s", output);
    wattroff(pWin, COLOR_PAIR(4) | A_BOLD);

    // clear the process bar
    wmove(pWin, 28, 1);
    wclrtoeol(pWin);

    box(pWin, 0, 0);

    wrefresh(pWin);

    int done = 0;
    int c;

    // wait for keyboard input
    while( done == 0 )
    {
        c = wgetch(pWin);
        switch (c) 
        {
            default:
                done = 1;
                break;
        }
    };
}

void process_item()
{
    //DEBUG_UI_MSG(&menuPre, "item userptr: item index %d, i = %p", item_index(current_item(pMenuData->pMenu)), pMenuData);
    draw_deps_selections();
}

// process "ENTER" key events from the Extra packages main menu
void process_pre_menu()
{
    MENU *pMenu = menuPre.pMenu;
    WINDOW *pWin = menuPre.pMenuWindow;

    PRE_MENU_CONFIG *pPreConfig = &(menuPre.pConfig)->pre_config;
    
    ITEM *pCurrentItem = current_item(pMenu);
    int index = item_index(pCurrentItem);

    int numDeps = 0;
    char depsMsg[DEFAULT_CHAR_SIZE];

    char args[LARGE_CHAR_SIZE];
    clear_str(args);

    char components[SMALL_CHAR_SIZE];
    clear_str(components);

    DEBUG_UI_MSG(&menuPre, "pre menu: item %d", index);

    bool isSelectable = item_opts(pCurrentItem) == O_SELECTABLE;

    if (isSelectable)
    {
        if (index == PRE_MENU_ITEM_DEPS_ROCM_INDEX)
        {
            pPreConfig->rocm_deps = !pPreConfig->rocm_deps;
            draw_deps_selections();

            pre_menu_toggle_grey_items( (pPreConfig->rocm_deps | pPreConfig->driver_deps) );
        }
        else if (index == PRE_MENU_ITEM_DEPS_DRIVER_INDEX)
        {
            pPreConfig->driver_deps = !pPreConfig->driver_deps;
            draw_deps_selections();

            pre_menu_toggle_grey_items( (pPreConfig->rocm_deps | pPreConfig->driver_deps) );
        }
        else if (index == PRE_MENU_ITEM_DEPS_LIST_INDEX) // list
        {
            char *pTitle;
            
            // set the components to list
            if (pPreConfig->rocm_deps)
            {
                strncat(components, "rocm ", SMALL_CHAR_SIZE - strlen(components) - 1);
                pTitle = "ROCm Dependencies";
            }
            if (pPreConfig->driver_deps) 
            {
                strncat(components, "amdgpu", SMALL_CHAR_SIZE - strlen(components) - 1);
                pTitle = "amdgpu driver Dependencies";
            }

            if (pPreConfig->rocm_deps && pPreConfig->driver_deps)
            {
                pTitle = "ROCm and amdgpu driver Dependencies";
            }

            sprintf(args, "deps=list %s", components);
            
            // run the dependency list command
            if (execute_cmd("./rocm-installer.sh", args, NULL) == 0)
            {
                display_scroll_window(pTitle, "Required:", DEPS_OUT_FILE, &numDeps);
            }

            sprintf(depsMsg, "%d Dependencies required. %s written.", numDeps, DEPS_OUT_FILE);
            print_menu_msg(&menuPre, COLOR_PAIR(3), depsMsg);
        }
        else if (index == PRE_MENU_ITEM_DEPS_VALIDATE_INDEX) // validate
        {
            unpost_menu(pMenu);

            draw_window_title(pWin, "Validate Dependencies");

            char arg1[SMALL_CHAR_SIZE];
            char arg2[SMALL_CHAR_SIZE];
            clear_str(arg1);
            clear_str(arg2);

            char *pArg1 = NULL;
            char *pArg2 = NULL;

            // set the components to validate
            if (pPreConfig->rocm_deps) 
            {
                strncat(arg1, "rocm", SMALL_CHAR_SIZE - strlen(arg1) - 1);
                pArg1 = arg1;
            }

            if (pPreConfig->driver_deps) 
            {
                if (strlen(arg1) > 0)
                {
                    strncat(arg2, "amdgpu", SMALL_CHAR_SIZE - strlen(arg2) - 1);
                    pArg2 = arg2;
                }
                else
                {
                    strncat(arg1, "amdgpu", SMALL_CHAR_SIZE - strlen(arg1) - 1);
                    pArg1 = arg1;
                }
            }

            // run the dependency validate command
            if (execute_cmd_with_progress("./rocm-installer.sh", "deps=validate", pArg1, pArg2) == 0)
            {
                if (display_scroll_window("Validate Dependencies", "Missing:", DEPS_OUT_FILE, &numDeps) != 0)
                {
                    wait_for_user_input(pWin, 3, 1, "All dependencies installed.");
                    print_menu_msg(&menuPre, COLOR_PAIR(4), "All dependencies installed.");
                }
                else
                {
                    sprintf(depsMsg, "%d Dependencies missing. %s written.", numDeps, DEPS_OUT_FILE);
                    print_menu_warning_msg(&menuPre, depsMsg);
                }
            }
            else
            {
                print_menu_err_msg(&menuPre, "Failed to validate dependencies.");
            }
        }
        else if (index == PRE_MENU_ITEM_DEPS_INSTALL_INDEX) // Install
        {
            // set the components to install
            if (pPreConfig->rocm_deps) strncat(components, "rocm ", SMALL_CHAR_SIZE - strlen(components) - 1);
            if (pPreConfig->driver_deps) strncat(components, "amdgpu", SMALL_CHAR_SIZE - strlen(components) - 1);
            sprintf(args, "deps=install-only %s", components);
	        
            // run the dependency install command
            if (execute_cmd("./rocm-installer.sh", args, pWin) == 0)
            {
                print_menu_msg(&menuPre, COLOR_PAIR(4), "All dependencies installed.");
            }
            else
            {
                print_menu_err_msg(&menuPre, "Failed to install dependencies.");
                draw_logs_path(pWin);
            }

            wrefresh(pWin);
        }
        else
        {
            DEBUG_UI_MSG(&menuPre, "Unknown item index");
        }
    }

    pre_menu_draw();
}

void create_pre_help_menu_window()
{
    MENU_DATA *pMenuData = menuPre.pHelpMenu;
    WINDOW *pMenuWindow = menuPre.pMenuWindow;

    // Create menu window w/ border and title
    create_menu(pMenuData, pMenuWindow, &preMenuHelpProps, &preMenuHelpItems, NULL);

    menu_opts_off(pMenuData->pMenu, O_SHOWDESC);

    // create form that displays verbose help menu
    create_help_form(pMenuData, pMenuWindow, PRE_HELP_MENU_DESC_STARTX, PRE_HELP_MENU_DESC_STARTY, HELP_MENU_DESC_WIDTH, HELP_MENU_OP_STARTX, HELP_MENU_OP_WIDTH, preMenuHelpOps, preMenuHelpDesc); 
}

