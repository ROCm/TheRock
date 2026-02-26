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
#include "rocm_menu.h"
#include "help_menu.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>


/***************** ROCm Main Menu Setup *****************/
char *rocmMenuMainOp[] = {
    "Install ROCm",
    "   ROCm Device",
    "   ROCm Components",
    "   ROCm Install Path",
    SKIPPABLE_MENU_ITEM,
    "Uninstall ROCm",
    SKIPPABLE_MENU_ITEM,
    "<HELP>",
    "<DONE>",
    (char*)NULL,
};

char *rocmMenuMainDesc[] = {
    "Enable/Disable ROCm install.  Enabling will search for ROCm.",
    "Select the ROCm device for installation.",
    "Select the ROCm components for installation.",
    "Set ROCm Install Target Directory.",
    SKIPPABLE_MENU_ITEM,
    "Uninstall runfile ROCm.",
    SKIPPABLE_MENU_ITEM,
    DEFAULT_VERBOSE_HELP_WINDOW_MSG,
    "Exit to Main Menu",
    (char*)NULL,
};

MENU_PROP rocmMenuMainProps  = {
    .pMenuTitle = "ROCm Options",
    .pMenuControlMsg = "<DONE> to exit",
    .numLines = ARRAY_SIZE(rocmMenuMainOp) - 1,
    .numCols = MAX_MENU_ITEM_COLS, 
    .starty = ROCM_MENU_ITEM_START_Y, 
    .startx = ROCM_MENU_ITEM_START_X, 
    .numItems = ARRAY_SIZE(rocmMenuMainOp)
};

ITEMLIST_PARAMS rocmMenuMainItems = {
    .numItems           = (ARRAY_SIZE(rocmMenuMainOp)),
    .pItemListTitle     = "Settings:",
    .pItemListChoices   = rocmMenuMainOp,
    .pItemListDesp      = rocmMenuMainDesc
};

// Forms
void process_rocm_menu_form(MENU_DATA *pMenuData);

// Menu draw/config
void rocm_menu_toggle_grey_items(bool enable);
void rocm_status_draw();
void rocm_menu_draw();
void rocm_menu_submenu_draw(MENU_DATA *pMenuData);

// ROCm Menu
void process_rocm_menu();

// ROCm Device Menu
void create_rocm_menu_device_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig);
void destroy_rocm_menu_device_window();
void do_rocm_menu_device();
void process_rocm_device_menu();
void update_rocm_device_name();
void reset_rocm_device_name();

// ROCm Component Menu
void create_rocm_menu_compo_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig);
void destroy_rocm_menu_compo_window();
void do_rocm_menu_compo();
void process_rocm_compo_menu();
void set_rocm_components_name(int index);
void clear_rocm_components_name();
void update_rocm_components_name();
void reset_rocm_components_name();

// ROCm Uninstall Menu
void create_rocm_uninstall_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig);
void destroy_rocm_uninstall_menu_window();
void do_rocm_uninstall_menu();
void process_rocm_uninstall_menu();
void process_rocm_uninstall_item();
void update_rocm_uinstall_menu();

// ROCm menus
MENU_DATA menuROCm = {0};
MENU_DATA menuROCmUninstall = {0};
bool gRocmStatusCheck = false;

// ROCm Device Menu
MENU_DATA menuROCmDevice = {0};
MENU_DATA menuHelpROCmDevice = {0};
MENU_PROP rocmMenuDeviceProps = {0};
ITEMLIST_PARAMS rocmMenuDeviceItems = {0};
char rocmMenuDeviceOps[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {0};
char rocmMenuDeviceDesc[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {0};

// ROCm Component Menu
MENU_DATA menuROCmCompo = {0};
MENU_PROP rocmMenuCompoProps = {0};
ITEMLIST_PARAMS rocmMenuCompoItems = {0};
char rocmMenuCompoOps[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {0};
char rocmMenuCompoDesc[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {0};

// ROCM Uninstall Menu
uint8_t rocm_paths_uninstall_state[MAX_PATHS] = {0};
char *rocm_paths_items[MAX_PATHS + 4] = {0};
char *rocm_paths_item_desc[MAX_PATHS + 4] = {0};

MENU_PROP rocmPathsProps = {0};
ITEMLIST_PARAMS rocmPathsItems = {0};

int g_uninstall_rocm_pkg_index = -1;
int g_uninstall_rocm_runfile_index = -1;
int g_uninstall_start_index = 0;
int g_uninstall_end_index = 19;


/**************** ROCm MENU **********************************************************************************/

void create_rocm_menu_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig)
{
    ROCM_MENU_CONFIG *pRocmConfig = &pConfig->rocm_config;

    // Create the ROCm options menu
    create_menu(&menuROCm, pMenuWindow, &rocmMenuMainProps, &rocmMenuMainItems, pConfig);

    // Create help menu
    create_help_menu_window(&menuROCm, ROCM_MENU_HELP_TITLE, ROCM_MENU_HELP_FILE);

    // Create the rocm device menu
    create_rocm_menu_device_window(pMenuWindow, pConfig);

    // Create the rocm component menu
    create_rocm_menu_compo_window(pMenuWindow, pConfig);

    // Set pointer to draw menu function when window is resized
    menuROCm.drawMenuFunc = rocm_menu_draw;

    // Set user pointer for 'ENTER' events
    set_menu_userptr(menuROCm.pMenu, process_rocm_menu);

    // Initialize the menu config settings
    sprintf(pRocmConfig->rocm_install_path, "%s", ROCM_MENU_DEFAULT_INSTALL_PATH);

    // Initialize the rocm config
    pRocmConfig->install_rocm = false;      // disable rocm install by default
    pRocmConfig->is_rocm_path_valid = true; // default path "/" is valid

    // set items to non-selectable
    set_menu_grey(menuROCm.pMenu, BLUE);
    menu_set_item_select(&menuROCm, menuROCm.itemList[0].numItems - 4, false);  // space before done
    rocm_menu_toggle_grey_items(false);
    
    // create a form for user input
    create_form(&menuROCm, pMenuWindow, ROCM_MENU_NUM_FORM_FIELDS, ROCM_MENU_FORM_FIELD_WIDTH, ROCM_MENU_FORM_FIELD_HEIGHT, 
            ROCM_MENU_FORM_ROW, ROCM_MENU_FORM_COL);

    strcpy(menuROCm.pFormList.formControlMsg, DEFAULT_FORM_CONTROL_MSG);

    // Initialize form field names and associated config settings
    set_form_userptr(menuROCm.pFormList.pForm, process_rocm_menu_form);
    set_field_buffer(menuROCm.pFormList.field[0], 0, ROCM_MENU_DEFAULT_INSTALL_PATH);
}

void destroy_rocm_menu_window()
{
    // destroy the sub-menus
    destroy_rocm_menu_device_window();
    destroy_rocm_menu_compo_window();

    destroy_help_menu(&menuROCm);
    destroy_menu(&menuROCmUninstall);
    destroy_menu(&menuROCm);
}

void rocm_menu_toggle_grey_items(bool enable)
{
    if (enable)
    {
        // enable all rocm option fields
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_INSTALL_ROCM_INDEX, true);
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_DEVICE_INDEX, true);
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_COMPO_INDEX, true);
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_ROCM_PATH_INDEX, true);
    }
    else
    {
        // disable all rocm option fields
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_INSTALL_ROCM_INDEX, false);
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_DEVICE_INDEX, false);
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_COMPO_INDEX, false);
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_ROCM_PATH_INDEX, false);
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_UNINSTALL_ROCM_INDEX, false);
    }
}

int find_rocm_with_progress(char *target) 
{
    int height = 3; 
    int width = PROGRESS_BAR_WIDTH + 5;
    int start_y = WIN_NUM_LINES;
    int start_x = WIN_START_X + 1;

    int status;
    int pipefd[2];
    int fd = -1;

    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCm.pConfig)->rocm_config;

    // check for a valid target install path
    if (!pRocmConfig->is_rocm_path_valid)
    {
        return -1;
    }

    // clear the current paths
    memset(pRocmConfig->rocm_paths, '\0', sizeof(pRocmConfig->rocm_paths));
    pRocmConfig->rocm_count = 0;

    if (pipe(pipefd) == -1) 
    {
        perror("pipe");
        return -1;
    }

    WINDOW *progress_win = newwin(height, width, start_y, start_x);
    wrefresh(progress_win);

    pid_t pid = fork();
    if (pid == 0) 
    {
        // Child

        close(pipefd[0]); // Close unused read end
        
        fd = open("/dev/null", O_WRONLY);
        if (fd == -1)
        {
            exit(1);
        }

        dup2(fd, 1);

        // Call the function
        status = find_rocm_installed(target, pRocmConfig->rocm_paths, &(pRocmConfig->rocm_count));

        // Write the result to the pipe
        write(pipefd[1], &pRocmConfig->rocm_count, sizeof(pRocmConfig->rocm_count));
        for (int i = 0; i < pRocmConfig->rocm_count; i++) 
        {
            write(pipefd[1], pRocmConfig->rocm_paths[i], sizeof(pRocmConfig->rocm_paths[i]));
        }

        close(pipefd[1]); // Close write end

        // exit with the function's return status
        exit(status);
    } 
    else if (pid > 0) 
    {
        // Parent
        
        close(pipefd[1]); // Close unused write end

        status = wait_with_progress_bar(pid, 5000, 0);

        // Read the result from the pipe
        read(pipefd[0], &pRocmConfig->rocm_count, sizeof(pRocmConfig->rocm_count));
        for (int i = 0; i < pRocmConfig->rocm_count; i++) 
        {
            read(pipefd[0], pRocmConfig->rocm_paths[i], sizeof(pRocmConfig->rocm_paths[i]));
        }

        close(pipefd[0]); // Close read end
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

    return status;
}

int check_target_for_package_install(char *target, char *rocm_loc)
{
    OFFLINE_INSTALL_CONFIG *pConfig = menuROCm.pConfig;

    int ret = 0;
    char rocm_core_name[LARGE_CHAR_SIZE];
    char rocm_core_ver[SMALL_CHAR_SIZE];

    // Check if the target is / and current rocm installed location is in /opt/rocm
    if ( (strcmp(target, "/") == 0) && (is_loc_opt_rocm(rocm_loc) == 1) )
    {
        // get the rocm-core package name
        if (get_rocm_core_pkg(pConfig->distroType, rocm_core_name, LARGE_CHAR_SIZE) == 0)
        {
            // check if the rocm-core package contains the loc rocm version - if yes = package manger install
            if (get_rocm_version_str_from_path(rocm_loc, rocm_core_ver) == 0)
            {
                char *rocm_chk = strstr(rocm_core_name, rocm_core_ver);
                if (NULL != rocm_chk)
                {
                    ret = 1;
                }
            }
        }
    }

    return ret;
}

void check_rocm_install_status()
{
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCm.pConfig)->rocm_config;

    if (!pRocmConfig->is_rocm_path_valid)
    {
        return;
    }

    gRocmStatusCheck = true;

    // init rocm path state
    pRocmConfig->is_rocm_installed = false;
    pRocmConfig->rocm_install_type = eINSTALL_NONE;
    pRocmConfig->rocm_pkg_path_index = -1;
    pRocmConfig->rocm_runfile_path_index = -1;

    // get the list of rocm install paths at the target path
    if (find_rocm_with_progress(pRocmConfig->rocm_install_path) == 0)
    {
        // check the installer rocm version against the locations found for a collision/conflict
        for (int i = 0; i < pRocmConfig->rocm_count; i++)
        {
            char installer_rocm_ver[LARGE_CHAR_SIZE];
            OFFLINE_INSTALL_CONFIG *pConfig = pRocmConfig->pConfig;
            sprintf(installer_rocm_ver, "rocm-%s", pConfig->rocmVersion);
            
            char *rocm_str = strstr(pRocmConfig->rocm_paths[i], installer_rocm_ver);
            if (rocm_str)
            {
                // rocm installation/s found at target
                pRocmConfig->is_rocm_installed = true;
                
                // installer is installing the same version of rocm for the current found path
                // check if the found path is in /opt/rocm and a package manager install
                if (check_target_for_package_install(pRocmConfig->rocm_install_path, pRocmConfig->rocm_paths[i]) == 1)
                {
                    // current target for install conflicts with package manager install
                    pRocmConfig->rocm_install_type = eINSTALL_PACKAGE;
                    pRocmConfig->rocm_pkg_path_index = i;
                    break;
                }
                else
                {
                    // current target for install conflict but is an runfile install
                    pRocmConfig->rocm_install_type = eINSTALL_RUNFILE;
                    pRocmConfig->rocm_runfile_path_index = i;
                    break;
                }
            }
        }

        // update the uninstall menu with the new set of rocm install paths found
        update_rocm_uinstall_menu();
    }

    // enable/disable uninstall based on if rocm installed and type of install
    if (pRocmConfig->is_rocm_installed)
    {
        // only enable uninstall for package manager installs if count > 1 (mixed installed)
        if (pRocmConfig->rocm_install_type == eINSTALL_PACKAGE)
        {
            if (pRocmConfig->rocm_count == 1)
            {
                menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_UNINSTALL_ROCM_INDEX, false);
            }
        }
        else
        {
            menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_UNINSTALL_ROCM_INDEX, true);
        }
    }
    else
    {
        menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_UNINSTALL_ROCM_INDEX, false);
    }

    // render any updates to the rocm install status
    rocm_status_draw();
}

void rocm_status_draw()
{
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCm.pConfig)->rocm_config;

    if (!pRocmConfig->install_rocm) 
    {
        clear_menu_msg(&menuROCm);
        return;
    }
    
    // check if the rocm target install path is valid - draw msg updates if valid
    if (pRocmConfig->is_rocm_path_valid)
    {
        OFFLINE_INSTALL_CONFIG *pConfig = pRocmConfig->pConfig;
        // check for the ROCm status and draw
        if (pRocmConfig->rocm_install_type == eINSTALL_NONE)
        {
            print_menu_msg(&menuROCm, GREEN, "ROCm %s: Install Path valid.", pConfig->rocmVersion);
        }
        else if (pRocmConfig->rocm_install_type == eINSTALL_PACKAGE)
        {
            print_menu_err_msg(&menuROCm, "ROCm %s package manager install found. Uninstall required.", pConfig->rocmVersion);
        }
        else if (pRocmConfig->rocm_install_type == eINSTALL_RUNFILE)
        {
            print_menu_warning_msg(&menuROCm, "ROCm %s runfile install found.  Uninstall optional.", pConfig->rocmVersion);
        }
        else
        {
            print_menu_err_msg(&menuROCm, "ROCm installation status unknown.");
        }
    }
    else
    {
        print_menu_err_msg(&menuROCm, "ROCm Install Path Invalid");
    }
}

void rocm_menu_draw()
{
    WINDOW *pMenuWindow = menuROCm.pMenuWindow;
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCm.pConfig)->rocm_config;

    char drawName[DEFAULT_CHAR_SIZE];

    menu_info_draw_bool(&menuROCm, ROCM_MENU_ITEM_INSTALL_ROCM_ROW, ROCM_MENU_FORM_COL, pRocmConfig->install_rocm);
    
    field_trim(pRocmConfig->rocm_install_path, drawName, ROCM_MENU_FORM_FIELD_WIDTH);
    mvwprintw(pMenuWindow, ROCM_MENU_FORM_ROW,  ROCM_MENU_FORM_COL, "%s", drawName);

    // draw the rocm device info
    if (pRocmConfig->install_rocm)
    {
        mvwprintw(pMenuWindow, ROCM_MENU_ITEM_DEVICE_ROW, ROCM_MENU_ITEM_DEVICE_COL, "%s", pRocmConfig->rocm_device);
    }
    else
    {
        wmove(pMenuWindow, ROCM_MENU_ITEM_DEVICE_ROW, ROCM_MENU_ITEM_DEVICE_COL);
        wclrtoeol(pMenuWindow);
    }

    // draw the rocm compoment info
    if (pRocmConfig->install_rocm)
    {
        mvwprintw(pMenuWindow, ROCM_MENU_ITEM_COMPO_ROW, ROCM_MENU_ITEM_COMPO_COL, "%s", pRocmConfig->rocm_components);
    }
    else
    {
        wmove(pMenuWindow, ROCM_MENU_ITEM_COMPO_ROW, ROCM_MENU_ITEM_COMPO_COL);
        wclrtoeol(pMenuWindow);
    }

    rocm_status_draw();

    menu_draw(&menuROCm);
}

void rocm_menu_submenu_draw(MENU_DATA *pMenuData)
{
    wclear(pMenuData->pMenuWindow);
    menu_draw(pMenuData);
}

void do_rocm_menu()
{  
    MENU *pMenu = menuROCm.pMenu;

    wclear(menuROCm.pMenuWindow);

    // draw the ROCm menu contents
    rocm_menu_draw(&menuROCm);

    // ROCm menu loop
    menu_loop(&menuROCm);

    unpost_menu(pMenu);
}

// process "ENTER" key events from the ROCm main menu
void process_rocm_menu()
{
    MENU *pMenu = menuROCm.pMenu;
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCm.pConfig)->rocm_config;

    ITEM *pCurrentItem = current_item(pMenu);

    int index = item_index(pCurrentItem);

    DEBUG_UI_MSG(&menuROCm, "ROCM Menu: %d, itemlist %d", index, menuROCm.curItemListIndex);

    if (index == ROCM_MENU_ITEM_INSTALL_ROCM_INDEX)
    {
        // check the rocm status
        if (!gRocmStatusCheck) check_rocm_install_status();

        pRocmConfig->install_rocm = !pRocmConfig->install_rocm;

        rocm_menu_toggle_grey_items(pRocmConfig->install_rocm);
        menu_info_draw_bool(&menuROCm, ROCM_MENU_ITEM_INSTALL_ROCM_ROW, ROCM_MENU_FORM_COL, pRocmConfig->install_rocm);

        // reset any state on rocm install toggle off
        if (!pRocmConfig->install_rocm)
        {
            // reset the rocm install check
            gRocmStatusCheck = false;
            if (pRocmConfig->rocm_install_type == eINSTALL_NONE)
            {
                clear_menu_msg(&menuROCm);
            }

            // reset the rocm device selection
            reset_rocm_device_name();

            // reset the rocm component selection
            reset_rocm_components_name();
        }
    }
    else if (index == ROCM_MENU_ITEM_DEVICE_INDEX)
    {
        // Display ROCm Device menu if ROCm is enabled
        if (pRocmConfig->install_rocm)
        {
            // switch to the rocm device sub-menu
            unpost_menu(pMenu);
            do_rocm_menu_device();
        }
    }
    else if (index == ROCM_MENU_ITEM_COMPO_INDEX)
    {
        // Display ROCm Component menu if ROCm is enabled
        if (pRocmConfig->install_rocm)
        {
            unpost_menu(pMenu);
            do_rocm_menu_compo();
        }
    }
    else if (index == ROCM_MENU_ITEM_ROCM_PATH_INDEX)
    {
        FORM *pForm = menuROCm.pFormList.pForm;
        if (pForm && (pRocmConfig->install_rocm))
        {
            // switch to the form for rocm install target path
            unpost_menu(pMenu);

            void (*ptrFormFnc)(MENU_DATA*);

            ptrFormFnc = form_userptr(pForm);
            if (NULL != ptrFormFnc)
            {
                ptrFormFnc((MENU_DATA*)&menuROCm);
            }
            else
            {
                DEBUG_UI_MSG(&menuROCm, "No user ptr for form");
            }

            check_rocm_install_status();
        }
    }
    else if (index == ROCM_MENU_ITEM_UNINSTALL_ROCM_INDEX)
    {
        // only uninstall if ROCm is installed
        if (pRocmConfig->is_rocm_installed)
        {
            // check for package manager install - if mixed, switch to uninstall menu
            if (pRocmConfig->rocm_install_type == eINSTALL_PACKAGE)
            {
                if (pRocmConfig->rocm_count > 1)
                {
                    clear_menu_msg(&menuROCm);
                    unpost_menu(pMenu);
                    do_rocm_uninstall_menu();
                }
            }
            else
            {
                clear_menu_msg(&menuROCm);
                unpost_menu(pMenu);
                do_rocm_uninstall_menu();
            }
        }

        check_rocm_install_status();
    }
    else
    {
        DEBUG_UI_MSG(&menuROCm, "Unknown item index");
    }

    rocm_menu_draw();
}

// process "ENTER" key event from menu if form userptr set
void process_rocm_menu_form(MENU_DATA *pMenuData)
{
    MENU *pMenu = pMenuData->pMenu;
    FORM *pForm = pMenuData->pFormList.pForm;

    ROCM_MENU_CONFIG *pRocmConfig = &(pMenuData->pConfig)->rocm_config;

    post_form(pForm);
    post_menu(pMenu);

    rocm_menu_draw(pMenuData);

    print_form_control_msg(pMenuData);

    // Switch to form control loop for entering data into given form field
    form_loop(pMenuData, false);

    unpost_form(pForm);
    unpost_menu(pMenu);

    // store the ROCm install target path on exit
    strcpy(pRocmConfig->rocm_install_path, field_buffer(pForm->field[0], 0));

    if (check_path_exists(pRocmConfig->rocm_install_path, MAX_FORM_FIELD_WIDTH) == 0)
    {
        pRocmConfig->is_rocm_path_valid = true;
    }
    else
    {
        pRocmConfig->is_rocm_path_valid = false;
    }

    DEBUG_UI_MSG(pMenuData, "ROCM path =%s", pRocmConfig->rocm_install_path);
}


/**************** ROCM Uninstall MENU ************************************************************************/

void draw_rocm_uninstall_types()
{
    WINDOW *pWin = menuROCmUninstall.pMenuWindow;
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmUninstall.pConfig)->rocm_config;

    int draw_index = 0;
    
    // mark each rocm install by type
    for (int i = 0; i < pRocmConfig->rocm_count; i++)
    {
        if (i >= g_uninstall_start_index && i <= g_uninstall_end_index)
        {
            if (i == g_uninstall_rocm_pkg_index)
            {
                // package manager install
                wattron(pWin, WHITE_ON_RED | A_BOLD);
                mvwprintw(pWin, ROCM_MENU_ITEM_START_Y+draw_index, 3, "P");
                wattroff(pWin, WHITE_ON_RED | A_BOLD);
            }
            else if (i == g_uninstall_rocm_runfile_index)
            {
                // runfile install with conflict - matches installers rocm version
                OFFLINE_INSTALL_CONFIG *pConfig = pRocmConfig->pConfig;
                wattron(pWin, YELLOW | A_BOLD);
                mvwprintw(pWin, ROCM_MENU_ITEM_START_Y+draw_index, 3, "C");
                wattroff(pWin, YELLOW | A_BOLD);

                char drawName[DEFAULT_CHAR_SIZE];
                field_trim(pRocmConfig->rocm_paths[i], drawName, 30);

                print_menu_warning_msg(&menuROCmUninstall, "ROCm %s runfile conflict: %s", pConfig->rocmVersion, drawName);
            }
            else
            {
                // runfile install - different from installer rocm version
                wattron(pWin, GREEN | A_BOLD);
                mvwprintw(pWin, ROCM_MENU_ITEM_START_Y+draw_index, 3, "R");
                wattroff(pWin, GREEN | A_BOLD);
            }

            draw_index++;
        }
    }
}

void rocm_uninstall_menu_draw()
{
    WINDOW *pWin = menuROCmUninstall.pMenuWindow;
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmUninstall.pConfig)->rocm_config;
    
    menu_draw(&menuROCmUninstall);

    // mark each rocm install by type
    draw_rocm_uninstall_types();

    // draw the uninstall legend
    wattron(pWin, WHITE | A_UNDERLINE);
    mvwprintw(pWin, 22, 65, "  Install type   ");
    wattroff(pWin, WHITE | A_UNDERLINE);

    wattron(pWin, WHITE_ON_RED | A_BOLD);
    mvwprintw(pWin, 23, 65, "P");
    wattroff(pWin, WHITE_ON_RED | A_BOLD);
    mvwprintw(pWin, 23, 66, " Package manager");

    wattron(pWin, YELLOW | A_BOLD);
    mvwprintw(pWin, 24, 65, "C");
    wattroff(pWin, YELLOW | A_BOLD);
    mvwprintw(pWin, 24, 66, " Runfile Conflict");

    wattron(pWin, GREEN | A_BOLD);
    mvwprintw(pWin, 25, 65, "R");
    wattroff(pWin, GREEN | A_BOLD);
    mvwprintw(pWin, 25, 66, " Runfile");

    ITEM **items = menu_items(menuROCmUninstall.pMenu);
    if(item_value(items[menuROCmUninstall.curItemSelection]) == TRUE)
    {
        print_menu_msg(&menuROCmUninstall, WHITE, "Uninstall: %s", pRocmConfig->rocm_paths[menuROCmUninstall.curItemSelection]);
    }
}

void do_rocm_uninstall_menu()
{
    rocm_uninstall_menu_draw();
    
    menu_loop(&menuROCmUninstall);

    wclear(menuROCmUninstall.pMenuWindow);

    unpost_menu(menuROCmUninstall.pMenu);
}

void create_rocm_uninstall_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig)
{
    ROCM_MENU_CONFIG *pRocmConfig = &pConfig->rocm_config;
    char uninstall_item_title[LARGE_CHAR_SIZE];
    int i;
    
    // set the path pointers to the found rocm paths
    if (pRocmConfig->rocm_count != 0) 
    {
        // check for "any" package manager install
        for (i = 0; i < pRocmConfig->rocm_count; i++)
        {
            if (check_target_for_package_install(pRocmConfig->rocm_install_path, pRocmConfig->rocm_paths[i]) == 1)
            {
                g_uninstall_rocm_pkg_index = i;
                break;
            }
        }

        // set the uninstall runfile index
        g_uninstall_rocm_runfile_index = pRocmConfig->rocm_runfile_path_index;

        // set the menu item names to the rocm paths found
        for (i = 0; i < pRocmConfig->rocm_count; i++) 
        {
            rocm_paths_items[i] = pRocmConfig->rocm_paths[i];
            rocm_paths_item_desc[i] = pRocmConfig->rocm_paths[i];
        }

        rocm_paths_items[i] = " ";
        rocm_paths_item_desc[i++] = " ";

        rocm_paths_items[i] = "<UNINSTALL>";
        rocm_paths_item_desc[i++] = "Uninstall selected ROCm installation.";

        rocm_paths_items[i] = "<DONE>";
        rocm_paths_item_desc[i++] = " Exit to ROCm Options Menu";

        int numItems = i;

        rocmPathsProps = (MENU_PROP) {
            .pMenuTitle = "ROCm Uninstall",
            .pMenuControlMsg = "<DONE> to exit : Space/Enter key to select/unselect uninstall location",
            .numLines = numItems - 1,
            .numCols = MAX_MENU_ITEM_COLS,
            .starty = ROCM_MENU_ITEM_START_Y,
            .startx = 4,
            .numItems = numItems
        };

        OFFLINE_INSTALL_CONFIG *pConfig = pRocmConfig->pConfig;
        sprintf(uninstall_item_title, "ROCm %s Install Locations: %d", pConfig->rocmVersion, pRocmConfig->rocm_count);

        rocmPathsItems = (ITEMLIST_PARAMS) {
            .numItems           = numItems,
            .pItemListTitle     = uninstall_item_title,
            .pItemListChoices   = rocm_paths_items,
            .pItemListDesp      = rocm_paths_item_desc
        };

        // Create the ROCm Sub-Menu
        create_menu(&menuROCmUninstall, pMenuWindow, &rocmPathsProps, &rocmPathsItems, pConfig);
        
        menuROCmUninstall.enableMultiSelection = false;   // single selection
        menuROCmUninstall.isMenuItemsSelectable = true;   // items are selectable

        // Make the menu multi valued
        menu_opts_off(menuROCmUninstall.pMenu, O_ONEVALUE);

        // set colour for item selection in the menu
        set_menu_fore(menuROCmUninstall.pMenu, WHITE | A_BOLD); // white/bold

        // Disable items from being selectable
        set_menu_grey(menuROCmUninstall.pMenu, BLUE);

        // set item userptrs
        ITEM **items = menu_items(menuROCmUninstall.pMenu);
        set_item_userptr(items[numItems - 2], process_rocm_uninstall_item);    // DONE
        set_item_userptr(items[numItems - 3], process_rocm_uninstall_item);    // UNINSTALL

        for (i = 0; i < pRocmConfig->rocm_count; i++)
        {
            set_item_userptr(items[i], process_rocm_uninstall_item);
        }

        // set the uninstall state / deselect
        for (i = 0; i < pRocmConfig->rocm_count; i++)
        {
            if (rocm_paths_uninstall_state[i] == 1)
            {
                menu_set_item_select(&menuROCmUninstall, i, false);
            }
        }

        // set the item index for the package manager install
        if (g_uninstall_rocm_pkg_index >= 0)
        {
            menu_set_item_select(&menuROCmUninstall, g_uninstall_rocm_pkg_index, false);
        }
    }

    // Set pointer to draw menu function when window is resized
    menuROCmUninstall.drawMenuFunc = rocm_uninstall_menu_draw;

    // Set user pointer for 'ENTER' events
    set_menu_userptr(menuROCmUninstall.pMenu, process_rocm_uninstall_menu);
}

void destroy_rocm_uninstall_menu_window()
{
    destroy_menu(&menuROCmUninstall);

    memset(&menuROCmUninstall, 0, sizeof(menuROCmUninstall));
    memset(rocm_paths_items, 0, sizeof(rocm_paths_items));
    memset(rocm_paths_item_desc, 0, sizeof(rocm_paths_item_desc));
    memset(rocm_paths_uninstall_state, 0, sizeof(rocm_paths_uninstall_state));

    g_uninstall_start_index = 0;
    g_uninstall_end_index = 19;
    
    g_uninstall_rocm_pkg_index = -1;
    g_uninstall_rocm_runfile_index = -1;
}

void update_rocm_uinstall_menu()
{
    destroy_rocm_uninstall_menu_window();

    create_rocm_uninstall_window(menuROCm.pMenuWindow, menuROCm.pConfig);
}

void uninstall_rocm_paths()
{
    int i;

    MENU *pMenu = menuROCmUninstall.pMenu;
    WINDOW *pWin = menuROCmUninstall.pMenuWindow;
    ITEM **items = menuROCmUninstall.itemList[0].items;

    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmUninstall.pConfig)->rocm_config;

    char target[LARGE_CHAR_SIZE];
    size_t len;

    int uninstall_index = -1;

    memset(target, '\0', LARGE_CHAR_SIZE);
    strcat(target, "target=");

    for(i = 0; i < item_count(pMenu); ++i)
    {
        if(item_value(items[i]) == TRUE)
        {
            len = strlen(pRocmConfig->rocm_paths[i]);
            strncat(target, pRocmConfig->rocm_paths[i], len);
            uninstall_index = i;
            break;
        }
    }
    
    // uninstall for specific path index
    if (uninstall_index >= 0)
    {
        strcat(target, " uninstall-rocm");

        // execute the ROCm uninstall command
        if (execute_cmd("./rocm-installer.sh", target, pWin) == 0)
        {
            print_menu_msg(&menuROCm, GREEN, "Uninstall Complete.");

            // update the state for the uninstalled item
            menu_set_item_select(&menuROCmUninstall, uninstall_index, false);
            delete_menu_item_selection_mark(&menuROCmUninstall, items[uninstall_index]);

            rocm_paths_uninstall_state[uninstall_index] = 1;
            pRocmConfig->rocm_count--;

            // if no rocm installs, disable the uninstall item on the rocm menu
            if (pRocmConfig->rocm_count == 0)
            {
                menu_set_item_select(&menuROCm, ROCM_MENU_ITEM_UNINSTALL_ROCM_INDEX, false);
                pRocmConfig->is_rocm_installed = false;
                pRocmConfig->rocm_install_type = eINSTALL_NONE;
            }
        }
        else
        {
            print_menu_err_msg(&menuROCm, "Uninstall Failed.");
        }

        wrefresh(pWin);
    }
}

// rocm uninstall item processing
void process_rocm_uninstall_item()
{
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmUninstall.pConfig)->rocm_config;
    MENU *pMenu = menuROCmUninstall.pMenu;
    ITEM **items = menu_items(pMenu);
    int index = item_index(current_item(pMenu));
    int i;

    DEBUG_UI_MSG(&menuROCmUninstall, "item userptr: item index %d : count %d", index, item_count(pMenu));

    if (index > g_uninstall_end_index)
    {
        // scroll down

        if (index == (pRocmConfig->rocm_count+1))
        {
            // skip the space between DONE and last rocm path
            g_uninstall_start_index += 2;
            g_uninstall_end_index += 2;
        }
        else
        {
            g_uninstall_start_index++;
            g_uninstall_end_index++;
        }
    }
    else if (index < g_uninstall_start_index)
    {
        // scroll up
        g_uninstall_end_index--;
        g_uninstall_start_index--;
    }
    
    draw_rocm_uninstall_types();

    // white/bold
    set_menu_fore(menuROCmUninstall.pMenu, WHITE | A_BOLD);

    // set any selected item to the select colour (cyan)
    for(i = 0; i < item_count(pMenu); ++i)
    {
        if(item_value(items[i]) == TRUE)
        {
            set_menu_fore(menuROCmUninstall.pMenu, CYAN | A_BOLD); 
        }
    }
}

// process "ENTER" key events from the ROCm main menu
void process_rocm_uninstall_menu()
{
    MENU *pMenu = menuROCmUninstall.pMenu;
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmUninstall.pConfig)->rocm_config;

    ITEM *pCurrentItem = current_item(pMenu);

    int index = item_index(pCurrentItem);
    int uninstall_index = item_count(pMenu) - 2;

    DEBUG_UI_MSG(&menuROCmUninstall, "ROCM Uninstall Menu: %d, itemlist %d", index, menuROCmUninstall.curItemListIndex);

    if (index == uninstall_index)
    {
        if (pRocmConfig->rocm_count > 0)
        {
            uninstall_rocm_paths();
        }

        if (pRocmConfig->rocm_count == 0)
        {
            menu_set_item_select(&menuROCmUninstall, uninstall_index, false);
        }

        DEBUG_UI_MSG(&menuROCmUninstall, "uninstall_index %d", uninstall_index);
    }
    else
    {
        DEBUG_UI_MSG(&menuROCmUninstall, "Unknown item index %d", item_count(pMenu));
    }

    rocm_uninstall_menu_draw();
}


/**************** ROCm Device MENU ***************************************************************************/

void create_rocm_menu_device_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig)
{
    int i, item_count;
    char *rocmMenuDeviceOpsPtrs[MAX_MENU_ITEMS];
    char *rocmMenuDeviceDescPtrs[MAX_MENU_ITEMS];

    // Read the item file and descriptions
    item_count = read_menu_items_from_files(ROCM_MENU_DEVICE_ITEMS_FILE, ROCM_MENU_DEVICE_ITEMDECS_FILE,
                                            rocmMenuDeviceOps, rocmMenuDeviceDesc);

    // Setup and create the device menu
    if (item_count != 0) 
    {
        for (int j = 0; j < MAX_MENU_ITEMS; j++)
        {
            rocmMenuDeviceOpsPtrs[j] = rocmMenuDeviceOps[j];
            rocmMenuDeviceDescPtrs[j] = rocmMenuDeviceDesc[j];
        }

        i = item_count;

        // Add menu system items - add one blank line for spacing before HELP
        strcpy(rocmMenuDeviceOps[i], " ");
        strcpy(rocmMenuDeviceDesc[i++], " ");

        strcpy(rocmMenuDeviceOps[i], "<HELP>");
        strcpy(rocmMenuDeviceDesc[i++], DEFAULT_VERBOSE_HELP_WINDOW_MSG);

        strcpy(rocmMenuDeviceOps[i], "<DONE>");
        strcpy(rocmMenuDeviceDesc[i++], " Exit to ROCm Options Menu");

        int numItems = i + 1;  // +1 for consistency with static arrays that include NULL terminator

        rocmMenuDeviceProps = (MENU_PROP) {
            .pMenuTitle = "ROCm Device Configuration",
            .pMenuControlMsg = "<DONE> to exit : Space/Enter to select/unselect device. Up/Down to scroll.",
            .numLines = numItems - 1,
            .numCols = MAX_MENU_ITEM_COLS,
            .starty = ROCM_MENU_ITEM_START_Y,
            .startx = 4,
            .numItems = numItems
        };

        rocmMenuDeviceItems = (ITEMLIST_PARAMS) {
            .numItems           = numItems,
            .pItemListTitle     = "Select which ROCm device you wish to install:",
            .pItemListChoices   = rocmMenuDeviceOpsPtrs,
            .pItemListDesp      = rocmMenuDeviceDescPtrs
        };

        // Create the ROCm Device Menu
        create_menu(&menuROCmDevice, pMenuWindow, &rocmMenuDeviceProps, &rocmMenuDeviceItems, pConfig);

        menuROCmDevice.enableMultiSelection = false;   // single selection
        menuROCmDevice.isMenuItemsSelectable = true;   // items are selectable

        // Set pointer to draw menu function when window is resized
        menuROCmDevice.drawMenuFunc = rocm_menu_submenu_draw;

        // Make the menu multi valued
        menu_opts_off(menuROCmDevice.pMenu, O_ONEVALUE);

        // set colour for item selection in the menu
        set_menu_fore(menuROCmDevice.pMenu, WHITE_ON_BLUE);

        // Create the ROCm Device help menu
        create_help_menu_window(&menuROCmDevice, ROCM_MENU_DEVICE_HELP_TITLE, ROCM_MENU_DEVICE_HELP_FILE);

        // Set user pointer for 'ENTER' events
        set_menu_userptr(menuROCmDevice.pMenu, process_rocm_device_menu);

        menuROCmDevice.clearErrMsgAfterUpOrDownKeyPress = true;
    }
}

void destroy_rocm_menu_device_window()
{
    destroy_help_menu(&menuROCmDevice);
    destroy_menu(&menuROCmDevice);

    memset(&menuROCmDevice, 0, sizeof(menuROCmDevice));
    memset(rocmMenuDeviceOps, 0, sizeof(rocmMenuDeviceOps));
    memset(rocmMenuDeviceDesc, 0, sizeof(rocmMenuDeviceDesc));
}

void do_rocm_menu_device()
{
    MENU *pMenu = menuROCmDevice.pMenu;

    rocm_menu_submenu_draw(&menuROCmDevice);

    // Skip past any initial skippable items (family headers) to first selectable item
    skip_menu_item_down_if_skippable(pMenu);
    print_menu_item_selection(&menuROCmDevice, MENU_SEL_START_Y, MENU_SEL_START_X);

    // ROCm device menu loop
    menu_loop(&menuROCmDevice);

    // Update the ROCm device config on exit
    update_rocm_device_name();

    unpost_menu(pMenu);

    // Clear the X that's added when user selects devices
    wclear(menuROCmDevice.pMenuWindow);
}

void process_rocm_device_menu()
{
    MENU *pMenu = menuROCmDevice.pMenu;

    ITEM *pCurrentItem = current_item(pMenu);
    int index = item_index(pCurrentItem);

    DEBUG_UI_MSG(&menuROCmDevice, "ROCM Device Menu: %d, itemlist %d", index, menuROCmDevice.curItemListIndex);

    if (index == 0)
    {
        
    }
    else
    {
        DEBUG_UI_MSG(&menuROCm, "Unknown item index");
    }
}

// Extract gfx code/group from item name like "    MI325X/MI300X/MI300A (gfx94x)"
// Returns pointer to gfx code within the string, or NULL if not found
const char* extract_gfx_code(const char *item_name)
{
    // Find opening parenthesis
    const char *start = strchr(item_name, '(');
    if (!start) return NULL;

    start++; // Move past '('

    // Check if it starts with "gfx"
    if (strncmp(start, "gfx", 3) != 0) return NULL;

    // Find closing parenthesis to validate format
    const char *end = strchr(start, ')');
    if (!end) return NULL;

    // Return pointer to gfx code/group (e.g., "gfx94x", "gfx950")
    return start;
}

void set_rocm_device_name(int index)
{
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmDevice.pConfig)->rocm_config;
    ITEM **items = menuROCmDevice.itemList[0].items;
    const char *full_item_name = item_name(items[index]);

    // Extract gfx code/group from format like "    MI325X/MI300X/MI300A (gfx94x)"
    const char *gfx_code = extract_gfx_code(full_item_name);

    if (gfx_code)
    {
        // Copy just the gfx code/group (e.g., "gfx94x", "gfx110x", "gfx950")
        const char *end = strchr(gfx_code, ')');
        size_t len = end - gfx_code;

        if (len < DEFAULT_CHAR_SIZE)
        {
            strncpy(pRocmConfig->rocm_device, gfx_code, len);
            pRocmConfig->rocm_device[len] = '\0';
        }
    }
    else
    {
        // Fallback: use full item name (for legacy compatibility)
        strcpy(pRocmConfig->rocm_device, full_item_name);
    }
}

void clear_rocm_device_name()
{
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmDevice.pConfig)->rocm_config;

    // clear the driver version and driver rocm version names
    clear_str(pRocmConfig->rocm_device);
}

void update_rocm_device_name()
{
    int i;
    MENU_DATA *pMenuData = &menuROCmDevice;

    MENU *pMenu = pMenuData->pMenu;
    ITEM **items = pMenuData->itemList[0].items;

    clear_rocm_device_name();

    // check for any selected items in the menu
    for(i = 0; i < item_count(pMenu); ++i)
    {
        if(item_value(items[i]) == TRUE)
        {
            set_rocm_device_name(i);
            break;
        }
    }
}

void reset_rocm_device_name()
{
    ITEM **items;

    if (menuROCmDevice.pMenu)
    {
        // reset the rocm device item selections
        items = menu_items(menuROCmDevice.pMenu);

        for (int i = 0; i < item_count(menuROCmDevice.pMenu); i++)
        {
            if (item_value(items[i]) == TRUE)
            {
                set_item_value(items[i], false);
            }

            delete_menu_item_selection_mark(&menuROCmDevice, items[i]);
        }

        menuROCmDevice.itemSelections = 0;

        // clear the rocm device
        clear_rocm_device_name();
    }
}


/**************** ROCm Component MENU ************************************************************************/

void create_rocm_menu_compo_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig)
{
    int i, item_count;
    char *rocmMenuCompoOpsPtrs[MAX_MENU_ITEMS];
    char *rocmMenuCompoDescPtrs[MAX_MENU_ITEMS];

    // Read the item file and descriptions
    item_count = read_menu_items_from_files(ROCM_MENU_COMPO_ITEMS_FILE, ROCM_MENU_COMPO_ITEMDECS_FILE,
                                            rocmMenuCompoOps, rocmMenuCompoDesc);

    // Setup and create the component menu
    if (item_count != 0)
    {
        for (int j = 0; j < MAX_MENU_ITEMS; j++)
        {
            rocmMenuCompoOpsPtrs[j] = rocmMenuCompoOps[j];
            rocmMenuCompoDescPtrs[j] = rocmMenuCompoDesc[j];
        }

        i = item_count;

        // Add blank separator before core-sdk
        strcpy(rocmMenuCompoOps[i], " ");
        strcpy(rocmMenuCompoDesc[i++], " ");

        // Add core-sdk item
        strcpy(rocmMenuCompoOps[i], "core-sdk");
        strcpy(rocmMenuCompoDesc[i++], "Complete SDK (includes all components above)");

        // Add menu system items
        strcpy(rocmMenuCompoOps[i], " ");
        strcpy(rocmMenuCompoDesc[i++], " ");

        strcpy(rocmMenuCompoOps[i], "<HELP>");
        strcpy(rocmMenuCompoDesc[i++], DEFAULT_VERBOSE_HELP_WINDOW_MSG);

        strcpy(rocmMenuCompoOps[i], "<DONE>");
        strcpy(rocmMenuCompoDesc[i++], " Exit to ROCm Options Menu");

        int numItems = i + 1;  // +1 for consistency with static arrays that include NULL terminator

        rocmMenuCompoProps = (MENU_PROP) {
            .pMenuTitle = "ROCm Component Selection",
            .pMenuControlMsg = "<DONE> to exit : Space/Enter to select/unselect components. Up/Down to scroll.",
            .numLines = numItems - 1,
            .numCols = MAX_MENU_ITEM_COLS,
            .starty = ROCM_MENU_ITEM_START_Y,
            .startx = 4,
            .numItems = numItems
        };

        rocmMenuCompoItems = (ITEMLIST_PARAMS) {
            .numItems           = numItems,
            .pItemListTitle     = "Select which component(s) you wish to install:",
            .pItemListChoices   = rocmMenuCompoOpsPtrs,
            .pItemListDesp      = rocmMenuCompoDescPtrs
        };

        // Create the ROCm Component Menu
        create_menu(&menuROCmCompo, pMenuWindow, &rocmMenuCompoProps, &rocmMenuCompoItems, pConfig);
        menuROCmCompo.isMenuItemsSelectable = true;   // items are selectable

        // Set pointer to draw menu function when window is resized
        menuROCmCompo.drawMenuFunc = rocm_menu_submenu_draw;

        // Make the menu multi valued
        menu_opts_off(menuROCmCompo.pMenu, O_ONEVALUE);

        // set colour for item selection in the menu
        set_menu_fore(menuROCmCompo.pMenu, WHITE_ON_BLUE);

        // Create the ROCm Component help menu
        create_help_menu_window(&menuROCmCompo, ROCM_MENU_COMPO_HELP_TITLE, ROCM_MENU_COMPO_HELP_FILE);

        // Set user pointer for 'ENTER' events
        set_menu_userptr(menuROCmCompo.pMenu, process_rocm_compo_menu);

        menuROCmCompo.clearErrMsgAfterUpOrDownKeyPress = true;
    }
}

void destroy_rocm_menu_compo_window()
{
    destroy_help_menu(&menuROCmCompo);
    destroy_menu(&menuROCmCompo);

    memset(&menuROCmCompo, 0, sizeof(menuROCmCompo));
    memset(rocmMenuCompoOps, 0, sizeof(rocmMenuCompoOps));
    memset(rocmMenuCompoDesc, 0, sizeof(rocmMenuCompoDesc));
}

void update_rocm_compo_selection_state()
{
    MENU *pMenu = menuROCmCompo.pMenu;
    ITEM **items = menuROCmCompo.itemList[0].items;
    int core_sdk_index = 5;  // core-sdk is now at index 5 (after blank separator)

    // Check if core-sdk is selected
    bool core_sdk_selected = item_value(items[core_sdk_index]);

    if (core_sdk_selected)
    {
        // Clear all X marks first
        for (int i = 0; i < item_count(pMenu); i++)
        {
            delete_menu_item_selection_mark(&menuROCmCompo, items[i]);
        }

        // Disable and highlight other components to show they're included
        for (int i = 0; i < core_sdk_index; i++)
        {
            item_opts_off(items[i], O_SELECTABLE);
            set_item_value(items[i], TRUE);  // Auto-select them
        }

        // Set disabled items to WHITE_ON_BLUE (same as cursor)
        set_menu_grey(pMenu, WHITE_ON_BLUE);

        // Redraw X marks for all selected items
        for (int i = 0; i < item_count(pMenu); i++)
        {
            if (item_value(items[i]))
            {
                add_menu_item_selection_mark(&menuROCmCompo, items[i]);
            }
        }
    }
    else
    {
        // Re-enable other components
        for (int i = 0; i < core_sdk_index; i++)
        {
            item_opts_on(items[i], O_SELECTABLE);
        }
        // Restore normal grey color for disabled items
        set_menu_grey(pMenu, BLUE);
    }
}

void do_rocm_menu_compo()
{
    MENU *pMenu = menuROCmCompo.pMenu;

    rocm_menu_submenu_draw(&menuROCmCompo);

    // ROCm component menu loop
    menu_loop(&menuROCmCompo);

    // Update the ROCm components config on exit
    update_rocm_components_name();

    unpost_menu(pMenu);

    // Clear the X that's added when user selects components
    wclear(menuROCmCompo.pMenuWindow);
}

void process_rocm_compo_menu()
{
#if ENABLE_MENU_DEBUG
    MENU *pMenu = menuROCmCompo.pMenu;
    ITEM *pCurrentItem = current_item(pMenu);
    int index = item_index(pCurrentItem);
    DEBUG_UI_MSG(&menuROCmCompo, "ROCM Component Menu: %d, itemlist %d", index, menuROCmCompo.curItemListIndex);
#endif

    // Update selection state after any selection change
    update_rocm_compo_selection_state();
}

void set_rocm_components_name(int index)
{
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmCompo.pConfig)->rocm_config;
    ITEM **items = menuROCmCompo.itemList[0].items;

    // update the rocm components name
    strcpy(pRocmConfig->rocm_components, item_name(items[index]));
}

void clear_rocm_components_name()
{
    ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmCompo.pConfig)->rocm_config;

    // clear the rocm components name
    clear_str(pRocmConfig->rocm_components);
}

void update_rocm_components_name()
{
    int i;
    MENU_DATA *pMenuData = &menuROCmCompo;

    MENU *pMenu = pMenuData->pMenu;
    ITEM **items = pMenuData->itemList[0].items;
     ROCM_MENU_CONFIG *pRocmConfig = &(menuROCmCompo.pConfig)->rocm_config;

    clear_rocm_components_name();

    // check for any selected items in the menu
    for(i = 0; i < item_count(pMenu); ++i)
    {
        if(item_value(items[i]) == TRUE)
        {
            strcat(pRocmConfig->rocm_components, item_name(items[i]));
            strcat(pRocmConfig->rocm_components, ",");
        }
    }

    if (strlen(pRocmConfig->rocm_components) != 0)
    {
        pRocmConfig->rocm_components[strlen(pRocmConfig->rocm_components)-1] = '\0';
    }
}

void reset_rocm_components_name()
{
    ITEM **items;

    if (menuROCmCompo.pMenu)
    {
        // reset the rocm component item selections
        items = menu_items(menuROCmCompo.pMenu);

        for (int i = 0; i < item_count(menuROCmCompo.pMenu); i++)
        {
            if (item_value(items[i]) == TRUE)
            {
                set_item_value(items[i], false);
            }

            delete_menu_item_selection_mark(&menuROCmCompo, items[i]);
        }

        menuROCmCompo.itemSelections = 0;

        // clear the rocm components
        clear_rocm_components_name();
    }
}
