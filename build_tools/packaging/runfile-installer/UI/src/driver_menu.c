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
#include "driver_menu.h"
#include "help_menu.h"
#include "utils.h"


// Driver Menu Setup
char *driverMenuOps[] = {
    "Install Driver",
    "   Start on install",
    SKIPPABLE_MENU_ITEM,
    "Uninstall Driver",
    SKIPPABLE_MENU_ITEM,
    "<HELP>",
    "<DONE>",
    (char*)NULL,
};

char *driverMenuDesc[] = {
    "Enable/Disable amdgpu driver install.  Enabling will search for amdgpu driver.",
    "Start up amdgpu driver after installation.",
    SKIPPABLE_MENU_ITEM,
    "Uninstall runfile amdgpu driver.",
    SKIPPABLE_MENU_ITEM,
    DEFAULT_VERBOSE_HELP_WINDOW_MSG,
    "Exit to Main Menu",
    (char*)NULL,
};


MENU_PROP driverMenuProps = {
    .pMenuTitle = "Driver Options",
    .pMenuControlMsg = "<DONE> to exit : Enter key to toggle selection",
    .numLines = ARRAY_SIZE(driverMenuOps) - 1,
    .numCols = MAX_MENU_ITEM_COLS,
    .starty = DRIVER_MENU_ITEM_START_Y,
    .startx = DRIVER_MENU_ITEM_START_X, 
    .numItems = ARRAY_SIZE(driverMenuOps)
};

ITEMLIST_PARAMS driverMenuItems = {
    .numItems           = (ARRAY_SIZE(driverMenuOps)),
    .pItemListTitle     = "Settings:",
    .pItemListChoices   = driverMenuOps,
    .pItemListDesp      = driverMenuDesc
};

// verbose help menu variables
// Spaces added/deleted from HelpOps and HelpDesc to ensure whole words aren't
// cut off between lines when displaying help menu.
char *driverMenuHelpOps[] = {
    "Install Driver",
    "Start on install",
    "Uninstall Driver",
    (char*)NULL,
};

char *driverMenuHelpDesc[] = {
    "Enable/Disable the inclusion the amdgpu driver         as part of the installation.",
    "Enable/Disable starting the amdgpu driver after        installation.",
    "Uninstall runfile-based amdgpu driver installation.    This is not available for package installed drivers.",
    (char*)NULL,
};

MENU_PROP driverMenuHelpProps = {
    .pMenuTitle = "Driver Options Help",
    .pMenuControlMsg = DEFAULT_VERBOSE_HELP_CONTROL_MSG,
    .numLines = 0,
    .numCols = MAX_MENU_ITEM_COLS,
    .starty = DRIVER_MENU_ITEM_START_Y,
    .startx = DRIVER_MENU_ITEM_START_X, 
    .numItems = 0
};

ITEMLIST_PARAMS driverMenuHelpItems = {
    .numItems           = 0,
    .pItemListTitle     = "Driver Install Settings Description:",
    .pItemListChoices   = 0,
    .pItemListDesp      = 0
};


void process_driver_menu();

// menu draw/config
void driver_menu_toggle_grey_items(bool enable);

// sub-menus
void create_driver_help_menu_window();

// menu draw
void driver_menu_draw();

MENU_DATA menuDriver = {0};
bool gDriverStatusCheck = false;


/**************** Driver MENU **********************************************************************************/

void create_driver_menu_window(WINDOW *pMenuWindow, OFFLINE_INSTALL_CONFIG *pConfig)
{
    // Create the driver options menu
    create_menu(&menuDriver, pMenuWindow, &driverMenuProps, &driverMenuItems, pConfig);

    // create verbose help menu
    menuDriver.pHelpMenu = calloc(1, sizeof(MENU_DATA));
    if (menuDriver.pHelpMenu)
    {
        create_driver_help_menu_window();
    }

    // Set pointer to draw menu function when window is resized
    menuDriver.drawMenuFunc = driver_menu_draw;

    // Set user pointers for 'ENTER' events
    set_menu_userptr(menuDriver.pMenu, process_driver_menu);

    // set items to non-selectable
    set_menu_grey(menuDriver.pMenu, COLOR_PAIR(5));
    menu_set_item_select(&menuDriver, menuDriver.itemList[0].numItems - 4, false);    // space before help
    driver_menu_toggle_grey_items(false);
}

void destroy_driver_menu_window()
{
    destroy_help_menu(menuDriver.pHelpMenu);
    destroy_menu(&menuDriver);
}

void driver_menu_toggle_grey_items(bool enable)
{
    DRIVER_MENU_CONFIG *pDriverConfig = &(menuDriver.pConfig)->driver_config;

    if (enable)
    {
        // enable all driver option fields
        if (pDriverConfig->install_driver)
        {
            menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_START_DRIVER_INDEX, true);
        }

        if (pDriverConfig->is_driver_installed)
        {
            menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX, true);
        }
    }
    else
    {
        // disable all driver option fields
        menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_INSTALL_DRIVER_INDEX, false);
        menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_START_DRIVER_INDEX, false);
        menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX, false);
    }
}

void check_driver_install_status()
{
    OFFLINE_INSTALL_CONFIG *pConfig = menuDriver.pConfig;
    DRIVER_MENU_CONFIG *pDriverConfig = &(menuDriver.pConfig)->driver_config;
    char amdgpu_dkms_path[DEFAULT_CHAR_SIZE];

    gDriverStatusCheck = true;

    // check if dkms is installed first
    if (is_dkms_pkg_installed(pConfig->distroType) == 0)
    {
        // if no dkms - driver cannot be installed
        gDriverStatusCheck = false;
        pDriverConfig->driver_install_type = eINSTALL_NODKMS;
        pDriverConfig->is_driver_installed = true;
        pDriverConfig->install_driver = false;
        menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_INSTALL_DRIVER_INDEX, false);
        return;
    }
    
    // check if for package installation of amdgpu-dkms
    if (is_amdgpu_dkms_pkg_installed(pConfig->distroType) > 0)
    {
        // package is installed
        pDriverConfig->driver_install_type = eINSTALL_PACKAGE;
        pDriverConfig->is_driver_installed = true;
    }
    else
    {
        OFFLINE_INSTALL_CONFIG *pConfig = pDriverConfig->pConfig;
        strcpy(amdgpu_dkms_path, DRIVER_DKMS_PATH);

        // if no package install, check for a runfile install - check for an amdgpu-dkms build
        if ( is_dir_exist(amdgpu_dkms_path) )
        {
            strcat(amdgpu_dkms_path, pConfig->amdgpuDkmsBuild);
            if ( is_dir_exist(amdgpu_dkms_path) )
            {
                // there is a dkms amdgpu - not package installed
                // if not package install, then there is a runfile dkms install of amdgpu
                pDriverConfig->driver_install_type = eINSTALL_RUNFILE;
                pDriverConfig->is_driver_installed = true;
            }
        }
        else
        {
            // no driver install
            pDriverConfig->driver_install_type = eINSTALL_NONE;
            pDriverConfig->is_driver_installed = false;
        }
    }
    
    // grey-out the driver install ops depending if the driver is installed or not
    menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_INSTALL_DRIVER_INDEX, !pDriverConfig->is_driver_installed);

    // allow uninstall for runfile only
    if (pDriverConfig->driver_install_type == eINSTALL_RUNFILE)
    {
        menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX, pDriverConfig->is_driver_installed);
    }
    else
    {
        menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX, false);
    }
}

void driver_clear_status()
{
    WINDOW *pMenuWindow = menuDriver.pMenuWindow;

    wmove(pMenuWindow, DRIVER_MENU_DRIVER_STATUS_INFO_ROW, DRIVER_MENU_DRIVER_STATUS_INFO_COL);
    wclrtoeol(pMenuWindow);
}

void driver_status_draw()
{
    WINDOW *pMenuWindow = menuDriver.pMenuWindow;
    DRIVER_MENU_CONFIG *pDriverConfig = &(menuDriver.pConfig)->driver_config;
    
    // check for the driver status and draw
    if (pDriverConfig->driver_install_type == 0)
    {
        print_menu_msg(&menuDriver, COLOR_PAIR(4), "amdgpu driver not installed.");
    }
    else if (pDriverConfig->driver_install_type == eINSTALL_PACKAGE)
    {
        print_menu_err_msg(&menuDriver, "amdgpu driver package install found. Uninstall required.");
    }
    else if (pDriverConfig->driver_install_type == eINSTALL_RUNFILE)
    {
        OFFLINE_INSTALL_CONFIG *pConfig = pDriverConfig->pConfig;
        mvwprintw(pMenuWindow, DRIVER_MENU_DRIVER_STATUS_INFO_ROW, DRIVER_MENU_DRIVER_STATUS_INFO_COL, "%s", pConfig->amdgpuDkmsBuild);
        print_menu_err_msg(&menuDriver, "amdgpu driver runfile install found.  Uninstall required.");
    }
    else if (pDriverConfig->driver_install_type == eINSTALL_NODKMS)
    {
        print_menu_err_msg(&menuDriver, "dkms is not installed. Unable to install amdgpu driver.");
    }
    else
    {
        print_menu_err_msg(&menuDriver, "amdgpu driver installation status unknown.");
    }
}

void driver_menu_draw()
{
    DRIVER_MENU_CONFIG *pDriverConfig = &(menuDriver.pConfig)->driver_config;

    menu_info_draw_bool(&menuDriver, DRIVER_MENU_ITEM_INSTALL_DRIVER_ROW, DRIVER_MENU_FORM_COL, pDriverConfig->install_driver);
    menu_info_draw_bool(&menuDriver, DRIVER_MENU_ITEM_START_DRIVER_ROW, DRIVER_MENU_FORM_COL, pDriverConfig->start_driver);

    menu_draw(&menuDriver);
}

void do_driver_menu()
{
    MENU *pMenu = menuDriver.pMenu;

    wclear(menuDriver.pMenuWindow);

    // draw the driver menu contents
    driver_menu_draw(&menuDriver);

    // driver menu loop
    menu_loop(&menuDriver);

    unpost_menu(pMenu);
}

// process "ENTER" key events from the Extra packages main menu
void process_driver_menu()
{
    MENU *pMenu = menuDriver.pMenu;
    WINDOW *pWin = menuDriver.pMenuWindow;

    DRIVER_MENU_CONFIG *pDriverConfig = &(menuDriver.pConfig)->driver_config;
    
    ITEM *pCurrentItem = current_item(pMenu);

    int index = item_index(pCurrentItem);

    DEBUG_UI_MSG(&menuDriver, "driver menu: item %d", index);

    if (index == DRIVER_MENU_ITEM_INSTALL_DRIVER_INDEX)
    {
        // check the driver status
        if (!gDriverStatusCheck) check_driver_install_status();
        driver_status_draw();

        // allow toggle of driver install if driver is not currently installed
        if (!pDriverConfig->is_driver_installed)
        {
            pDriverConfig->install_driver = !pDriverConfig->install_driver;
            menu_info_draw_bool(&menuDriver, DRIVER_MENU_ITEM_INSTALL_DRIVER_ROW, DRIVER_MENU_FORM_COL, pDriverConfig->install_driver);
            driver_menu_toggle_grey_items(pDriverConfig->install_driver);

            if (!pDriverConfig->install_driver)
            {
                gDriverStatusCheck = false; // reset the driver install check if install driver toggled off
                clear_menu_msg(&menuDriver);
            }
        }
    }
    else if (index == DRIVER_MENU_ITEM_START_DRIVER_INDEX)
    {
        // allow toggle of start driver only for driver install
        if (pDriverConfig->install_driver)
        {
            pDriverConfig->start_driver = !pDriverConfig->start_driver;
            menu_info_draw_bool(&menuDriver, DRIVER_MENU_ITEM_START_DRIVER_ROW, DRIVER_MENU_FORM_COL, pDriverConfig->start_driver);
        }
    }
    else if (index == DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX)
    {
        // only uninstall if the driver is installed
        if (pDriverConfig->is_driver_installed && (pDriverConfig->driver_install_type == eINSTALL_RUNFILE))
        {
            // execute the amdgpu uninstall command
            if (execute_cmd("./rocm-installer.sh", "uninstall-amdgpu", pWin) == 0)
            {
                print_menu_msg(&menuDriver, COLOR_PAIR(4), "Uninstall Complete. Reboot required.");

                // driver install success, disable the uninstall item on the driver menu and reset driver install status
                menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX, false);
                pDriverConfig->is_driver_installed = false;
                gDriverStatusCheck = false;

                driver_clear_status();
            }
            else
            {
                print_menu_err_msg(&menuDriver, "Uninstall Failed.");
            }

            wrefresh(pWin);
        }
    }
    else
    {
        DEBUG_UI_MSG(&menuDriver, "Unknown item index");
    }

    driver_menu_draw(&menuDriver);
}

void create_driver_help_menu_window()
{
    MENU_DATA *pMenuData = menuDriver.pHelpMenu;
    WINDOW *pMenuWindow = menuDriver.pMenuWindow;
    
    // Create menu window w/ border and title
    create_menu(pMenuData, pMenuWindow, &driverMenuHelpProps, &driverMenuHelpItems, NULL);

    menu_opts_off(pMenuData->pMenu, O_SHOWDESC);

    // create form that displays verbose help menu
    create_help_form(pMenuData, pMenuWindow, DRIVER_HELP_MENU_DESC_STARTX, DRIVER_HELP_MENU_DESC_STARTY, HELP_MENU_DESC_WIDTH, HELP_MENU_OP_STARTX, HELP_MENU_OP_WIDTH, driverMenuHelpOps, driverMenuHelpDesc); 
}
