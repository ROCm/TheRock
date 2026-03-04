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

void process_driver_menu();

// menu draw/config
void driver_menu_toggle_grey_items(bool enable);

// menu draw
void driver_menu_draw();

MENU_DATA menuDriver = {0};
bool gDriverStatusCheck = false;

// Global config pointers (defined in rocm_ui.c)
extern OFFLINE_INSTALL_CONFIG *g_pConfig;
extern DRIVER_MENU_CONFIG *g_pDriverConfig;


/**************** Driver MENU **********************************************************************************/

void create_driver_menu_window(WINDOW *pMenuWindow)
{
    // Create the driver options menu
    create_menu(&menuDriver, pMenuWindow, &driverMenuProps, &driverMenuItems, g_pConfig);

    // Create help menu
    create_help_menu_window(&menuDriver, DRIVER_MENU_HELP_TITLE, DRIVER_MENU_HELP_FILE);

    // Set pointer to draw menu function when window is resized
    menuDriver.drawMenuFunc = driver_menu_draw;

    // Set user pointers for 'ENTER' events
    set_menu_userptr(menuDriver.pMenu, process_driver_menu);

    // set items to non-selectable
    set_menu_grey(menuDriver.pMenu, BLUE);
    menu_set_item_select(&menuDriver, menuDriver.itemList[0].numItems - 4, false);    // space before help
    driver_menu_toggle_grey_items(false);
}

void destroy_driver_menu_window()
{
    destroy_help_menu(&menuDriver);
    destroy_menu(&menuDriver);
}

void driver_menu_toggle_grey_items(bool enable)
{
    if (enable)
    {
        // enable all driver option fields
        if (g_pDriverConfig->install_driver)
        {
            menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_START_DRIVER_INDEX, true);
        }

        if (g_pDriverConfig->is_driver_installed)
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
    char amdgpu_dkms_path[DEFAULT_CHAR_SIZE];

    gDriverStatusCheck = true;

    // check if dkms is installed first
    if (is_dkms_pkg_installed(g_pConfig->distroType) == 0)
    {
        // if no dkms - driver cannot be installed
        gDriverStatusCheck = false;
        g_pDriverConfig->driver_install_type = eINSTALL_NODKMS;
        g_pDriverConfig->is_driver_installed = true;
        g_pDriverConfig->install_driver = false;
        menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_INSTALL_DRIVER_INDEX, false);
        return;
    }

    // check if for package installation of amdgpu-dkms
    if (is_amdgpu_dkms_pkg_installed(g_pConfig->distroType) > 0)
    {
        // package is installed
        g_pDriverConfig->driver_install_type = eINSTALL_PACKAGE;
        g_pDriverConfig->is_driver_installed = true;
    }
    else
    {
        strcpy(amdgpu_dkms_path, DRIVER_DKMS_PATH);

        // Check for runfile install - look for any amdgpu version in DKMS (same as rocm-installer.sh)
        // Use check_dkms_status() which parses `dkms status` to detect any installed amdgpu driver
        char dkms_output[LARGE_CHAR_SIZE];
        if (check_dkms_status(dkms_output, LARGE_CHAR_SIZE) == 0)
        {
            // Check if amdgpu is in the DKMS output
            if (strstr(dkms_output, "amdgpu") != NULL)
            {
                // DKMS has amdgpu (runfile install, not package)
                g_pDriverConfig->driver_install_type = eINSTALL_RUNFILE;
                g_pDriverConfig->is_driver_installed = true;
            }
            else
            {
                // DKMS exists but no amdgpu driver
                g_pDriverConfig->driver_install_type = eINSTALL_NONE;
                g_pDriverConfig->is_driver_installed = false;
            }
        }
        else
        {
            // DKMS check failed or no driver install
            g_pDriverConfig->driver_install_type = eINSTALL_NONE;
            g_pDriverConfig->is_driver_installed = false;
        }
    }
    
    // grey-out the driver install ops depending if the driver is installed or not
    menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_INSTALL_DRIVER_INDEX, !g_pDriverConfig->is_driver_installed);

    // allow uninstall for runfile only
    if (g_pDriverConfig->driver_install_type == eINSTALL_RUNFILE)
    {
        menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX, g_pDriverConfig->is_driver_installed);
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
    
    // check for the driver status and draw
    if (g_pDriverConfig->driver_install_type == 0)
    {
        print_menu_msg(&menuDriver, GREEN, "amdgpu driver not installed.");
    }
    else if (g_pDriverConfig->driver_install_type == eINSTALL_PACKAGE)
    {
        print_menu_err_msg(&menuDriver, "amdgpu driver package install found. Uninstall required.");
    }
    else if (g_pDriverConfig->driver_install_type == eINSTALL_RUNFILE)
    {
        mvwprintw(pMenuWindow, DRIVER_MENU_DRIVER_STATUS_INFO_ROW, DRIVER_MENU_DRIVER_STATUS_INFO_COL, "%s", g_pConfig->amdgpuDkmsBuild);
        print_menu_err_msg(&menuDriver, "amdgpu driver runfile install found.  Uninstall required.");
    }
    else if (g_pDriverConfig->driver_install_type == eINSTALL_NODKMS)
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

    menu_info_draw_bool(&menuDriver, DRIVER_MENU_ITEM_INSTALL_DRIVER_ROW, DRIVER_MENU_FORM_COL, g_pDriverConfig->install_driver);
    menu_info_draw_bool(&menuDriver, DRIVER_MENU_ITEM_START_DRIVER_ROW, DRIVER_MENU_FORM_COL, g_pDriverConfig->start_driver);

    menu_draw(&menuDriver);
}

void do_driver_menu()
{
    MENU *pMenu = menuDriver.pMenu;

    wclear(menuDriver.pMenuWindow);

    // draw the driver menu contents
    driver_menu_draw();

    // driver menu loop
    menu_loop(&menuDriver);

    unpost_menu(pMenu);
}

// process "ENTER" key events from the Extra packages main menu
void process_driver_menu()
{
    MENU *pMenu = menuDriver.pMenu;
    WINDOW *pWin = menuDriver.pMenuWindow;
    ITEM *pCurrentItem = current_item(pMenu);

    int index = item_index(pCurrentItem);

    DEBUG_UI_MSG(&menuDriver, "driver menu: item %d", index);

    if (index == DRIVER_MENU_ITEM_INSTALL_DRIVER_INDEX)
    {
        // check the driver status
        if (!gDriverStatusCheck) check_driver_install_status();
        driver_status_draw();

        // allow toggle of driver install if driver is not currently installed
        if (!g_pDriverConfig->is_driver_installed)
        {
            g_pDriverConfig->install_driver = !g_pDriverConfig->install_driver;
            menu_info_draw_bool(&menuDriver, DRIVER_MENU_ITEM_INSTALL_DRIVER_ROW, DRIVER_MENU_FORM_COL, g_pDriverConfig->install_driver);
            driver_menu_toggle_grey_items(g_pDriverConfig->install_driver);

            if (!g_pDriverConfig->install_driver)
            {
                gDriverStatusCheck = false; // reset the driver install check if install driver toggled off
                clear_menu_msg(&menuDriver);
            }
        }
    }
    else if (index == DRIVER_MENU_ITEM_START_DRIVER_INDEX)
    {
        // allow toggle of start driver only for driver install
        if (g_pDriverConfig->install_driver)
        {
            g_pDriverConfig->start_driver = !g_pDriverConfig->start_driver;
            menu_info_draw_bool(&menuDriver, DRIVER_MENU_ITEM_START_DRIVER_ROW, DRIVER_MENU_FORM_COL, g_pDriverConfig->start_driver);
        }
    }
    else if (index == DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX)
    {
        // only uninstall if the driver is installed
        if (g_pDriverConfig->is_driver_installed && (g_pDriverConfig->driver_install_type == eINSTALL_RUNFILE))
        {
            // execute the amdgpu uninstall command
            if (execute_cmd("./rocm-installer.sh", "uninstall-amdgpu", pWin) == 0)
            {
                print_menu_msg(&menuDriver, GREEN, "Uninstall Complete. Reboot required.");

                // driver install success, disable the uninstall item on the driver menu and reset driver install status
                menu_set_item_select(&menuDriver, DRIVER_MENU_ITEM_UNINSTALL_DRIVER_INDEX, false);
                g_pDriverConfig->is_driver_installed = false;
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

    driver_menu_draw();
}
