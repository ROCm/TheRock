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
#include "rocm_menu.h"
#include "driver_menu.h"
#include "post_menu.h"
#include "menu_data.h"

#include "config.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/utsname.h>
#include <sys/stat.h>

#define MAIN_MENU_ITEM_START_Y          7   // minimum starting y/row
#define MAIN_MENU_ITEM_START_X          1   // minimum starting x/col

// rocm menu indices
#define MAIN_MENU_ITEM_PRE_INDEX        0
#define MAIN_MENU_ITEM_ROCM_INDEX       2
#define MAIN_MENU_ITEM_DRIVER_INDEX     3
#define MAIN_MENU_ITEM_POST_INDEX       5
#define MAIN_MENU_ITEM_INSTALL_INDEX    7


// Main Menu Setup
char *mainMenuOps[] = {
    "Pre-Install Configuration",
    SKIPPABLE_MENU_ITEM,
    "ROCm Options",
    "Driver Options",
    SKIPPABLE_MENU_ITEM,
    "Post-Install Configuration",
    SKIPPABLE_MENU_ITEM,
    "< INSTALL >",
    (char *)NULL,
};

char *mainMenuDesc[] = {
    "Pre-installation configuration.",
    " ",
    "Set ROCm install options.",
    "Set amdgpu driver install options.",
    " ",
    "Post-installation configuration.",
    " ",
    "Install ROCm configuration.",
    (char*)NULL,
};

MENU_PROP mainMenuProperties = {
    .pMenuTitle = "ROCm Runfile Installer",
    .pMenuControlMsg = "F1 to exit",
    .numLines = ARRAY_SIZE(mainMenuOps) - 1,
    .numCols = MAX_MENU_ITEM_COLS,
    .startx = MAIN_MENU_ITEM_START_X,
    .starty = MAIN_MENU_ITEM_START_Y,
    .numItems = ARRAY_SIZE(mainMenuOps)
};

ITEMLIST_PARAMS mainMenuItems = {
    .numItems           = ARRAY_SIZE(mainMenuOps),
    .pItemListTitle     = "Main Menu",
    .pItemListChoices   = mainMenuOps,
    .pItemListDesp      = mainMenuDesc
};

void get_os_release_value(char *key, char *value) 
{
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen("/etc/os-release", "r");
    if (fp == NULL) 
    {
        return;
    }

    while ((read = getline(&line, &len, fp)) != -1) 
    {
        if (strstr(line, key) != NULL) 
        {
            if (key[0] == line[0])
            {
                char *p = strchr(line, '=');
                if (p[1] == '\"')
                {
                    strcpy(value, p + 2);
                     value[strlen(value) - 2] = '\0';
                }
                else
                {
                    strcpy(value, p + 1);
                    value[strlen(value) - 1] = '\0';
                }
                break;
            }
        }
    }
    fclose(fp);

    if (line) 
    {
        free(line);
    }
}

int get_os_info(OFFLINE_INSTALL_CONFIG *pConfig)
{
    uint32_t i;
    struct utsname unameData;

    if (uname(&unameData) < 0)
    {
        return -1;
    }
    
    strcpy(pConfig->kernelVersion, unameData.release);

    get_os_release_value("PRETTY_NAME", pConfig->distroName);
    get_os_release_value("ID", pConfig->distroID);
    get_os_release_value("VERSION_ID", pConfig->distroVersion);

    char *debList[] = {"ubuntu", "debian"};
    char *elList[]  = {"rhel", "ol"};
    char *sleList[] = {"sles"};

    for (i = 0; i < ARRAY_SIZE(debList); i++) 
    {
        if (strstr(pConfig->distroID, debList[i]) != NULL) 
        {
            pConfig->distroType = eDISTRO_TYPE_DEB;
        }
    }

    for (i = 0; i < ARRAY_SIZE(elList); i++) 
    {
        if (strstr(pConfig->distroID, elList[i]) != NULL) 
        {
            pConfig->distroType = eDISTRO_TYPE_EL;
        }
    }

    for (i = 0; i < ARRAY_SIZE(sleList); i++) 
    {
        if (strstr(pConfig->distroID, sleList[i]) != NULL) 
        {
            pConfig->distroType = eDISTRO_TYPE_SLE;
        }
    }

    return 0;
}

/**
 * read_version_file - Read installer version information from VERSION file
 * @pConfig: Pointer to configuration structure to populate
 *
 * Reads the VERSION file with format:
 *   Line 1: INSTALLER_VERSION (e.g., "2.0.0")
 *   Line 2: ROCM_VER (e.g., "7.11.0")
 *   Line 3: BUILD_TAG (e.g., "1", "rc1", "nightly")
 *   Line 4: BUILD_RUNID (e.g., "99999", "1")
 *   Line 5: BUILD_PULL_TAG (e.g., "20260219-22188089855")
 *   Line 6: AMDGPU_DKMS_BUILD_NUM (e.g., "6.18.4-2286447")
 *
 * Returns: 0 on success, -1 on failure
 */
int read_version_file(OFFLINE_INSTALL_CONFIG *pConfig)
{
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int line_num = 0;

    fp = fopen("./VERSION", "r");
    if (fp == NULL)
    {
        // Fallback: try relative path if running from different location
        fp = fopen("VERSION", "r");
        if (fp == NULL)
        {
            fprintf(stderr, "ERROR: Could not open VERSION file\n");
            return -1;
        }
    }

    while ((read = getline(&line, &len, fp)) != -1 && line_num < 6)
    {
        // Remove trailing newline
        if (read > 0 && line[read - 1] == '\n')
            line[read - 1] = '\0';

        switch(line_num)
        {
            case 0:
                strncpy(pConfig->installerVersion, line, sizeof(pConfig->installerVersion) - 1);
                pConfig->installerVersion[sizeof(pConfig->installerVersion) - 1] = '\0';
                break;
            case 1:
                strncpy(pConfig->rocmVersion, line, sizeof(pConfig->rocmVersion) - 1);
                pConfig->rocmVersion[sizeof(pConfig->rocmVersion) - 1] = '\0';
                break;
            case 2:
                strncpy(pConfig->buildTag, line, sizeof(pConfig->buildTag) - 1);
                pConfig->buildTag[sizeof(pConfig->buildTag) - 1] = '\0';
                break;
            case 3:
                strncpy(pConfig->buildRunId, line, sizeof(pConfig->buildRunId) - 1);
                pConfig->buildRunId[sizeof(pConfig->buildRunId) - 1] = '\0';
                break;
            case 4:
                strncpy(pConfig->buildPullTag, line, sizeof(pConfig->buildPullTag) - 1);
                pConfig->buildPullTag[sizeof(pConfig->buildPullTag) - 1] = '\0';
                break;
            case 5:
                strncpy(pConfig->amdgpuDkmsBuild, line, sizeof(pConfig->amdgpuDkmsBuild) - 1);
                pConfig->amdgpuDkmsBuild[sizeof(pConfig->amdgpuDkmsBuild) - 1] = '\0';
                break;
        }
        line_num++;
    }

    fclose(fp);
    if (line)
    {
        free(line);
    }

    return 0;
}

void main_menu_draw(MENU_DATA *pMenuData, OFFLINE_INSTALL_CONFIG *pConfig)
{
    ROCM_MENU_CONFIG *pRocmConfig = &pConfig->rocm_config;

    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    wclear(pMenuWindow);

    char installer_build[DEFAULT_CHAR_SIZE];
    sprintf(installer_build, "%s", pConfig->distroName);

    // resizes pMenuWindow and subwindow that displays menu items to its original
    // size in case user resized terminal window
    resize_and_reposition_window_and_subwindow(pMenuData, WIN_NUM_LINES, WIN_WIDTH_COLS);

    print_menu_title(pMenuData, MENU_TITLE_Y, MENU_TITLE_X, WIN_WIDTH_COLS, "ROCm Runfile Installer", COLOR_PAIR(2));
    print_menu_item_title(pMenuData, 3, 2, installer_build, COLOR_PAIR(9));

    print_menu_control_msg(pMenuData);

    print_version(pMenuData);

    // Display a warning if ROCm is installed on the system
    if (pRocmConfig->install_rocm && pRocmConfig->is_rocm_installed && pRocmConfig->is_rocm_path_valid)
    {
        if (pRocmConfig->rocm_install_type == eINSTALL_PACKAGE)
        {
            print_menu_err_msg(pMenuData, "ROCm %s package manager installed for target.", pConfig->rocmVersion);
        }
        else
        {
            print_menu_warning_msg(pMenuData, "ROCm %s installed for target", pConfig->rocmVersion);
        }
    }
    else
    {
        clear_menu_msg(pMenuData);
    }

    box(pMenuData->pMenuWindow, 0, 0);
}

void config_install(OFFLINE_INSTALL_CONFIG *pConfig, char *cmdArgs)
{
    char installcomps[SMALL_CHAR_SIZE];
    char target[LARGE_CHAR_SIZE];
    char postrocm[SMALL_CHAR_SIZE];
    char postinstall[SMALL_CHAR_SIZE];

    clear_str(installcomps);
    clear_str(target);
    clear_str(postrocm);
    clear_str(postinstall);

    // Check if rocm is being installed
    if (pConfig->rocm_config.install_rocm)
    {
        sprintf(installcomps, "rocm");
        sprintf(target, "target=%s", pConfig->rocm_config.rocm_install_path);

        // add rocm post-install if required
        if (pConfig->post_config.rocm_post)
        {
            sprintf(postrocm, "%s", "postrocm");
        }
    }

    // Check if amdgpu is being installed
    if (pConfig->driver_config.install_driver)
    {
        strcat(installcomps, " amdgpu");

        // Check if for amdgpu start
        if (pConfig->driver_config.start_driver)
        {
            strcat(installcomps, " amdgpu-start");
        }
    }

    // Set post-install gpu access args
    if (pConfig->post_config.current_user_grp)
    {
        sprintf(postinstall, "%s", "gpu-access=user");
    }
    else if (pConfig->post_config.all_user_grp)
    {
        sprintf(postinstall, "%s", "gpu-access=all");
    }

    sprintf(cmdArgs, "%s %s %s %s", installcomps, target, postrocm, postinstall);
}

void set_install_state(MENU_DATA *pMenuData, OFFLINE_INSTALL_CONFIG *pConfig)
{
    ROCM_MENU_CONFIG *pRocmConfig = &pConfig->rocm_config;
    bool installable = false;

    if (pConfig->driver_config.install_driver)
    {
        installable = true;
    }

    if (pRocmConfig->install_rocm)
    {
        if (pRocmConfig->is_rocm_path_valid && (pRocmConfig->rocm_pkg_path_index < 0) )
        {
            installable = true;
        }
        else
        {
            installable = false;   
        }
    }

    // update the install menu item if install is valid
    if (installable)
    {
        menu_set_item_select(pMenuData, MAIN_MENU_ITEM_INSTALL_INDEX, true);
    }
    else
    {
        menu_set_item_select(pMenuData, MAIN_MENU_ITEM_INSTALL_INDEX, false);
    }

    pConfig->install = installable;
}

int main()
{
    WINDOW *menuWindow;
    MENU *pMenu;
    ITEM *pCurrentItem;

    OFFLINE_INSTALL_CONFIG offlineConfig = {0};
    MENU_DATA menuMain = {0};

    int c;
    int done = 0;
    int status = -1;

    // Read VERSION file BEFORE ncurses init for error reporting
    if (read_version_file(&offlineConfig) != 0)
    {
        fprintf(stderr, "Failed to read VERSION file. Exiting.\n");
        return 1;
    }

    // Get distro/kernel info on the system
    get_os_info(&offlineConfig);

    // Initialize config pointers for submenus to access version info
    offlineConfig.rocm_config.pConfig = &offlineConfig;
    offlineConfig.driver_config.pConfig = &offlineConfig;

    // Set TERMINFO path for static ncurses compatibility across distros
    // Static ncurses built on AlmaLinux 8.10 looks for terminfo in /usr/share/terminfo
    // Ubuntu 22.04 has it in /lib/terminfo, Ubuntu 24.04 has it in /usr/share/terminfo
    if (!getenv("TERMINFO")) {
        // Check common terminfo locations and set TERMINFO accordingly
        struct stat st;
        if (stat("/lib/terminfo", &st) == 0 && S_ISDIR(st.st_mode)) {
            // Ubuntu 22.04 and older store terminfo in /lib
            setenv("TERMINFO", "/lib/terminfo", 1);
        } else if (stat("/usr/share/terminfo", &st) == 0 && S_ISDIR(st.st_mode)) {
            // Ubuntu 24.04, RHEL, and most distros use /usr/share/terminfo
            setenv("TERMINFO", "/usr/share/terminfo", 1);
        }
    }

    // Set TERM fallback for better compatibility
    char *term = getenv("TERM");
    if (!term || !*term) {
        setenv("TERM", "linux", 1);
    }

    // Initialize ncurses
    initscr();
    start_color();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);
    
    // init colors
    init_pair(1, COLOR_RED, COLOR_BLACK);
    init_pair(2, COLOR_CYAN, COLOR_BLACK);

    init_pair(3, COLOR_WHITE, COLOR_BLACK);
    init_pair(4, COLOR_GREEN, COLOR_BLACK);
    init_pair(5, COLOR_BLUE, COLOR_BLACK);

    init_pair(6, COLOR_WHITE, COLOR_RED);
    init_pair(7, COLOR_BLACK, COLOR_GREEN);
    init_pair(8, COLOR_BLACK, COLOR_WHITE);

    init_pair(9, COLOR_MAGENTA, COLOR_BLACK);
    init_pair(10, COLOR_YELLOW, COLOR_BLACK);
    init_pair(11, COLOR_BLACK, COLOR_MAGENTA);
    init_pair(12, COLOR_WHITE, COLOR_BLUE);
    init_pair(13, COLOR_WHITE, COLOR_YELLOW);

    // Create the window to be associated with the menu
    menuWindow = newwin(WIN_NUM_LINES, WIN_WIDTH_COLS, WIN_START_Y, WIN_START_X);
    keypad(menuWindow, TRUE);

    // Create the main menu
    create_menu(&menuMain, menuWindow, &mainMenuProperties, &mainMenuItems, &offlineConfig);

    pMenu = menuMain.pMenu;

    // set items to non-selectable for the main menu
    set_menu_grey(pMenu, COLOR_PAIR(5));
    menu_set_item_select(&menuMain, MAIN_MENU_ITEM_INSTALL_INDEX, false);  // install

    // Create the various main option menus
    create_pre_menu_window(menuWindow, &offlineConfig);
    create_rocm_menu_window(menuWindow, &offlineConfig);
    create_driver_menu_window(menuWindow, &offlineConfig);
    create_post_menu_window(menuWindow, &offlineConfig);

    // Draw the main menu
    main_menu_draw(&menuMain, &offlineConfig);

    // Post the main menu
    post_menu(pMenu);
    print_menu_item_selection(&menuMain, MENU_SEL_START_Y, MENU_SEL_START_X);
    wrefresh(menuWindow);
    
    while( done == 0 )
    {
        c = wgetch(menuWindow);
        switch(c)
        {	
            case KEY_RESIZE: // Terminal window resize
                if (should_window_be_resized(menuWindow, WIN_NUM_LINES,WIN_WIDTH_COLS))
                {
                    reset_window_before_resizing(&menuMain);
                    
                    main_menu_draw(&menuMain, &offlineConfig);
                    post_menu(pMenu);
                }
                break;

            case KEY_DOWN:
                menu_driver(pMenu, REQ_DOWN_ITEM);

                skip_menu_item_down_if_skippable(pMenu);

                break;

            case KEY_UP:
                menu_driver(pMenu, REQ_UP_ITEM);

                skip_menu_item_up_if_skippable(pMenu);

                break;

            case KEY_F(1):
                done = 1;
                break;

            case 10:    // Enter
                pCurrentItem = current_item(pMenu);
                unpost_menu(pMenu);

                if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_PRE_INDEX )
                {
                    do_pre_menu();
                }
                else if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_ROCM_INDEX )
                {
                    do_rocm_menu();
                    set_install_state(&menuMain, &offlineConfig);
                }
                else if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_DRIVER_INDEX )
                {
                    do_driver_menu();
                    set_install_state(&menuMain, &offlineConfig);
                }
                else if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_POST_INDEX )
                {
                    do_post_menu();
                }
                else if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_INSTALL_INDEX )
                {
                    if (offlineConfig.install)
                    {
                        done = 1;
                        status = 0;
                    }
                }
                else
                {
                    print_menu_err_msg(&menuMain, "Invalid item");
                }

                // return to the main menu
                main_menu_draw(&menuMain, &offlineConfig);
                post_menu(pMenu);

                break;
        }

        print_menu_item_selection(&menuMain, MENU_SEL_START_Y, MENU_SEL_START_X);
        wrefresh(menuWindow);
    } 

    destroy_menu(&menuMain);

    destroy_pre_menu_window();
    destroy_rocm_menu_window();
    destroy_post_menu_window();
    
    delwin(menuWindow);
    endwin();

    // call the creator script
    if (status == 0)
    {
        char cmd[LARGE_CHAR_SIZE];
        clear_str(cmd);

        char cmdArgs[DEFAULT_CHAR_SIZE];
        clear_str(cmdArgs);

        config_install(&offlineConfig, cmdArgs);

        sprintf(cmd, "./rocm-installer.sh %s", cmdArgs);
        printf("Running: %s\n", cmd);

        system(cmd);
    }

    return 0;

}


