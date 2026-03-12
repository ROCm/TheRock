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

// Global config pointers
OFFLINE_INSTALL_CONFIG *g_pConfig = NULL;
ROCM_MENU_CONFIG *g_pRocmConfig = NULL;
DRIVER_MENU_CONFIG *g_pDriverConfig = NULL;
POST_MENU_CONFIG *g_pPostConfig = NULL;
PRE_MENU_CONFIG *g_pPreConfig = NULL;


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

void get_os_release_value(char *key, char *value, size_t max_size)
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
                    strncpy(value, p + 2, max_size - 1);
                    value[max_size - 1] = '\0';
                    if (strlen(value) >= 2)
                        value[strlen(value) - 2] = '\0';
                }
                else
                {
                    strncpy(value, p + 1, max_size - 1);
                    value[max_size - 1] = '\0';
                    if (strlen(value) >= 1)
                        value[strlen(value) - 1] = '\0';
                }
                break;
            }
        }
    }
    fclose(fp);

    free(line);
}

int get_os_info()
{
    uint32_t i;
    struct utsname unameData;

    if (uname(&unameData) < 0)
    {
        return -1;
    }

    strcpy(g_pConfig->kernelVersion, unameData.release);

    get_os_release_value("PRETTY_NAME", g_pConfig->distroName,    sizeof(g_pConfig->distroName));
    get_os_release_value("ID",           g_pConfig->distroID,      sizeof(g_pConfig->distroID));
    get_os_release_value("VERSION_ID",   g_pConfig->distroVersion, sizeof(g_pConfig->distroVersion));

    char *debList[] = {"ubuntu", "debian"};
    char *elList[]  = {"rhel", "ol", "rocky", "centos", "almalinux", "amzn"};
    char *sleList[] = {"sles"};

    for (i = 0; i < ARRAY_SIZE(debList); i++)
    {
        if (strstr(g_pConfig->distroID, debList[i]) != NULL)
        {
            g_pConfig->distroType = eDISTRO_TYPE_DEB;
        }
    }

    for (i = 0; i < ARRAY_SIZE(elList); i++)
    {
        if (strstr(g_pConfig->distroID, elList[i]) != NULL)
        {
            g_pConfig->distroType = eDISTRO_TYPE_EL;
        }
    }

    for (i = 0; i < ARRAY_SIZE(sleList); i++)
    {
        if (strstr(g_pConfig->distroID, sleList[i]) != NULL)
        {
            g_pConfig->distroType = eDISTRO_TYPE_SLE;
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
 *   Line 5: BUILD_TAG_INFO (e.g., "20260219-22188089855")
 *   Line 6: AMDGPU_DKMS_BUILD_NUM (e.g., "6.18.4-2286447")
 *
 * Returns: 0 on success, -1 on failure
 */
int read_version_file()
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
                strncpy(g_pConfig->installerVersion, line, sizeof(g_pConfig->installerVersion) - 1);
                g_pConfig->installerVersion[sizeof(g_pConfig->installerVersion) - 1] = '\0';
                break;
            case 1:
                strncpy(g_pConfig->rocmVersion, line, sizeof(g_pConfig->rocmVersion) - 1);
                g_pConfig->rocmVersion[sizeof(g_pConfig->rocmVersion) - 1] = '\0';
                break;
            case 2:
                strncpy(g_pConfig->buildTag, line, sizeof(g_pConfig->buildTag) - 1);
                g_pConfig->buildTag[sizeof(g_pConfig->buildTag) - 1] = '\0';
                break;
            case 3:
                strncpy(g_pConfig->buildRunId, line, sizeof(g_pConfig->buildRunId) - 1);
                g_pConfig->buildRunId[sizeof(g_pConfig->buildRunId) - 1] = '\0';
                break;
            case 4:
                strncpy(g_pConfig->buildTagInfo, line, sizeof(g_pConfig->buildTagInfo) - 1);
                g_pConfig->buildTagInfo[sizeof(g_pConfig->buildTagInfo) - 1] = '\0';
                break;
            case 5:
                strncpy(g_pConfig->amdgpuDkmsBuild, line, sizeof(g_pConfig->amdgpuDkmsBuild) - 1);
                g_pConfig->amdgpuDkmsBuild[sizeof(g_pConfig->amdgpuDkmsBuild) - 1] = '\0';
                break;
        }
        line_num++;
    }

    fclose(fp);
    free(line);

    return 0;
}

void main_menu_install_draw(MENU_DATA *pMenuData)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    char drawName[DEFAULT_CHAR_SIZE];
    int start_row = 14;

    // Draw the current install configuration
    if (g_pRocmConfig->install_rocm || (g_pDriverConfig->install_driver))
    {
        wattron(pMenuWindow, WHITE | A_ITALIC | A_UNDERLINE | A_BOLD);
        mvwprintw(pMenuWindow, start_row, 50, "Install Configuration");
        wattroff(pMenuWindow, WHITE | A_ITALIC | A_UNDERLINE | A_BOLD);
        start_row +=2;
    }

    // ROCm
    if (g_pRocmConfig->install_rocm)
    {
        wattron(pMenuWindow, WHITE | A_ITALIC );
        mvwprintw(pMenuWindow, start_row, 52, "ROCm Install:");
        wattroff(pMenuWindow, WHITE | A_ITALIC );
        menu_info_draw_bool(pMenuData, start_row, 66, g_pRocmConfig->install_rocm);

        // ROCm device
        if (strlen(g_pRocmConfig->rocm_device) > 0)
        {
            wattron(pMenuWindow, GREEN | A_BOLD);
            mvwprintw(pMenuWindow, start_row+1, 58, "Device: %s", g_pRocmConfig->rocm_device);
            wattroff(pMenuWindow, GREEN | A_BOLD);
        }
        else
        {
            wattron(pMenuWindow, RED | A_BOLD);
            mvwprintw(pMenuWindow, start_row+1, 58, "Device: not selected.");
            wattroff(pMenuWindow, RED | A_BOLD);
        }

        // ROCm components
        if (strlen(g_pRocmConfig->rocm_components) > 0)
        {
            clear_str(drawName);
            field_trim(g_pRocmConfig->rocm_components, drawName, 17);

            wattron(pMenuWindow, GREEN | A_BOLD);
            mvwprintw(pMenuWindow, start_row+2, 54, "Components: %s", drawName);
            wattroff(pMenuWindow, GREEN | A_BOLD);
        }
        else
        {
            wattron(pMenuWindow, RED | A_BOLD);
            mvwprintw(pMenuWindow, start_row+2, 54, "Components: not selected.");
            wattroff(pMenuWindow, RED | A_BOLD);
        }

        // ROCm install path
        if (g_pRocmConfig->is_rocm_path_valid)
        {
            clear_str(drawName);
            field_trim(g_pRocmConfig->rocm_install_path, drawName, 17);

            wattron(pMenuWindow, GREEN | A_BOLD);
            mvwprintw(pMenuWindow, start_row+3, 52, "Install Path: %s", drawName);
            wattroff(pMenuWindow, GREEN | A_BOLD);
        }
        else
        {
            wattron(pMenuWindow, RED | A_BOLD);
            mvwprintw(pMenuWindow, start_row+3, 52, "Install Path: Invalid.");
            wattroff(pMenuWindow, RED | A_BOLD);
        }

        start_row +=5;
    }

    // Driver
    if (g_pDriverConfig->install_driver)
    {
        wattron(pMenuWindow, WHITE | A_ITALIC );
        mvwprintw(pMenuWindow, start_row, 50, "Driver Install: %d", g_pDriverConfig->install_driver);
        wattroff(pMenuWindow, WHITE | A_ITALIC );
        menu_info_draw_bool(pMenuData, start_row, 66, g_pDriverConfig->install_driver);
    }
}

void main_menu_draw(MENU_DATA *pMenuData)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    wclear(pMenuWindow);

    char installer_build[DEFAULT_CHAR_SIZE];
    sprintf(installer_build, "%s", g_pConfig->distroName);

    // resizes pMenuWindow and subwindow that displays menu items to its original
    // size in case user resized terminal window
    resize_and_reposition_window_and_subwindow(pMenuData, WIN_NUM_LINES, WIN_WIDTH_COLS);

    print_menu_title(pMenuData, MENU_TITLE_Y, MENU_TITLE_X, WIN_WIDTH_COLS, "ROCm Runfile Installer", CYAN);
    print_menu_item_title(pMenuData, 3, 2, installer_build, MAGENTA);

    print_menu_control_msg(pMenuData);

    print_version(pMenuData);

    // Display a warning if ROCm is installed on the system
    if (g_pRocmConfig->install_rocm && g_pRocmConfig->is_rocm_installed && g_pRocmConfig->is_rocm_path_valid)
    {
        if (g_pRocmConfig->rocm_install_type == eINSTALL_PACKAGE)
        {
            print_menu_err_msg(pMenuData, "ROCm %s package manager installed for target.", g_pConfig->rocmVersion);
        }
        else
        {
            print_menu_warning_msg(pMenuData, "ROCm %s installed for target", g_pConfig->rocmVersion);
        }
    }
    else
    {
        clear_menu_msg(pMenuData);
    }

    main_menu_install_draw(pMenuData);

    box(pMenuData->pMenuWindow, 0, 0);
}

/* Append a space-separated token to buf, skipping empty tokens.
 * Using strncat rather than snprintf avoids -Wformat-truncation: the
 * theoretical maximum of all tokens combined exceeds DEFAULT_CHAR_SIZE,
 * so a single snprintf call trips the warning under -Werror. */
static void append_token(char *buf, size_t buf_size, const char *token)
{
    if (!token || !token[0]) return;
    size_t current = strlen(buf);
    if (current > 0 && current < buf_size - 1) {
        buf[current++] = ' ';
        buf[current]   = '\0';
    }
    strncat(buf, token, buf_size - strlen(buf) - 1);
}

void config_install(char *cmdArgs)
{
    char installcomps[SMALL_CHAR_SIZE];
    char target[LARGE_CHAR_SIZE];
    char gfx[LARGE_CHAR_SIZE];
    char compo[LARGE_CHAR_SIZE];
    char postrocm[SMALL_CHAR_SIZE];
    char postinstall[SMALL_CHAR_SIZE];

    clear_str(installcomps);
    clear_str(target);
    clear_str(gfx);
    clear_str(compo);
    clear_str(postrocm);
    clear_str(postinstall);

    // Check if rocm is being installed
    if (g_pRocmConfig->install_rocm)
    {
        sprintf(installcomps, "rocm");
        sprintf(target, "target=%s", g_pRocmConfig->rocm_install_path);

        // Add gfx device selection if specified
        if (strlen(g_pRocmConfig->rocm_device) > 0)
        {
            sprintf(gfx, "gfx=%s", g_pRocmConfig->rocm_device);
        }

        // Add component selection if specified
        if (strlen(g_pRocmConfig->rocm_components) > 0)
        {
            sprintf(compo, "compo=%s", g_pRocmConfig->rocm_components);
        }

        // add nopostrocm to disable post rocm install - default is enabled in the installer
        if (!g_pPostConfig->rocm_post)
        {
            sprintf(postrocm, "%s", "nopostrocm");
        }
    }

    // Check if amdgpu is being installed
    if (g_pDriverConfig->install_driver)
    {
        strcat(installcomps, " amdgpu");

        // Check if for amdgpu start
        if (g_pDriverConfig->start_driver)
        {
            strcat(installcomps, " amdgpu-start");
        }
    }

    // Set post-install gpu access args
    if (g_pPostConfig->current_user_grp)
    {
        sprintf(postinstall, "%s", "gpu-access=user");
    }
    else if (g_pPostConfig->all_user_grp)
    {
        sprintf(postinstall, "%s", "gpu-access=all");
    }

    cmdArgs[0] = '\0';
    append_token(cmdArgs, DEFAULT_CHAR_SIZE, installcomps);
    append_token(cmdArgs, DEFAULT_CHAR_SIZE, target);
    append_token(cmdArgs, DEFAULT_CHAR_SIZE, gfx);
    append_token(cmdArgs, DEFAULT_CHAR_SIZE, compo);
    append_token(cmdArgs, DEFAULT_CHAR_SIZE, postrocm);
    append_token(cmdArgs, DEFAULT_CHAR_SIZE, postinstall);
}

void set_install_state(MENU_DATA *pMenuData)
{
    bool installable = false;

    if (g_pDriverConfig->install_driver)
    {
        installable = true;
    }

    if (g_pRocmConfig->install_rocm)
    {
        installable = true;
        installable &= g_pRocmConfig->is_rocm_path_valid;
        installable &= (g_pRocmConfig->rocm_pkg_path_index < 0);
        installable &= (strlen(g_pRocmConfig->rocm_device) > 0);
        installable &= (strlen(g_pRocmConfig->rocm_components) > 0);
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

    g_pConfig->install = installable;
}

static void handle_key_enter(ITEM *pCurrentItem, MENU_DATA *pMenuMain, int *done, int *status)
{
    MENU *pMenu = pMenuMain->pMenu;

    unpost_menu(pMenu);

    if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_PRE_INDEX )
    {
        do_pre_menu();
    }
    else if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_ROCM_INDEX )
    {
        do_rocm_menu();
        set_install_state(pMenuMain);
    }
    else if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_DRIVER_INDEX )
    {
        do_driver_menu();
        set_install_state(pMenuMain);
    }
    else if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_POST_INDEX )
    {
        do_post_menu();
    }
    else if ( item_index(pCurrentItem) == MAIN_MENU_ITEM_INSTALL_INDEX )
    {
        if (g_pConfig->install)
        {
            *done = 1;
            *status = 0;
        }
    }
    else
    {
        print_menu_err_msg(pMenuMain, "Invalid item");
    }

    // return to the main menu
    main_menu_draw(pMenuMain);
    post_menu(pMenu);
}

int main()
{
    WINDOW *menuWindow;
    MENU *pMenu;
    ITEM *pCurrentItem;

    static OFFLINE_INSTALL_CONFIG offlineConfig = {0};
    MENU_DATA menuMain = {0};

    int c;
    int done = 0;
    int status = -1;

    // Initialize global config pointers
    g_pConfig       = &offlineConfig;
    g_pRocmConfig   = &offlineConfig.rocm_config;
    g_pDriverConfig = &offlineConfig.driver_config;
    g_pPostConfig   = &offlineConfig.post_config;
    g_pPreConfig    = &offlineConfig.pre_config;

    // Read VERSION file BEFORE ncurses init for error reporting
    if (read_version_file() != 0)
    {
        fprintf(stderr, "Failed to read VERSION file. Exiting.\n");
        return 1;
    }

    // Get distro/kernel info on the system
    if (get_os_info() != 0)
    {
        fprintf(stderr, "ERROR: Failed to detect OS information. Exiting.\n");
        return 1;
    }

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
    
    // Single colors (foreground on black background)
    init_pair(1, COLOR_RED, COLOR_BLACK);
    init_pair(2, COLOR_CYAN, COLOR_BLACK);
    init_pair(3, COLOR_WHITE, COLOR_BLACK);
    init_pair(4, COLOR_GREEN, COLOR_BLACK);
    init_pair(5, COLOR_BLUE, COLOR_BLACK);
    init_pair(6, COLOR_MAGENTA, COLOR_BLACK);
    init_pair(7, COLOR_YELLOW, COLOR_BLACK);

    // White foreground on colored backgrounds
    init_pair(8, COLOR_WHITE, COLOR_RED);
    init_pair(9, COLOR_WHITE, COLOR_BLUE);
    init_pair(10, COLOR_WHITE, COLOR_YELLOW);

    // Black foreground on colored backgrounds
    init_pair(11, COLOR_BLACK, COLOR_GREEN);
    init_pair(12, COLOR_BLACK, COLOR_WHITE);
    init_pair(13, COLOR_BLACK, COLOR_MAGENTA);

    // Create the window to be associated with the menu
    menuWindow = newwin(WIN_NUM_LINES, WIN_WIDTH_COLS, WIN_START_Y, WIN_START_X);
    keypad(menuWindow, TRUE);

    // Create the main menu
    create_menu(&menuMain, menuWindow, &mainMenuProperties, &mainMenuItems, &offlineConfig);

    pMenu = menuMain.pMenu;

    // set items to non-selectable for the main menu
    set_menu_grey(pMenu, BLUE);
    menu_set_item_select(&menuMain, MAIN_MENU_ITEM_INSTALL_INDEX, false);  // install

    // Create the various main option menus
    create_pre_menu_window(menuWindow);
    create_rocm_menu_window(menuWindow);
    create_driver_menu_window(menuWindow);
    create_post_menu_window(menuWindow);

    // Draw the main menu
    main_menu_draw(&menuMain);

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

                    main_menu_draw(&menuMain);
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
                handle_key_enter(pCurrentItem, &menuMain, &done, &status);
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

        // Defence-in-depth: re-validate the install path immediately before
        // building the shell command. The install button is already gated by
        // is_rocm_path_valid, so this should never fire in normal flow.
        if (g_pRocmConfig->install_rocm &&
            !validate_install_path(g_pRocmConfig->rocm_install_path))
        {
            fprintf(stderr, "ERROR: Install path contains invalid characters. Aborting.\n");
            endwin();
            return 1;
        }

        config_install(cmdArgs);

        sprintf(cmd, "./rocm-installer.sh %s", cmdArgs);
        printf("Running: %s\n", cmd);

        system(cmd);
    }

    return 0;

}


