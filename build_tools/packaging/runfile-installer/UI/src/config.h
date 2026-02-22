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
#ifndef _CONFIG_H
#define _CONFIG_H

#include "install_types.h"


/* Pre Install Menu Configuration ****************************************************************/

// Structure for all driver menu configuration settings
typedef struct _PRE_MENU_CONFIG
{
    bool rocm_deps;
    bool driver_deps;
}PRE_MENU_CONFIG;

/* ROCm Menu Configuration **********************************************************************/

// Structure for all rocm main/sub menu configuration settings
typedef struct _ROCM_MENU_CONFIG
{
    bool install_rocm;
    char rocm_install_path[DEFAULT_CHAR_SIZE];

    bool is_rocm_path_valid;
    bool is_rocm_installed;
    int rocm_pkg_path_index;
    int rocm_runfile_path_index;

    INTSTALL_TYPE rocm_install_type;
    char rocm_paths[MAX_PATHS][LARGE_CHAR_SIZE];
    int  rocm_count;

    // Pointer to parent config for accessing version info
    struct _OFFLINE_INSTALL_CONFIG *pConfig;
}ROCM_MENU_CONFIG;

/* Driver Menu Configuration ********************************************************************/

// Structure for all driver menu configuration settings
typedef struct _DRIVER_MENU_CONFIG
{
    bool install_driver;
    bool start_driver;

    bool           is_driver_installed;
    INTSTALL_TYPE  driver_install_type;
    char           dkms_status[LARGE_CHAR_SIZE];

    // Pointer to parent config for accessing version info
    struct _OFFLINE_INSTALL_CONFIG *pConfig;
}DRIVER_MENU_CONFIG;

/* Post Install Menu Configuration **************************************************************/

// Structure for all post install menu configuration settings
typedef struct _POST_MENU_CONFIG
{
    bool current_user_grp;
    bool all_user_grp;
    bool rocm_post;
}POST_MENU_CONFIG;


/* Global Configuration ************************************************************************/

// Global Installer create configuration
typedef struct _OFFLINE_INSTALL_CONFIG
{
    // Version info read from VERSION file at runtime
    char installerVersion[64];      // Line 1: INSTALLER_VERSION (e.g., "2.0.0")
    char rocmVersion[64];            // Line 2: ROCM_VER (e.g., "7.11.0")
    char buildTag[64];               // Line 3: BUILD_TAG (e.g., "1", "rc1", "nightly")
    char buildRunId[64];             // Line 4: BUILD_RUNID (e.g., "99999", "1")
    char buildPullTag[64];           // Line 5: BUILD_PULL_TAG (e.g., "20260219-22188089855")
    char amdgpuDkmsBuild[64];        // Line 6: AMDGPU_DKMS_BUILD_NUM (e.g., "6.18.4-2286447")

    // Runtime distro detection
    char distroName[64];
    char distroID[64];
    char distroVersion[64];

    DISTRO_TYPE distroType;
    char kernelVersion[128];

    PRE_MENU_CONFIG     pre_config;
    ROCM_MENU_CONFIG    rocm_config;
    DRIVER_MENU_CONFIG  driver_config;
    POST_MENU_CONFIG    post_config;

    // global configuration
    bool                install;
}OFFLINE_INSTALL_CONFIG;


#endif // _CONFIG_H

