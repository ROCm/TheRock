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

/*
 * test_stubs.c
 *
 * Provides definitions for the five global config pointers that are normally
 * owned and initialised by rocm_ui.c.  In the production binary, rocm_ui.c
 * stack-allocates an OFFLINE_INSTALL_CONFIG and points all five globals at it.
 *
 * For unit tests, rocm_ui.c is excluded from the build (it defines main()).
 * These stubs satisfy the linker for any compiled module that declares the
 * pointers as extern.
 *
 * The tests in this suite never call functions that dereference these
 * globals, so NULL is a safe sentinel value.  Future tests that exercise
 * functions which DO dereference these globals must initialise them in
 * their cmocka setup() callbacks before calling production code.
 */

#include "config.h"

OFFLINE_INSTALL_CONFIG *g_pConfig       = NULL;
ROCM_MENU_CONFIG       *g_pRocmConfig   = NULL;
DRIVER_MENU_CONFIG     *g_pDriverConfig = NULL;  /* needed when driver_menu.c is added */
POST_MENU_CONFIG       *g_pPostConfig   = NULL;  /* needed when post_menu.c is added   */
PRE_MENU_CONFIG        *g_pPreConfig    = NULL;  /* needed when pre_menu.c is added     */
