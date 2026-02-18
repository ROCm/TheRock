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
#ifndef _UTILIS_H
#define _UTILIS_H

#include <stdbool.h>
#include "install_types.h"


#define TOGGLE_BIT(val, bitIndx) val ^= (1 << bitIndx)
#define TOGGLE_FALSE(val, bitIndx) val &= ~(1 << bitIndx) 


int calculate_text_height(char *desc, int width);
int get_char_array_size(char *array[]);
bool is_field_empty(char *text);
int get_field_length(char *text, int field_width);
void field_trim(char *src, char *dst, int max);

int check_url(char *url);
int check_path_exists(char *path, int max);

void remove_slash(char *str);
void remove_end_spaces(char *str, int max);
int clear_str(char *str);

bool is_dir_exist(char *path);

int is_rocm_pkg_installed(DISTRO_TYPE distroType);
int find_rocm_installed(char *target, char fpaths[MAX_PATHS][LARGE_CHAR_SIZE], int *pCount);
int get_rocm_version_str_from_path(char *rocm_loc, char *rocm_core_ver);
int get_rocm_core_pkg(DISTRO_TYPE distroType, char *rocm_core_out, size_t out_size);
int is_loc_opt_rocm(char *rocm_loc);

int is_dkms_pkg_installed(DISTRO_TYPE distroType);
int is_amdgpu_dkms_pkg_installed(DISTRO_TYPE distroType);
int check_dkms_status(char *dkms_out, size_t dkms_out_size);

int execute_cmd(const char *script, const char *args, WINDOW *pWin);

#endif // _UTILIS_H
