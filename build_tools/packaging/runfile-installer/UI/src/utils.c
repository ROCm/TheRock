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
#include "utils.h"

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <dirent.h>
#include <libgen.h>
#include <ctype.h>

int calculate_text_height(char *desc, int width)
{
    int desc_length = strlen(desc);
    int rows_needed = ceil(desc_length/width);
    return rows_needed + 1;
}

int get_char_array_size(char *array[])
{
    // array must have last element defined as NULL
    int size = 0;
    if (array != NULL) 
    {
        while (array[size] != NULL) {
            size++;
        }

    }
    return size;
}

bool is_field_empty(char *text)
{
    int count = strlen(text);
    int i = 0;
   
    while (i < count)
    {
        // When user deletes char(s) from form field, it simply replaces the deleted
        // char w/ the ' ' (space) char. So strlen still returns an incorrect value,
        // even when the string is empty. This is a fix for that.
        if (text[i] != ' ')
        {
            return false;
        }

        i++;
    }

    return true;
}

int get_field_length(char *text, int field_width)
{
    char temp[DEFAULT_CHAR_SIZE];
    int i;

    strcpy(temp, text);

    for (i = 0; i < field_width; i++)
    {
        if ((temp[i] == ' ') || (temp[i] == '\0'))
            break;
    }

    return i;
}

void field_trim(char *src, char *dst, int max)
{
    int field_len = get_field_length(src, max);
    
    memset(dst, '\0', DEFAULT_CHAR_SIZE);
    strncpy(dst, src, (max-3));
    if (field_len > (max -3) )
    {
        strcat(dst, "...");
    }
}

int check_url(char *url) 
{
    char command[DEFAULT_CHAR_SIZE];

    sprintf(command, "wget -q --spider %s", url);
    
    return system(command);
}

int check_path_exists(char *path, int max)
{
    int ret = 0;
    struct stat buffer;

    remove_end_spaces(path, max);

    ret = stat(path, &buffer);

    return ret;
}

void remove_slash(char *str)
{
    int len = strlen(str);

    for (int i = 0; i < len; i++)
    {
        if (str[i] == '/')
        {
            str[i] = '-';
        }
    }
}

void remove_end_spaces(char *str, int max)
{
    int field_len = get_field_length(str, max);

    char temp[DEFAULT_CHAR_SIZE];

    memset(temp, '\0', DEFAULT_CHAR_SIZE);
    strncpy(temp, str, field_len);
    
    strcpy(str, temp);
}

int clear_str(char *str)
{
    if (NULL == str)
    {
        return -1;
    }

    memset(str, '\0', strlen(str));

    return 0;
}

bool is_dir_exist(char *path)
{
    DIR* dir = opendir(path);
    if (dir) 
    {
        /* Directory exists. */
        closedir(dir);
        return true;
    } 
    else 
    {
        return false;
    }

}

int is_rocm_pkg_installed(DISTRO_TYPE distroType) 
{
    int status;
    
    if (distroType == eDISTRO_TYPE_DEB)
    {
        status = system("dpkg -l rocm-core > /dev/null 2>&1");
    }
    else
    {
        status = system("rpm -q rocm-core > /dev/null 2>&1");
    }

    if (status == -1) 
    {
        perror("system");
        return 0;
    }

    // return status child process/exit status
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

// Helper function to extract the version number from a path
int extract_version(const char *path, char *version) 
{
    const char *start = strstr(path, "rocm-");
    if (start) 
    {
        start += strlen("rocm-"); // Move past "rocm-"
        const char *end = start;

        while (*end && (isdigit(*end) || *end == '.')) 
        {
            end++;
        }
        strncpy(version, start, end - start);
        version[end - start] = '\0';
        return 0;
    }

    return -1;
}

// Helper function to compare paths by version
int compare_versions(const void *a, const void *b) 
{
    char version_a[SMALL_CHAR_SIZE], version_b[SMALL_CHAR_SIZE];

    extract_version(*(const char **)a, version_a);
    extract_version(*(const char **)b, version_b);

    return strcmp(version_a, version_b);
}

int find_rocm_installed(char *target, char fpaths[MAX_PATHS][LARGE_CHAR_SIZE], int *pCount)
{
    FILE *fp;
    char path[LARGE_CHAR_SIZE];
    char search_path[DEFAULT_CHAR_SIZE];
    char rocm_depth[SMALL_CHAR_SIZE];
    int status;

    if (pCount == NULL)
    {
        return -1;
    }

    // Command to find the "version" file in directories containing "opt/rocm/.info"
    // and skip any directory paths containing "rocm-installer"
    char command[LARGE_CHAR_SIZE];

    // Search only from the target path if provided
    if (target)
    {
        if (strcmp(target, "/") == 0)
        {
            sprintf(search_path, "/opt");
        }
        else
        {
            sprintf(search_path, "%s", target);
        }

        sprintf(rocm_depth, "-maxdepth 4");
    }
    else
    {
        sprintf(search_path, "/");
        rocm_depth[0] = '\0';
    }

    sprintf(command, "find %s %s -type f -path '*/rocm-*/.info/version' ! -path '*/rocm-installer/component-rocm/*' -print 2>/dev/null", search_path, rocm_depth);

    // Open a pipe to the command
    fp = popen(command, "r");
    if (fp == NULL)
    {
        perror("popen failed");
        exit(1);
    }

    // Initialize found_count
    *pCount = 0;

    // Temporary array to store pointers to paths
    char *temp_paths[MAX_PATHS];

    // Read the output of the command
    while (fgets(path, sizeof(path), fp) != NULL)
    {
        // Remove the newline character from the path
        path[strcspn(path, "\n")] = '\0';

        // Remove the substring ".info/version" from the path
        char *pos = strstr(path, ".info/version");
        if (pos != NULL)
        {
            *pos = '\0'; // Terminate the string at the start of ".info/version"
        }

        // Save the path to the temporary array
        if (*pCount < MAX_PATHS)
        {
            temp_paths[*pCount] = strdup(path); // Allocate memory for the path
            (*pCount)++;
        }
        else
        {
            fprintf(stderr, "Warning: Maximum number of paths reached. Some paths may not be stored.\n");
            break;
        }
    }

    // Close the command and get the exit status
    status = pclose(fp);
    if (status == -1)
    {
        perror("pclose");
        return -1;
    }

    if (*pCount == 0)
    {
        return -1;
    }

    // Sort the paths by version
    qsort(temp_paths, *pCount, sizeof(char *), compare_versions);

    // Copy the sorted paths to the fpaths array
    for (int i = 0; i < *pCount; i++)
    {
        strncpy(fpaths[i], temp_paths[i], LARGE_CHAR_SIZE - 1);
        fpaths[i][LARGE_CHAR_SIZE - 1] = '\0';
        free(temp_paths[i]); // Free the allocated memory
    }

    return 0;
}

int get_rocm_version_str_from_path(char *rocm_loc, char *rocm_core_ver)
{
    char rocm_ver[SMALL_CHAR_SIZE];
    int ret = -1;

    // get the base directory name from the current location
    char *rocm_dir = basename(rocm_loc);
    
    // extract "rocm-"
    char *rocm_str = strstr(rocm_dir, "rocm-");
    if (NULL != rocm_str)
    {
        // extract the version
        strcpy(rocm_ver, rocm_str + strlen("rocm-"));

        // convert to a core version number
        int x, y, z;
        if (sscanf(rocm_ver, "%d.%d.%d", &x, &y, &z) == 3)
        {
            sprintf(rocm_core_ver, "%d%02d%02d", x, y, z);

            ret = 0;
        }
    }

    return ret;
}

int get_rocm_core_pkg(DISTRO_TYPE distroType, char *rocm_core_out, size_t out_size)
{
    FILE *fp;
    char path[LARGE_CHAR_SIZE];
    int status;

    // Open the command for reading
    if (distroType == eDISTRO_TYPE_DEB)
    {
        fp = popen("dpkg -l | grep rocm-core", "r");
    }
    else
    {
        fp = popen("rpm -q rocm-core", "r");
    }

    if (fp == NULL) 
    {
        perror("popen failed");
        return -1;
    }

    // Read the output a line at a time and store it in rocm_core_out
    rocm_core_out[0] = '\0'; // Initialize rocm_core_out to an empty string
    while (fgets(path, sizeof(path), fp) != NULL) 
    {
        strncat(rocm_core_out, path, out_size - strlen(rocm_core_out) - 1);
    }

    // Close the command and get the exit status
    status = pclose(fp);
    if (status == -1) 
    {
        perror("pclose");
        return -1;
    }

    // Check if the command was successful
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) 
    {
        // the command maybe success, check that dkms provide output
        if (strlen(rocm_core_out) > 0)
        {
            return 0;
        }
        return -1;
    } 
    else 
    {
        return -1;
    }
}

int is_loc_opt_rocm(char *rocm_loc)
{
    char *rocm_base = "/opt/rocm";

    // Check if the rocm loc is in /opt/rocm
    if (strncmp(rocm_loc, rocm_base, strlen(rocm_base)) == 0)
    {
        char next_chr = rocm_loc[strlen(rocm_base)];
        //if (next_chr == '\0' || next_chr == '-' || next_chr == '/')
        if (next_chr == '-')
        {
            return 1;
        }
    }

    return 0;
}

int is_dkms_pkg_installed(DISTRO_TYPE distroType)
{
    int status;
    
    if (distroType == eDISTRO_TYPE_DEB)
    {
        status = system("apt list --installed 2>&1 | grep dkms  > /dev/null 2>&1");
    }
    else
    {
        status = system("rpm -q dkms > /dev/null 2>&1");
    }

    if (status == -1) 
    {
        perror("system");
        return 0;
    }

    // return status child process/exit status
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int is_amdgpu_dkms_pkg_installed(DISTRO_TYPE distroType) 
{
    int status;
    
    if (distroType == eDISTRO_TYPE_DEB)
    {
        status = system("apt list --installed 2>&1 | grep amdgpu-dkms  > /dev/null 2>&1");
    }
    else
    {
        status = system("rpm -q amdgpu-dkms > /dev/null 2>&1");
    }

    if (status == -1) 
    {
        perror("system");
        return 0;
    }

    // return status child process/exit status
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int check_dkms_status(char *dkms_out, size_t dkms_out_size)
{
    FILE *fp;
    char path[LARGE_CHAR_SIZE];
    int status;

    // Open the command for reading
    fp = popen("dkms status 2>&1", "r");
    if (fp == NULL) 
    {
        perror("popen failed");
        return -1;
    }

    // Read the output a line at a time and store it in dkms_out
    dkms_out[0] = '\0'; // Initialize dkms_out to an empty string
    while (fgets(path, sizeof(path), fp) != NULL) 
    {
        strncat(dkms_out, path, dkms_out_size - strlen(dkms_out) - 1);
    }

    // Close the command and get the exit status
    status = pclose(fp);
    if (status == -1) 
    {
        perror("pclose");
        return -1;
    }

    // Check if the command was successful
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) 
    {
        // the command maybe success, check that dkms provide output
        if (strlen(dkms_out) > 0)
        {
            char *p = strstr(dkms_out, ", x86_64:");
            if (p != NULL)
            {
                *p = '\0';
            }

            return 0;
        }
        return -1;
    } 
    else 
    {
        return -1;
    }
}

int execute_cmd(const char *script, const char *args, WINDOW *pWin)
{
    int ret = 1;

    char cmd[LARGE_CHAR_SIZE];
    clear_str(cmd);

    if (pWin)
    {
        // exit ncurses for the command
        def_prog_mode();
	    endwin();

        sprintf(cmd, "%s %s", script, args);
    }
    else
    {
        sprintf(cmd, "%s %s > /dev/null 2>&1", script, args);
    }

    // execute the command
    ret = system(cmd);
    if ((WEXITSTATUS(ret)) == 0)
    {
        ret = 0;
    }

    if (pWin)
    {
        // return to ncurses
        reset_prog_mode();
        wrefresh(pWin);
    }

    return ret;
}
