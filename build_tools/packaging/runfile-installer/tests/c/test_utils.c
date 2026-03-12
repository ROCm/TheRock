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
 * test_utils.c
 *
 * Unit tests for the pure-logic functions in utils.c.  None of these tests
 * call system(), popen(), or ncurses — they exercise string/path utilities
 * that can be validated with straightforward input/output assertions.
 *
 * Coverage — pure-logic functions (no I/O or system calls):
 *
 *   Tested here:
 *     extract_version           (internal — forward-declared)
 *     compare_versions          (internal — forward-declared)
 *     is_field_empty
 *     get_field_length
 *     field_trim
 *     calculate_text_height
 *     get_char_array_size
 *     remove_slash
 *     remove_end_spaces
 *     clear_str
 *     get_rocm_version_str_from_path
 *     is_loc_opt_rocm
 *
 *   Not testable without mocking (require system(), popen(), or ncurses):
 *     exit_error, check_url, check_path_exists, is_dir_exist,
 *     is_rocm_pkg_installed, find_rocm_installed, get_rocm_core_pkg,
 *     is_dkms_pkg_installed, is_amdgpu_dkms_pkg_installed,
 *     check_dkms_status, execute_cmd
 *
 * Build and run:
 *   cd runfile-installer/tests
 *   cmake -B build -DCMAKE_BUILD_TYPE=Debug .
 *   cmake --build build
 *   ctest --test-dir build --output-on-failure -R test_utils
 */

/* cmocka requires these three headers in exactly this order */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <string.h>

#include "utils.h"
#include "install_types.h"

/* extract_version and compare_versions are internal helpers in utils.c —
 * not exported in utils.h.  Forward-declare them so we can test them directly.
 * They are worth testing because they have a subtle path-format assumption
 * (see test_extract_version_core_path_no_match below). */
int extract_version(const char *path, char *version);
int compare_versions(const void *a, const void *b);


/* ═══════════════════════════════════════════════════════════════════════════
 * extract_version
 *
 * Searches the full path string for the substring "core-" and extracts the
 * digit/dot sequence that immediately follows it.  Returns -1 if the anchor
 * is absent or if no digits follow it immediately.
 *
 * The paths produced by find_rocm_installed() look like:
 *   /opt/rocm/core-7.11.0
 * The "core-" anchor correctly matches these paths.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_extract_version_versioned_path(void **state)
{
    /* "/opt/rocm/core-7.11.0" — standard path from find_rocm_installed() */
    char ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(0, extract_version("/opt/rocm/core-7.11.0", ver));
    assert_string_equal("7.11.0", ver);
}

static void test_extract_version_embedded_in_longer_path(void **state)
{
    /* "core-" found mid-path; extraction stops at the first non-digit/non-dot */
    char ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(0, extract_version("/some/path/rocm/core-6.2.4/lib", ver));
    assert_string_equal("6.2.4", ver);
}

static void test_extract_version_core_path_extracts_version(void **state)
{
    /* "/opt/rocm/core-7.11.0" — the "core-" anchor is present and digits
     * follow immediately.  Extraction succeeds and returns the version. */
    char ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(0, extract_version("/opt/rocm/core-7.11.0", ver));
    assert_string_equal("7.11.0", ver);
}

static void test_extract_version_no_core_substring(void **state)
{
    /* No "core-" anchor present — extraction returns -1, buffer unchanged */
    char ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(-1, extract_version("/opt/something/lib-1.0.0", ver));
    assert_string_equal("", ver);
}

static void test_extract_version_empty_path(void **state)
{
    char ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(-1, extract_version("", ver));
    assert_string_equal("", ver);
}

static void test_extract_version_core_anchor_no_version_digits(void **state)
{
    /* "core-" anchor present but not followed by any digits.
     * The digit-scanning loop exits immediately (end == start).
     * The guard returns -1 — no false success. */
    char ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(-1, extract_version("/opt/rocm/core-", ver));
    assert_string_equal("", ver);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * compare_versions
 *
 * qsort comparator wrapping extract_version().
 * Only meaningful for paths that contain "rocm-" (see extract_version note).
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_compare_versions_less_than(void **state)
{
    const char *a = "/opt/rocm/core-7.10.0";
    const char *b = "/opt/rocm/core-7.11.0";
    assert_true(compare_versions(&a, &b) < 0);
}

static void test_compare_versions_greater_than(void **state)
{
    const char *a = "/opt/rocm/core-7.11.0";
    const char *b = "/opt/rocm/core-7.10.0";
    assert_true(compare_versions(&a, &b) > 0);
}

static void test_compare_versions_equal(void **state)
{
    const char *a = "/opt/rocm/core-7.11.0";
    const char *b = "/opt/rocm/core-7.11.0";
    assert_int_equal(0, compare_versions(&a, &b));
}

static void test_compare_versions_cross_major(void **state)
{
    /* Numeric comparison: 9 < 10, even though "10" < "9" lexicographically */
    const char *older = "/opt/rocm/core-9.0.0";
    const char *newer = "/opt/rocm/core-10.0.0";
    assert_true(compare_versions(&older, &newer) < 0);
}

static void test_compare_versions_core_paths_correct_ordering(void **state)
{
    /* Core paths from find_rocm_installed() are now correctly ordered.
     * extract_version() matches the "core-" anchor and compare_versions()
     * compares numerically — no UB, no uninitialised buffers. */
    const char *a = "/opt/rocm/core-7.10.0";
    const char *b = "/opt/rocm/core-7.11.0";
    assert_true(compare_versions(&a, &b) < 0);
    assert_true(compare_versions(&b, &a) > 0);
    assert_int_equal(0, compare_versions(&a, &a));
}


/* ═══════════════════════════════════════════════════════════════════════════
 * is_field_empty
 *
 * Returns true when a string consists entirely of space characters.  This
 * handles the ncurses form behaviour where deleting characters replaces them
 * with spaces rather than shrinking the buffer.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_is_field_empty_all_spaces(void **state)
{
    char text[] = "      ";
    assert_true(is_field_empty(text));
}

static void test_is_field_empty_single_space(void **state)
{
    char text[] = " ";
    assert_true(is_field_empty(text));
}

static void test_is_field_empty_empty_string(void **state)
{
    char text[] = "";
    assert_true(is_field_empty(text));
}

static void test_is_field_empty_has_non_space_char(void **state)
{
    char text[] = " /opt ";
    assert_false(is_field_empty(text));
}

static void test_is_field_empty_single_non_space_char(void **state)
{
    char text[] = "x";
    assert_false(is_field_empty(text));
}

static void test_is_field_empty_non_space_at_end(void **state)
{
    /* Spaces before a non-space should not be considered empty */
    char text[] = "    x";
    assert_false(is_field_empty(text));
}


/* ═══════════════════════════════════════════════════════════════════════════
 * get_field_length
 *
 * Returns the index of the first space or NUL character in text, capped at
 * field_width.  Effectively the length of the leading non-space content.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_get_field_length_no_spaces(void **state)
{
    /* No spaces → returns the full string length (capped at field_width) */
    char text[] = "/opt/rocm";
    assert_int_equal(9, get_field_length(text, 20));
}

static void test_get_field_length_trailing_spaces(void **state)
{
    /* Stops at the first space, returns 9 */
    char text[] = "/opt/rocm   ";
    assert_int_equal(9, get_field_length(text, 20));
}

static void test_get_field_length_leading_space_returns_zero(void **state)
{
    /* First character is a space → returns 0 immediately */
    char text[] = "   /opt";
    assert_int_equal(0, get_field_length(text, 20));
}

static void test_get_field_length_capped_by_field_width(void **state)
{
    /* Content is longer than field_width — returns field_width */
    char text[] = "abcdefghij";
    assert_int_equal(5, get_field_length(text, 5));
}

static void test_get_field_length_empty_string(void **state)
{
    char text[] = "";
    assert_int_equal(0, get_field_length(text, 20));
}


/* ═══════════════════════════════════════════════════════════════════════════
 * field_trim
 *
 * Copies up to (max-3) characters from src into dst, appending "..." if the
 * field content (up to first space) exceeds that limit.
 *
 * Important nuance: the copy uses strncpy(dst, src, max-3) — it always copies
 * max-3 bytes from src, which may include spaces within the string.  The
 * field_len check controls only whether "..." is appended.
 *
 * dst must be at least DEFAULT_CHAR_SIZE (256) bytes — the function calls
 * memset(dst, '\0', DEFAULT_CHAR_SIZE) unconditionally.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_field_trim_short_string_no_truncation(void **state)
{
    /* "hello" (5 chars) with max=10: field_len=5, 5 <= max-3=7, no "..." */
    char dst[DEFAULT_CHAR_SIZE] = {0};
    char src[] = "hello";
    field_trim(src, dst, 10);
    assert_string_equal("hello", dst);
}

static void test_field_trim_truncation_adds_ellipsis(void **state)
{
    /* "verylongword" (12 chars) with max=8: field_len=8 (capped), 8 > max-3=5,
     * strncpy copies 5 chars "veryl", then "..." appended → "veryl..." */
    char dst[DEFAULT_CHAR_SIZE] = {0};
    char src[] = "verylongword";
    field_trim(src, dst, 8);
    assert_string_equal("veryl...", dst);
}

static void test_field_trim_exact_boundary_no_ellipsis(void **state)
{
    /* "ab" (2 chars) with max=5: field_len=2, 2 <= max-3=2, no "..." */
    char dst[DEFAULT_CHAR_SIZE] = {0};
    char src[] = "ab";
    field_trim(src, dst, 5);
    assert_string_equal("ab", dst);
}

static void test_field_trim_one_over_boundary_adds_ellipsis(void **state)
{
    /* "abc" (3 chars) with max=5: field_len=3, 3 > max-3=2,
     * strncpy copies 2 chars "ab", then "..." appended → "ab..." */
    char dst[DEFAULT_CHAR_SIZE] = {0};
    char src[] = "abc";
    field_trim(src, dst, 5);
    assert_string_equal("ab...", dst);
}

static void test_field_trim_string_with_embedded_space(void **state)
{
    /* "hello world" with max=10: field_len=5 (stops at space), 5 <= max-3=7,
     * strncpy copies 7 chars "hello w" (includes the space).  No "...".
     * This is expected — field_len only governs "..." append, not copy count. */
    char dst[DEFAULT_CHAR_SIZE] = {0};
    char src[] = "hello world";
    field_trim(src, dst, 10);
    assert_string_equal("hello w", dst);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * calculate_text_height
 *
 * Returns the number of display rows needed for a string of given length,
 * using integer ceiling division: (desc_length + width - 1) / width + 1.
 * Correctly handles all lengths including non-multiples of width.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_calculate_text_height_empty_string(void **state)
{
    /* 0 / 10 = 0 (integer), ceil(0) = 0, return 1 */
    char text[] = "";
    assert_int_equal(1, calculate_text_height(text, 10));
}

static void test_calculate_text_height_shorter_than_width(void **state)
{
    /* (5 + 9) / 10 = 1, return 2 — string fits in one row, +1 for padding */
    char text[] = "hello";
    assert_int_equal(2, calculate_text_height(text, 10));
}

static void test_calculate_text_height_exact_multiple(void **state)
{
    /* 10 / 5 = 2 (exact), ceil(2) = 2, return 3 — correct for exact multiples */
    char text[] = "hello12345";
    assert_int_equal(3, calculate_text_height(text, 5));
}

static void test_calculate_text_height_non_exact_multiple(void **state)
{
    /* (11 + 4) / 5 = 3, return 4 — remainder correctly rounds up to a full row */
    char text[] = "hello world";
    assert_int_equal(4, calculate_text_height(text, 5));
}


/* ═══════════════════════════════════════════════════════════════════════════
 * get_char_array_size
 *
 * Counts elements in a NULL-terminated array of char pointers.
 * Returns 0 for a NULL array.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_get_char_array_size_null_array(void **state)
{
    assert_int_equal(0, get_char_array_size(NULL));
}

static void test_get_char_array_size_empty_array(void **state)
{
    char *arr[] = { NULL };
    assert_int_equal(0, get_char_array_size(arr));
}

static void test_get_char_array_size_one_element(void **state)
{
    char *arr[] = { "hello", NULL };
    assert_int_equal(1, get_char_array_size(arr));
}

static void test_get_char_array_size_three_elements(void **state)
{
    char *arr[] = { "core", "core-dev", "dev-tools", NULL };
    assert_int_equal(3, get_char_array_size(arr));
}


/* ═══════════════════════════════════════════════════════════════════════════
 * remove_slash
 *
 * Replaces every '/' character in str with '-'.  Modifies in-place.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_remove_slash_single_slash(void **state)
{
    char str[] = "rocm/7.11.0";
    remove_slash(str);
    assert_string_equal("rocm-7.11.0", str);
}

static void test_remove_slash_leading_slash(void **state)
{
    char str[] = "/opt/rocm";
    remove_slash(str);
    assert_string_equal("-opt-rocm", str);
}

static void test_remove_slash_no_slash_unchanged(void **state)
{
    char str[] = "rocm";
    remove_slash(str);
    assert_string_equal("rocm", str);
}

static void test_remove_slash_empty_string_unchanged(void **state)
{
    char str[] = "";
    remove_slash(str);
    assert_string_equal("", str);
}

static void test_remove_slash_consecutive_slashes(void **state)
{
    char str[] = "a//b";
    remove_slash(str);
    assert_string_equal("a--b", str);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * get_rocm_version_str_from_path
 *
 * Calls basename() on the path then searches for "rocm-" in the result.
 * Works for paths like /opt/rocm-7.11.0 (basename = "rocm-7.11.0").
 * Returns -1 for core paths like /opt/rocm/core-7.11.0
 * (basename = "core-7.11.0", which contains no "rocm-").
 *
 * Output format is "%d%02d%02d": major unpadded, minor and patch zero-padded
 * to 2 digits.  7.11.0 → "71100",  6.2.1 → "60201".
 *
 * Note: uses basename(3) from <libgen.h> which may modify its argument.
 * Always pass a mutable char array, never a string literal.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_rocm_version_str_from_versioned_path(void **state)
{
    char path[] = "/opt/rocm-7.11.0";
    char core_ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(0, get_rocm_version_str_from_path(path, core_ver));
    /* "%d%02d%02d": 7 + "11" + "00" = "71100" (major is NOT zero-padded) */
    assert_string_equal("71100", core_ver);
}

static void test_rocm_version_str_single_digit_minor(void **state)
{
    char path[] = "/opt/rocm-6.2.1";
    char core_ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(0, get_rocm_version_str_from_path(path, core_ver));
    /* 6 + "02" + "01" = "60201" */
    assert_string_equal("60201", core_ver);
}

static void test_rocm_version_str_core_path_returns_error(void **state)
{
    /* basename("core-7.11.0") contains no "rocm-" → returns -1, buffer unchanged */
    char path[] = "/opt/rocm/core-7.11.0";
    char core_ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(-1, get_rocm_version_str_from_path(path, core_ver));
    assert_string_equal("", core_ver);
}

static void test_rocm_version_str_unrelated_path_returns_error(void **state)
{
    char path[] = "/opt/other";
    char core_ver[SMALL_CHAR_SIZE] = {0};
    assert_int_equal(-1, get_rocm_version_str_from_path(path, core_ver));
    assert_string_equal("", core_ver);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * is_loc_opt_rocm
 *
 * Returns 1 if the path starts with "/opt/rocm-" (the versioned install
 * location), 0 otherwise.  The character immediately following "/opt/rocm"
 * must be '-' for the path to be recognised.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_is_loc_opt_rocm_versioned(void **state)
{
    char path[] = "/opt/rocm-7.11.0";
    assert_int_equal(1, is_loc_opt_rocm(path));
}

static void test_is_loc_opt_rocm_no_dash_suffix(void **state)
{
    /* The 10th character is '\0' — not '-' */
    char path[] = "/opt/rocm";
    assert_int_equal(0, is_loc_opt_rocm(path));
}

static void test_is_loc_opt_rocm_slash_separator(void **state)
{
    /* The 10th character is '/' — not '-' */
    char path[] = "/opt/rocm/core-7.11.0";
    assert_int_equal(0, is_loc_opt_rocm(path));
}

static void test_is_loc_opt_rocm_unrelated_path(void **state)
{
    char path[] = "/usr/local/rocm";
    assert_int_equal(0, is_loc_opt_rocm(path));
}

static void test_is_loc_opt_rocm_different_prefix(void **state)
{
    /* Path starts correctly but is in a different base directory */
    char path[] = "/home/user/rocm-7.11.0";
    assert_int_equal(0, is_loc_opt_rocm(path));
}


/* ═══════════════════════════════════════════════════════════════════════════
 * clear_str
 *
 * Zeroes the first strlen(str) bytes of str.  Returns 0 on success, -1 if
 * str is NULL.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_clear_str_zeros_all_bytes(void **state)
{
    char buf[] = "hello world";
    size_t len = strlen(buf);

    assert_int_equal(0, clear_str(buf));

    for (size_t i = 0; i < len; i++) {
        assert_int_equal('\0', (int)buf[i]);
    }
}

static void test_clear_str_empty_string_succeeds(void **state)
{
    /* strlen("") = 0, memset of 0 bytes → no-op, returns 0 */
    char buf[] = "";
    assert_int_equal(0, clear_str(buf));
}

static void test_clear_str_null_returns_error(void **state)
{
    assert_int_equal(-1, clear_str(NULL));
}


/* ═══════════════════════════════════════════════════════════════════════════
 * remove_end_spaces
 *
 * Trims trailing spaces from str in-place.  The max parameter is the field
 * width used to find where the content ends.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_remove_end_spaces_trims_trailing(void **state)
{
    char str[DEFAULT_CHAR_SIZE] = "/opt/rocm   ";
    remove_end_spaces(str, 12);
    assert_string_equal("/opt/rocm", str);
}

static void test_remove_end_spaces_no_trailing_spaces_unchanged(void **state)
{
    char str[DEFAULT_CHAR_SIZE] = "/opt/rocm";
    remove_end_spaces(str, 9);
    assert_string_equal("/opt/rocm", str);
}

static void test_remove_end_spaces_all_spaces_gives_empty(void **state)
{
    char str[DEFAULT_CHAR_SIZE] = "     ";
    remove_end_spaces(str, 5);
    assert_string_equal("", str);
}

static void test_remove_end_spaces_preserves_embedded_spaces(void **state)
{
    /* Only trailing spaces are stripped. Spaces within the content are
     * preserved, supporting paths such as "/home/my user/rocm". */
    char str[DEFAULT_CHAR_SIZE] = "/opt/my rocm   ";
    remove_end_spaces(str, 15);
    assert_string_equal("/opt/my rocm", str);
}

static void test_remove_end_spaces_path_with_embedded_space(void **state)
{
    /* A path containing a space with no trailing padding is returned unchanged,
     * confirming that embedded spaces are not stripped. */
    char str[DEFAULT_CHAR_SIZE] = "/home/my user";
    remove_end_spaces(str, 13);
    assert_string_equal("/home/my user", str);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* extract_version */
        cmocka_unit_test(test_extract_version_versioned_path),
        cmocka_unit_test(test_extract_version_embedded_in_longer_path),
        cmocka_unit_test(test_extract_version_core_path_extracts_version),
        cmocka_unit_test(test_extract_version_no_core_substring),
        cmocka_unit_test(test_extract_version_empty_path),
        cmocka_unit_test(test_extract_version_core_anchor_no_version_digits),

        /* compare_versions */
        cmocka_unit_test(test_compare_versions_less_than),
        cmocka_unit_test(test_compare_versions_greater_than),
        cmocka_unit_test(test_compare_versions_equal),
        cmocka_unit_test(test_compare_versions_cross_major),
        cmocka_unit_test(test_compare_versions_core_paths_correct_ordering),

        /* is_field_empty */
        cmocka_unit_test(test_is_field_empty_all_spaces),
        cmocka_unit_test(test_is_field_empty_single_space),
        cmocka_unit_test(test_is_field_empty_empty_string),
        cmocka_unit_test(test_is_field_empty_has_non_space_char),
        cmocka_unit_test(test_is_field_empty_single_non_space_char),
        cmocka_unit_test(test_is_field_empty_non_space_at_end),

        /* get_field_length */
        cmocka_unit_test(test_get_field_length_no_spaces),
        cmocka_unit_test(test_get_field_length_trailing_spaces),
        cmocka_unit_test(test_get_field_length_leading_space_returns_zero),
        cmocka_unit_test(test_get_field_length_capped_by_field_width),
        cmocka_unit_test(test_get_field_length_empty_string),

        /* field_trim */
        cmocka_unit_test(test_field_trim_short_string_no_truncation),
        cmocka_unit_test(test_field_trim_truncation_adds_ellipsis),
        cmocka_unit_test(test_field_trim_exact_boundary_no_ellipsis),
        cmocka_unit_test(test_field_trim_one_over_boundary_adds_ellipsis),
        cmocka_unit_test(test_field_trim_string_with_embedded_space),

        /* calculate_text_height */
        cmocka_unit_test(test_calculate_text_height_empty_string),
        cmocka_unit_test(test_calculate_text_height_shorter_than_width),
        cmocka_unit_test(test_calculate_text_height_exact_multiple),
        cmocka_unit_test(test_calculate_text_height_non_exact_multiple),

        /* get_char_array_size */
        cmocka_unit_test(test_get_char_array_size_null_array),
        cmocka_unit_test(test_get_char_array_size_empty_array),
        cmocka_unit_test(test_get_char_array_size_one_element),
        cmocka_unit_test(test_get_char_array_size_three_elements),

        /* remove_slash */
        cmocka_unit_test(test_remove_slash_single_slash),
        cmocka_unit_test(test_remove_slash_leading_slash),
        cmocka_unit_test(test_remove_slash_no_slash_unchanged),
        cmocka_unit_test(test_remove_slash_empty_string_unchanged),
        cmocka_unit_test(test_remove_slash_consecutive_slashes),

        /* get_rocm_version_str_from_path */
        cmocka_unit_test(test_rocm_version_str_from_versioned_path),
        cmocka_unit_test(test_rocm_version_str_single_digit_minor),
        cmocka_unit_test(test_rocm_version_str_core_path_returns_error),
        cmocka_unit_test(test_rocm_version_str_unrelated_path_returns_error),

        /* is_loc_opt_rocm */
        cmocka_unit_test(test_is_loc_opt_rocm_versioned),
        cmocka_unit_test(test_is_loc_opt_rocm_no_dash_suffix),
        cmocka_unit_test(test_is_loc_opt_rocm_slash_separator),
        cmocka_unit_test(test_is_loc_opt_rocm_unrelated_path),
        cmocka_unit_test(test_is_loc_opt_rocm_different_prefix),

        /* clear_str */
        cmocka_unit_test(test_clear_str_zeros_all_bytes),
        cmocka_unit_test(test_clear_str_empty_string_succeeds),
        cmocka_unit_test(test_clear_str_null_returns_error),

        /* remove_end_spaces */
        cmocka_unit_test(test_remove_end_spaces_trims_trailing),
        cmocka_unit_test(test_remove_end_spaces_no_trailing_spaces_unchanged),
        cmocka_unit_test(test_remove_end_spaces_all_spaces_gives_empty),
        cmocka_unit_test(test_remove_end_spaces_preserves_embedded_spaces),
        cmocka_unit_test(test_remove_end_spaces_path_with_embedded_space),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
