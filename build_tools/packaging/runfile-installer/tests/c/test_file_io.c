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
 * test_file_io.c
 *
 * Unit tests for read_file_for_items() in menu_data.c.
 *
 * read_file_for_items() reads a newline-delimited text file into a
 * pre-allocated 2D char array, stripping trailing newlines.  It accepts
 * the filepath as a parameter, so tests supply mkstemp() temporary files —
 * no production code changes required.
 *
 * Key constraints (from menu_data.h):
 *   MAX_MENU_ITEMS     = 50   — maximum number of lines the function accepts
 *   MAX_MENU_ITEM_NAME = 79   — buffer size; fgets reads at most 78 chars
 *
 * Line length behaviour:
 *   <= 77 chars + '\n' : read cleanly in one fgets call
 *      78 chars + '\n' : fgets fills the buffer without consuming '\n';
 *                        the fix drains the '\n' and skips empty buffers —
 *                        count is correctly 1 (verified below)
 *      79+ chars       : line is split across multiple fgets calls; the fix
 *                        drains the remainder to keep the stream aligned
 *
 * Build and run:
 *   cd runfile-installer/tests
 *   cmake -B build -DCMAKE_BUILD_TYPE=Debug .
 *   cmake --build build
 *   ctest --test-dir build --output-on-failure -R test_file_io
 */

/* cmocka requires these three headers in exactly this order */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdio.h>
#include <stdlib.h>   /* mkstemp */
#include <string.h>
#include <unistd.h>   /* write, close, unlink */

#include "menu_data.h"


/* ─── Helper ─────────────────────────────────────────────────────────────────
 * Creates a mkstemp() temp file containing content, then closes it.
 * path_template must be a mutable char array ending in "XXXXXX".
 * The caller is responsible for unlink()-ing the file after the test.
 */
static void write_temp_file(char *path_template, const char *content)
{
    int fd = mkstemp(path_template);
    assert_int_not_equal(-1, fd);

    if (content && strlen(content) > 0) {
        ssize_t len = (ssize_t)strlen(content);
        assert_int_equal(len, write(fd, content, (size_t)len));
    }

    close(fd);
}

/* Variant for tests that need a file with N identical short lines. */
static void write_temp_file_n_lines(char *path_template, int n)
{
    int fd = mkstemp(path_template);
    assert_int_not_equal(-1, fd);

    FILE *f = fdopen(fd, "w");
    assert_non_null(f);

    for (int i = 0; i < n; i++) {
        fprintf(f, "item%d\n", i);
    }

    fclose(f);  /* also closes fd */
}


/* ═══════════════════════════════════════════════════════════════════════════
 * read_file_for_items
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_read_file_for_items_reads_content(void **state)
{
    /* Normal use: realistic content matching the actual device/component files.
     * Verifies count, line content, and that trailing newlines are stripped. */
    char path[] = "/tmp/test_read_file_io_XXXXXX";
    write_temp_file(path,
        "    MI325X/MI300X/MI300A (gfx94x)\n"
        "    RX 7900 XTX/XT/GRE (gfx110x)\n"
        "core-sdk\n");

    char lines[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {{0}};
    int count = read_file_for_items(path, lines);

    assert_int_equal(3, count);
    assert_string_equal("    MI325X/MI300X/MI300A (gfx94x)", lines[0]);
    assert_string_equal("    RX 7900 XTX/XT/GRE (gfx110x)", lines[1]);
    assert_string_equal("core-sdk", lines[2]);

    /* Verify newlines are stripped — no '\n' in any stored line */
    for (int i = 0; i < count; i++) {
        assert_null(strchr(lines[i], '\n'));
    }

    unlink(path);
}

static void test_read_file_for_items_missing_file(void **state)
{
    char lines[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {{0}};
    assert_int_equal(-1, read_file_for_items("/tmp/does_not_exist_rocm_test", lines));
}

static void test_read_file_for_items_empty_file(void **state)
{
    char path[] = "/tmp/test_read_file_io_XXXXXX";
    write_temp_file(path, "");

    char lines[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {{0}};
    assert_int_equal(0, read_file_for_items(path, lines));

    unlink(path);
}

static void test_read_file_for_items_blank_lines_skipped(void **state)
{
    /* Blank lines in the file are skipped — they do not become empty menu
     * entries.  A file with content "first\n\nthird\n" yields 2 items. */
    char path[] = "/tmp/test_read_file_io_XXXXXX";
    write_temp_file(path, "first\n\nthird\n");

    char lines[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {{0}};
    int count = read_file_for_items(path, lines);

    assert_int_equal(2, count);
    assert_string_equal("first", lines[0]);
    assert_string_equal("third", lines[1]);

    unlink(path);
}

static void test_read_file_for_items_safe_boundary_line(void **state)
{
    /* A line of exactly 77 chars + '\n' (78 bytes total) is the longest line
     * that fgets reads cleanly in one call with the 79-byte buffer.
     * fgets reads all 77 chars plus the '\n', strips the '\n', stores 77 chars. */
    char path[] = "/tmp/test_read_file_io_XXXXXX";

    char content[80] = {0};        /* 77 'a' + '\n' + '\0' */
    memset(content, 'a', 77);
    content[77] = '\n';
    write_temp_file(path, content);

    char lines[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {{0}};
    int count = read_file_for_items(path, lines);

    assert_int_equal(1, count);

    char expected[78] = {0};
    memset(expected, 'a', 77);
    assert_string_equal(expected, lines[0]);

    unlink(path);
}

static void test_read_file_for_items_long_line_no_ghost(void **state)
{
    /* A line of 78 chars + '\n' previously triggered a ghost empty entry.
     *
     * fgets(buffer, 79, file) reads at most 78 chars.  For a 78-char line,
     * it fills the buffer without consuming the '\n'.  The fix detects this
     * condition and drains the '\n' from the stream before the next iteration.
     * An empty-line guard then skips any empty buffer that slips through.
     * The result is count=1 — no ghost entry. */
    char path[] = "/tmp/test_read_file_io_XXXXXX";

    char content[81] = {0};        /* 78 'a' + '\n' + '\0' */
    memset(content, 'a', 78);
    content[78] = '\n';
    write_temp_file(path, content);

    char lines[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {{0}};
    int count = read_file_for_items(path, lines);

    assert_int_equal(1, count);

    char expected[79] = {0};
    memset(expected, 'a', 78);
    assert_string_equal(expected, lines[0]);

    unlink(path);
}

static void test_read_file_for_items_at_max_capacity(void **state)
{
    /* A file with exactly MAX_MENU_ITEMS (50) lines returns 50. */
    char path[] = "/tmp/test_read_file_io_XXXXXX";
    write_temp_file_n_lines(path, MAX_MENU_ITEMS);

    char lines[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {{0}};
    int count = read_file_for_items(path, lines);

    assert_int_equal(MAX_MENU_ITEMS, count);
    assert_string_equal("item0",                          lines[0]);
    assert_string_equal("item49", lines[MAX_MENU_ITEMS - 1]);

    unlink(path);
}

static void test_read_file_for_items_exceeds_max_capacity(void **state)
{
    /* A file with MAX_MENU_ITEMS+1 (51) lines returns -1.
     * The overflow is detected after the 51st line is read from the file
     * but before it is stored — no out-of-bounds write occurs. */
    char path[] = "/tmp/test_read_file_io_XXXXXX";
    write_temp_file_n_lines(path, MAX_MENU_ITEMS + 1);

    char lines[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME] = {{0}};
    assert_int_equal(-1, read_file_for_items(path, lines));

    unlink(path);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_read_file_for_items_reads_content),
        cmocka_unit_test(test_read_file_for_items_missing_file),
        cmocka_unit_test(test_read_file_for_items_empty_file),
        cmocka_unit_test(test_read_file_for_items_blank_lines_skipped),
        cmocka_unit_test(test_read_file_for_items_safe_boundary_line),
        cmocka_unit_test(test_read_file_for_items_long_line_no_ghost),
        cmocka_unit_test(test_read_file_for_items_at_max_capacity),
        cmocka_unit_test(test_read_file_for_items_exceeds_max_capacity),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
