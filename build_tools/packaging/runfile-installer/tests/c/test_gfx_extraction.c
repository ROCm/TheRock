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
 * test_gfx_extraction.c
 *
 * Unit tests for extract_gfx_code() in rocm_menu.c.
 *
 * extract_gfx_code() is a pure string function — it takes a menu item name
 * such as "    MI325X/MI300X/MI300A (gfx94x)" and returns a pointer into the
 * input string at the start of the gfx code (e.g. "gfx94x)"), or NULL if the
 * format is not recognised.  It does not touch global state or ncurses.
 *
 * The function is not declared in rocm_menu.h (it is an internal helper), so
 * we forward-declare it here.
 *
 * Build and run:
 *   cd runfile-installer/tests
 *   cmake -B build -DCMAKE_BUILD_TYPE=Debug .
 *   cmake --build build
 *   ctest --test-dir build --output-on-failure -R test_gfx_extraction
 */

/* cmocka requires these three headers in exactly this order */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <string.h>

/* extract_gfx_code is defined in rocm_menu.c but not declared in rocm_menu.h */
const char *extract_gfx_code(const char *item_name);


/* ─── Helper ─────────────────────────────────────────────────────────────────
 * extract_gfx_code returns a pointer INTO the input string at the start of
 * the gfx code.  The code ends at the closing ')'.  This helper copies the
 * code into a NUL-terminated buffer so we can use assert_string_equal.
 */
static void gfx_to_str(const char *code_ptr, char *buf, size_t buf_size)
{
    if (code_ptr == NULL) {
        buf[0] = '\0';
        return;
    }

    const char *end = strchr(code_ptr, ')');
    if (end == NULL) {
        buf[0] = '\0';
        return;
    }

    size_t len = (size_t)(end - code_ptr);
    if (len >= buf_size) {
        len = buf_size - 1;
    }

    strncpy(buf, code_ptr, len);
    buf[len] = '\0';
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Happy-path tests — real item names from rocm_device_items
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_gfx_instinct_family(void **state)
{
    const char *code = extract_gfx_code("    MI325X/MI300X/MI300A (gfx94x)");
    assert_non_null(code);
    char buf[32] = {0};
    gfx_to_str(code, buf, sizeof(buf));
    assert_string_equal("gfx94x", buf);
}

static void test_gfx_mi355x(void **state)
{
    const char *code = extract_gfx_code("    MI355X/MI350 (gfx950)");
    assert_non_null(code);
    char buf[32] = {0};
    gfx_to_str(code, buf, sizeof(buf));
    assert_string_equal("gfx950", buf);
}

static void test_gfx_numeric_family_suffix(void **state)
{
    /* Ryzen AI — 4-digit gfx code */
    const char *code = extract_gfx_code("    Max PRO+ 395/Max PRO 390/385/380 (gfx1151)");
    assert_non_null(code);
    char buf[32] = {0};
    gfx_to_str(code, buf, sizeof(buf));
    assert_string_equal("gfx1151", buf);
}

static void test_gfx_wildcard_rdna3(void **state)
{
    /* gfxNNNx covers a whole GPU family */
    const char *code = extract_gfx_code("    RX 7900 XTX/XT/GRE (gfx110x)");
    assert_non_null(code);
    char buf[32] = {0};
    gfx_to_str(code, buf, sizeof(buf));
    assert_string_equal("gfx110x", buf);
}

static void test_gfx_wildcard_rdna4(void **state)
{
    const char *code = extract_gfx_code("    RX 9070 XT/GRE/9070 (gfx120x)");
    assert_non_null(code);
    char buf[32] = {0};
    gfx_to_str(code, buf, sizeof(buf));
    assert_string_equal("gfx120x", buf);
}

static void test_gfx_workstation(void **state)
{
    const char *code = extract_gfx_code("    W7900/W7800/W7700/V710 (gfx110x)");
    assert_non_null(code);
    char buf[32] = {0};
    gfx_to_str(code, buf, sizeof(buf));
    assert_string_equal("gfx110x", buf);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Error / boundary tests — inputs that must return NULL
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_gfx_no_parenthesis_returns_null(void **state)
{
    assert_null(extract_gfx_code("MI300X"));
}

static void test_gfx_paren_without_gfx_prefix_returns_null(void **state)
{
    /* '(' present but content does not start with "gfx" */
    assert_null(extract_gfx_code("Radeon (some-code)"));
}

static void test_gfx_unclosed_paren_returns_null(void **state)
{
    /* Function requires a closing ')' to validate the format */
    assert_null(extract_gfx_code("MI300X (gfx94x"));
}

static void test_gfx_empty_string_returns_null(void **state)
{
    assert_null(extract_gfx_code(""));
}

static void test_gfx_family_header_returns_null(void **state)
{
    /* Menu item that is a section label, not a selectable GPU */
    assert_null(extract_gfx_code("Instinct:"));
}

static void test_gfx_skippable_item_returns_null(void **state)
{
    /* The blank separator used to skip menu items has no gfx code */
    assert_null(extract_gfx_code(" "));
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* happy path */
        cmocka_unit_test(test_gfx_instinct_family),
        cmocka_unit_test(test_gfx_mi355x),
        cmocka_unit_test(test_gfx_numeric_family_suffix),
        cmocka_unit_test(test_gfx_wildcard_rdna3),
        cmocka_unit_test(test_gfx_wildcard_rdna4),
        cmocka_unit_test(test_gfx_workstation),

        /* error / boundary */
        cmocka_unit_test(test_gfx_no_parenthesis_returns_null),
        cmocka_unit_test(test_gfx_paren_without_gfx_prefix_returns_null),
        cmocka_unit_test(test_gfx_unclosed_paren_returns_null),
        cmocka_unit_test(test_gfx_empty_string_returns_null),
        cmocka_unit_test(test_gfx_family_header_returns_null),
        cmocka_unit_test(test_gfx_skippable_item_returns_null),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
