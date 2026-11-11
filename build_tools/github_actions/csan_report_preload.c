/* Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

/* The pinned CSAN runtime parses log_path without applying it. Initialize the
 * public sanitizer report API in every test process without rebuilding ROCm.
 */
#include <stdlib.h>

extern void __sanitizer_set_report_path(const char *path);

__attribute__((constructor)) static void configure_csan_report_path(void) {
  const char *prefix = getenv("CSAN_REPORT_PREFIX");
  if (prefix && *prefix)
    __sanitizer_set_report_path(prefix);
}
