// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <stdio.h>

#include <amd_comgr/amd_comgr.h>

int main(int argc, char **argv) {
  size_t major, minor;
  amd_comgr_get_version(&major, &minor);
  printf("amd_comgr version: %zu.%zu\n", major, minor);
  return 0;
}
