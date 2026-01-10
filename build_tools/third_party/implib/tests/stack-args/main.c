/*
 * Copyright 2024 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include "interposed.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static void usage(char *progname) {
  fprintf(stderr, "Usage: %s float/int\n", progname);
  exit(1);
}

int main(int argc, char *argv[]) {
  if (argc != 2)
    usage(argv[0]);

  if (0 == strcmp(argv[1], "int")) {
    foo(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  } else if (0 == strcmp(argv[1], "float")) {
    bar(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  } else {
    usage(argv[0]);
  }

  return 0;
}
