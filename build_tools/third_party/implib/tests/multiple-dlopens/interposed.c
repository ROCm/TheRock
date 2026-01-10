/*
 * Copyright 2022 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include <stdio.h>
#include "interposed.h"

__attribute__((visibility("default")))
int foo(int x, float y) {
  printf("Calling foo from libtest: %d %g\n", x, y);
  return 0xf00;
}

__attribute__((visibility("default")))
int bar(int x, int y, int z) {
  printf("Calling bar from libtest: %d %d %d\n", x, y, z);
  return 0xba7;
}
