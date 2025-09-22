/*
 * Copyright 2022 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <assert.h>

#include "interposed.h"

void test() {
  int x, y;
  // Slow path
  x = foo(25, 0.5),
  y = bar(11, 22, 33);
  printf("Results: %x %x\n", x, y);
  // Fast path
  x = foo(35, 0.25);
  y = bar(44, 55, 66);
  printf("Results: %x %x\n", x, y);
}

static void *handle;

void *my_load_library(const char *name) {
  if (!handle)
    handle = dlopen(name, RTLD_LOCAL | RTLD_LAZY);
  return handle;
}

int main() {
  extern void _libinterposed_so_tramp_reset(void);

  for (int i = 0; i < 2; ++i) {
    test();
    dlclose(handle);
    handle = 0;
    assert(dlopen("libinterposed.so", RTLD_NOLOAD) == 0);
    _libinterposed_so_tramp_reset();
  }

  return 0;
}
