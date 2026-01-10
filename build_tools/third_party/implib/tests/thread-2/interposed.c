/*
 * Copyright 2025 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include <stdio.h>
#include "interposed.h"

static int glob;

__attribute__((constructor))
static void init() {
  glob = foo0(11);  // 21
}

__attribute__((visibility("default")))
int foo0(int x) {
  return 0 + foo1(x);
}

__attribute__((visibility("default")))
int foo1(int x) {
  return 1 + foo2(x);
}

__attribute__((visibility("default")))
int foo2(int x) {
  return 2 + foo3(x);
}

__attribute__((visibility("default")))
int foo3(int x) {
  return 3 + foo4(x);
}

__attribute__((visibility("default")))
int foo4(int x) {
  return 4 + x;
}

int bar() {
  return glob;
}
