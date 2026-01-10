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

__attribute__((visibility("default")))
int foo(int x) {
  return 2 * x;
}

__attribute__((visibility("default")))
int bar() {
  return 123;
}
