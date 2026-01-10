/*
 * Copyright 2025 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "interposed.h"

int main() {
  printf("%d\n", foo(5) + bar());
  return 0;
}
