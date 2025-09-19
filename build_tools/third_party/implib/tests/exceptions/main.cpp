/*
 * Copyright 2019-2022 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include <stdio.h>

#include "interposed.h"

int main() {
  try {
    foo(1, 2);
  } catch (const err &e) {
    printf("Caught exception: %d\n", e.cookie);
  }
  return 0;
}
