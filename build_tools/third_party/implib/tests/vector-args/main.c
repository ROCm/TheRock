/*
 * Copyright 2024 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include <stdio.h>

#include "interposed.h"

int main() {
  vector_type x;
  int n = sizeof(x) / sizeof(x[0]);

  int i;
  for (i = 0; i < n; ++i)
    x[i] = i;

  vector_type res = foo(x);
  for (i = 0; i < n; ++i) {
    int ref = 3 * x[i];
    if (res[i] != ref) {
      printf("%d: %d (expecting %d)\n", i, res[i], ref);
      return 1;
    }
  }
  return 0;
}
