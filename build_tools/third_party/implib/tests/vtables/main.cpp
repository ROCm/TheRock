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

A *a, *b;
int flag = 1;

int main() {
  if (flag) {
    a = new A;
    b = new B;
  }

  a->foo(100, 200);
  b->foo(100, 200);

  return 0;
}
