/*
 * Copyright 2019-2020 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include "interposed.h"

#include <stdio.h>

void A::foo(int x, int y) {
  printf("A::foo: %d\n", x + y);
}

void B::foo(int x, int y) {
  printf("B::foo: %d\n", x - y);
}
