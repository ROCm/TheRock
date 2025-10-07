/*
 * Copyright 2022 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include <stdio.h>

#define VIS __attribute__((visibility("default")))
class VIS A {
public:
  virtual void foo(int x, int y);
};

void A::foo(int x, int y) {
  printf("A::foo: %d\n", x + y);
}

void foo(int x) {
  printf("foo: %d\n", x);
}
