/*
 * Copyright 2020 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include "interposed.h"

int main() {
  foo(1, 2);
  return 0;
}
