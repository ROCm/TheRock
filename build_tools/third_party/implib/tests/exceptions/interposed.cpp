/*
 * Copyright 2019-2020 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include "interposed.h"

__attribute__((visibility("default")))
void foo(int x, int y) {
  throw err(x + y);
}
