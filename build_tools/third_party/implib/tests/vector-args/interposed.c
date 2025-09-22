/*
 * Copyright 2024 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include "interposed.h"

__attribute__((visibility("default")))
vector_type foo(vector_type x) {
  return 3 *x;
}

extern vector_type dummy(vector_type x0, vector_type x1, vector_type x2, vector_type x3, vector_type x4, vector_type x5, vector_type x6, vector_type x7);

__attribute__((constructor)) void touch_vector_regs() {
  vector_type zero = {0};
  dummy(zero, zero, zero, zero, zero, zero, zero, zero);
}
