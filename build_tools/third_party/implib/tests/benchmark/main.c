/*
 * Copyright 2025 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#define NITER 1000000000L

#ifdef BASELINE
__attribute((noipa)) void foo() {}
#else
#include "interposed.h"
#endif

int main() {
  for (long i = 0; i < NITER; ++i)
    foo();
  return 0;
}
