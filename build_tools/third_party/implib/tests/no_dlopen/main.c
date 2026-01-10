/*
 * Copyright 2022 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include <dlfcn.h>

int main() {
  dlopen("libinterposed.so", RTLD_GLOBAL | RTLD_LAZY);
#ifdef SHLIB
  extern void shlib_test();
  shlib_test();
#else
  extern void test();
  test();
#endif
  return 0;
}
