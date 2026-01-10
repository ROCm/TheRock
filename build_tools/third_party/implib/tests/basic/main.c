/*
 * Copyright 2017-2022 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

int main() {
#ifdef SHLIB
  extern void shlib_test();
  shlib_test();
#else
  extern void test();
  test();
#endif
  return 0;
}
