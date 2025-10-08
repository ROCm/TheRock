/*
 * Copyright 2022 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

__attribute__((visibility("default")))
void shlib_test() {
  extern void test();
  test();
}
