/*
 * Copyright 2019-2022 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#ifndef INTERPOSED_H
#define INTERPOSED_H

#define VIS __attribute__((visibility("default")))
class VIS A {
public:
  virtual void foo(int x, int y);
};

class VIS B : public A {
public:
  virtual void foo(int x, int y);
};

#endif
