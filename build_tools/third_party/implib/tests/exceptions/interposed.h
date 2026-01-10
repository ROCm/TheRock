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

#include <stdexcept>

struct err : public std::runtime_error {
  int cookie;
  err(int cookie) : std::runtime_error(""), cookie(cookie) {}
};

extern void foo(int x, int y);

#endif
