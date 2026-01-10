/*
 * Copyright 2025 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include <pthread.h>
#include <unistd.h>

#include "interposed.h"

#if defined __mips && __mips == 32
// For some reason pthread_create fails with EAGAIN
#define N 32
#else
#define N 128
#endif

static int args[N];
static pthread_t tids[N];
static pthread_barrier_t bar;

void *run(void *arg_) {
  int rc = pthread_barrier_wait(&bar);
  if (PTHREAD_BARRIER_SERIAL_THREAD != rc && 0 != rc)
    abort();

  int *arg = (int *)arg_;

  int (*foo)(int);
  switch(*arg % 10) {
    case 0:
      foo = foo0;
      break;
    case 1:
      foo = foo1;
      break;
    case 2:
      foo = foo2;
      break;
    case 3:
      foo = foo3;
      break;
    case 4:
      foo = foo4;
      break;
    case 5:
      foo = foo5;
      break;
    case 6:
      foo = foo6;
      break;
    case 7:
      foo = foo7;
      break;
    case 8:
      foo = foo8;
      break;
    case 9:
      foo = foo9;
      break;
    default:
      abort();
  }

  *arg = foo(*arg);

  return 0;
}

int main() {
  int exp = 0;
  for (int i = 0; i < N; ++i) {
    args[i] = i;
    exp += 1 + i + (i % 10);
  }

  if (0 != pthread_barrier_init(&bar, 0, N))
    abort();

  for (int i = 0; i < N; ++i) {
    if (0 != pthread_create(&tids[i], 0, run, &args[i]))
      abort();
  }

  for (int i = 0; i < N; ++i) {
    if (0 != pthread_join(tids[i], 0))
      abort();
  }

  int res = 0;
  for (int i = 0; i < N; ++i)
    res += args[i];

  if (res != exp)
    printf("Result: %d (%d exp)\n", res, exp);
  else
    printf("Correct result\n");

  return 0;
}
