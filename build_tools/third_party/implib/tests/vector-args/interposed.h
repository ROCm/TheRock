/*
 * Copyright 2024 Yury Gribov
 *
 * The MIT License (MIT)
 * 
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#ifndef INTERPOSED_H
#define INTERPOSED_H

// Determine number of 32-bit ints in native vector type
// for each supported platform
#if defined __AVX512H__ /* ZMM regs */
# define VECTOR_BITSIZE 512
#elif defined __AVX__ /* YMM regs */
# define VECTOR_BITSIZE 256
#elif defined __SSE__ /* XMM regs */ \
    || defined __aarch64__ /* NEON regs */
# define VECTOR_BITSIZE 128
#elif defined __MMX__ /* MMX regs */ \
    || defined __arm__  /* NEON regs */
# define VECTOR_BITSIZE 64
#elif defined __i386__
  // x86 has no vectors by default
# define VECTOR_BITSIZE 32
#else
# error "Unknown platform"
#endif

typedef int vector_type __attribute__((vector_size(VECTOR_BITSIZE / 8)));

extern vector_type foo(vector_type x);

#endif
