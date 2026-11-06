// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdio.h>

#include <fftw3.h>

int main(void) {
  fftw_complex *in = fftw_alloc_complex(4);
  fftw_complex *out = fftw_alloc_complex(4);
  fftw_plan plan;
  int ok;

  if (!in || !out) {
    return 1;
  }
  in[0][0] = 1.0;
  in[0][1] = 0.0;
  in[1][0] = 0.0;
  in[1][1] = 0.0;
  in[2][0] = 0.0;
  in[2][1] = 0.0;
  in[3][0] = 0.0;
  in[3][1] = 0.0;
  plan = fftw_plan_dft_1d(4, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  if (!plan) {
    return 1;
  }
  fftw_execute(plan);
  ok = out[0][0] > 0.9 && out[0][0] < 1.1;
  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);
  if (!ok) {
    fprintf(stderr, "double plan failed\n");
    return 1;
  }
  printf("fftw-3.3.10 double ok\n");
  return 0;
}
