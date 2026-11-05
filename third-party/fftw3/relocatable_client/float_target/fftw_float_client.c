// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdio.h>

#include <fftw3.h>

int main(void) {
  fftwf_complex *in = fftwf_alloc_complex(4);
  fftwf_complex *out = fftwf_alloc_complex(4);
  fftwf_plan plan;
  int ok;

  if (!in || !out) {
    return 1;
  }
  in[0][0] = 1.0f;
  in[0][1] = 0.0f;
  in[1][0] = 0.0f;
  in[1][1] = 0.0f;
  in[2][0] = 0.0f;
  in[2][1] = 0.0f;
  in[3][0] = 0.0f;
  in[3][1] = 0.0f;
  plan = fftwf_plan_dft_1d(4, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  if (!plan) {
    return 1;
  }
  fftwf_execute(plan);
  ok = out[0][0] > 0.9f && out[0][0] < 1.1f;
  fftwf_destroy_plan(plan);
  fftwf_free(in);
  fftwf_free(out);
  if (!ok) {
    fprintf(stderr, "float plan failed\n");
    return 1;
  }
  printf("fftw-3.3.10 float ok\n");
  return 0;
}
