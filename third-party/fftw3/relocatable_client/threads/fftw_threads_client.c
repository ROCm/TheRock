// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdio.h>

#include <fftw3.h>

static int run_double(void) {
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
  return ok ? 0 : 1;
}

static int run_float(void) {
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
  return ok ? 0 : 1;
}

int main(void) {
  if (fftw_init_threads() == 0 || fftwf_init_threads() == 0) {
    fprintf(stderr, "fftw thread init failed\n");
    return 1;
  }
  fftw_plan_with_nthreads(2);
  fftwf_plan_with_nthreads(2);
  if (run_double() != 0) {
    fprintf(stderr, "double plan failed\n");
    return 1;
  }
  if (run_float() != 0) {
    fprintf(stderr, "float plan failed\n");
    return 1;
  }
  printf("fftw-3.3.10 double float threads ok\n");
  fftw_cleanup_threads();
  fftwf_cleanup_threads();
  return 0;
}
