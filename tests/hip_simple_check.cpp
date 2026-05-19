// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Minimal HIP kernel WITHOUT printf, to isolate whether the gfx1150
// sanity-test hang (TheRock#3199) is caused by printf buffer handling
// or by kernel dispatch itself.
//
// Includes a SIGALRM watchdog that dumps a backtrace + /proc/self/maps
// if the process hangs for 20 seconds.

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

static void dump_backtrace_and_maps(int sig) {
  fprintf(stderr, "\n=== WATCHDOG: process hung for 20s (signal %d) ===\n",
          sig);

  // Dump backtrace addresses
  void *frames[64];
  int n = backtrace(frames, 64);
  fprintf(stderr, "=== backtrace (%d frames) ===\n", n);
  backtrace_symbols_fd(frames, n, STDERR_FILENO);

  // Dump /proc/self/maps so addresses can be resolved offline
  fprintf(stderr, "\n=== /proc/self/maps ===\n");
  FILE *maps = fopen("/proc/self/maps", "r");
  if (maps) {
    char buf[512];
    while (fgets(buf, sizeof(buf), maps))
      fputs(buf, stderr);
    fclose(maps);
  }

  // Dump /proc/self/stack (kernel stack, if readable)
  fprintf(stderr, "\n=== /proc/self/stack ===\n");
  FILE *stack = fopen("/proc/self/stack", "r");
  if (stack) {
    char buf[512];
    while (fgets(buf, sizeof(buf), stack))
      fputs(buf, stderr);
    fclose(stack);
  } else {
    fprintf(stderr, "(not readable)\n");
  }

  _exit(2);
}

__global__ void squares_no_printf(int *buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  buf[i] = i * i;
}

int main() {
  // Set up watchdog: SIGALRM after 20s
  signal(SIGALRM, dump_backtrace_and_maps);
  alarm(20);

  constexpr int gridsize = 1;
  constexpr int blocksize = 64;
  constexpr int size = gridsize * blocksize;
  int *d_buf;
  fprintf(stderr, "hip_simple_check: hipHostMalloc\n");
  hipHostMalloc(&d_buf, size * sizeof(int));
  fprintf(stderr, "hip_simple_check: hipLaunchKernelGGL\n");
  hipLaunchKernelGGL(squares_no_printf, gridsize, blocksize, 0, 0, d_buf);
  fprintf(stderr, "hip_simple_check: hipDeviceSynchronize\n");
  hipDeviceSynchronize();

  // Cancel watchdog — we didn't hang
  alarm(0);
  fprintf(stderr, "hip_simple_check: checking results\n");

  int mismatches_count = 0;
  for (int i = 0; i < size; ++i) {
    int square = i * i;
    if (d_buf[i] != square) {
      fprintf(stderr,
              "Element at index %d expected value %d, actual value: %d\n", i,
              square, d_buf[i]);
      ++mismatches_count;
    }
  }
  if (mismatches_count > 0) {
    fprintf(stderr, "There were %d mismatches\n", mismatches_count);
    return 1;
  }

  fprintf(stderr, "hip_simple_check: PASSED\n");
  return 0;
}
