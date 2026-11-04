// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

extern "C" void hip_probe(int *value) { ++*value; }

// Compile a device pass too; host sanitizers must not instrument this kernel.
extern "C" __global__ void device_probe(int *value) { ++*value; }
