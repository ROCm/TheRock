// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

constexpr int kWaveSize = 32;
constexpr int kPackedFp8RegsPerLane = 16;
constexpr int kF32RegsPerLane = 8;
constexpr int kPackedFp8MatrixWords = kWaveSize * kPackedFp8RegsPerLane;
constexpr int kOutputElements = kWaveSize * kF32RegsPerLane;
constexpr float kExpectedAllOnesDot = 128.0f;
constexpr float kValidationTolerance = 0.001f;

using PackedFp8WmmaFrag =
    __attribute__((__vector_size__(kPackedFp8RegsPerLane * sizeof(int32_t)))) int32_t;
using F32WmmaFrag =
    __attribute__((__vector_size__(kF32RegsPerLane * sizeof(float)))) float;

bool isGfx1250(const char* arch) {
  return arch != nullptr && std::strncmp(arch, "gfx1250", 7) == 0;
}

uint32_t packFp8x4(uint8_t value) {
  const uint32_t v = value;
  return v | (v << 8) | (v << 16) | (v << 24);
}

int checkHip(hipError_t err, const char* expr, const char* file, int line) {
  if (err == hipSuccess) {
    return 0;
  }
  std::cerr << "HIP error at " << file << ":" << line << " for " << expr << ": "
            << hipGetErrorString(err) << " (" << static_cast<int>(err) << ")\n";
  return 1;
}

}  // namespace

#define HIP_CHECK(expr)                                       \
  do {                                                        \
    if (int rc = checkHip((expr), #expr, __FILE__, __LINE__)) \
      return rc;                                              \
  } while (false)

__device__ F32WmmaFrag runGfx1250Fp8Wmma(PackedFp8WmmaFrag a,
                                         PackedFp8WmmaFrag b,
                                         F32WmmaFrag c) {
#if defined(__gfx1250__)
#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8)
  return __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(a, b, static_cast<int16_t>(0),
                                                      c, false, false);
#else
#error "gfx1250 hotswap smoke test requires __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8"
#endif
#else
  return c;
#endif
}

__global__ __launch_bounds__(kWaveSize) void gfx1250Fp8WmmaMatmulSmoke(
    const uint32_t* a, const uint32_t* b, float* out) {
  const int lane = static_cast<int>(threadIdx.x);

  PackedFp8WmmaFrag aFrag;
  PackedFp8WmmaFrag bFrag;
  F32WmmaFrag cFrag = {};

  for (int i = 0; i < kPackedFp8RegsPerLane; ++i) {
    const int offset = lane * kPackedFp8RegsPerLane + i;
    aFrag[i] = static_cast<int32_t>(a[offset]);
    bFrag[i] = static_cast<int32_t>(b[offset]);
  }

  // All lanes in the wave participate in a 16x16x128 FP8 matmul. With every
  // A and B element set to 1.0 and C initialized to 0, each FP32 output is 128.
  cFrag = runGfx1250Fp8Wmma(aFrag, bFrag, cFrag);

  for (int i = 0; i < kF32RegsPerLane; ++i) {
    out[lane * kF32RegsPerLane + i] = cFrag[i];
  }
}

int main() {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    std::cout << "No HIP devices found; skipping gfx1250 hotswap smoke test.\n";
    return 0;
  }

  HIP_CHECK(hipSetDevice(0));

  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  if (!isGfx1250(props.gcnArchName)) {
    std::cout << "Device arch is " << props.gcnArchName
              << "; skipping gfx1250 hotswap smoke test.\n";
    return 0;
  }
  if (props.warpSize != kWaveSize) {
    std::cerr << "gfx1250 hotswap smoke test requires wave size " << kWaveSize
              << ", device reports " << props.warpSize << ".\n";
    return 1;
  }

  const uint8_t fp8One =
      __hip_cvt_float_to_fp8(1.0f, __HIP_NOSAT, __HIP_E4M3);
  const uint32_t packedFp8Ones = packFp8x4(fp8One);

  std::array<uint32_t, kPackedFp8MatrixWords> hostA{};
  std::array<uint32_t, kPackedFp8MatrixWords> hostB{};
  std::array<float, kOutputElements> hostOut{};
  hostA.fill(packedFp8Ones);
  hostB.fill(packedFp8Ones);

  uint32_t* deviceA = nullptr;
  uint32_t* deviceB = nullptr;
  float* deviceOut = nullptr;
  HIP_CHECK(hipMalloc(&deviceA, hostA.size() * sizeof(hostA[0])));
  HIP_CHECK(hipMalloc(&deviceB, hostB.size() * sizeof(hostB[0])));
  HIP_CHECK(hipMalloc(&deviceOut, hostOut.size() * sizeof(hostOut[0])));
  HIP_CHECK(hipMemcpy(deviceA, hostA.data(), hostA.size() * sizeof(hostA[0]),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceB, hostB.data(), hostB.size() * sizeof(hostB[0]),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(deviceOut, 0, hostOut.size() * sizeof(hostOut[0])));

  gfx1250Fp8WmmaMatmulSmoke<<<1, kWaveSize>>>(deviceA, deviceB, deviceOut);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hostOut.data(), deviceOut, hostOut.size() * sizeof(hostOut[0]),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(deviceOut));
  HIP_CHECK(hipFree(deviceB));
  HIP_CHECK(hipFree(deviceA));

  for (std::size_t i = 0; i < hostOut.size(); ++i) {
    const float value = hostOut[i];
    if (!std::isfinite(value) ||
        std::fabs(value - kExpectedAllOnesDot) > kValidationTolerance) {
      std::cerr << "Unexpected WMMA output at element " << i << ": " << value
                << " expected " << kExpectedAllOnesDot << "\n";
      return 1;
    }
  }

  std::cout << "gfx1250 B0-to-A0 hotswap FP8 WMMA matmul smoke test completed.\n";
  return 0;
}
