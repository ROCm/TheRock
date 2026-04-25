// Ahead-of-time HIP smoke for macOS eGPU bring-up.
//
// Build from the TheRock checkout with:
//   HIP_PATH=$PWD/build-macos-egpu/core/clr/dist \
//   ROCM_PATH=$PWD/build-macos-egpu/core/clr/dist \
//   PATH=$PWD/build-macos-egpu/core/clr/dist/bin:$PWD/build-macos-egpu/core/clr/dist/lib/llvm/bin:$PATH \
//   build-macos-egpu/core/clr/dist/bin/hipcc --offload-arch=gfx1201 \
//     userspace_driver/debug/hip_macos_hipcc_smoke.hip \
//     -o /tmp/hip_macos_hipcc_smoke

#include <hip/hip_runtime.h>

#include <cstdio>

#define CHECK_HIP(expr)                                                              \
  do {                                                                               \
    hipError_t e = (expr);                                                           \
    std::printf("%s -> %d %s\n", #expr, static_cast<int>(e), hipGetErrorString(e)); \
    std::fflush(stdout);                                                             \
    if (e != hipSuccess) return 10;                                                  \
  } while (0)

__global__ void add2(float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  x[i] += 2.0f;
}

int main() {
  int count = 0;
  CHECK_HIP(hipGetDeviceCount(&count));
  std::printf("count=%d\n", count);

  char name[256] = {};
  CHECK_HIP(hipDeviceGetName(name, sizeof(name), 0));
  std::printf("name=%s\n", name);

  float host[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float* dev = nullptr;
  CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&dev), sizeof(host)));
  CHECK_HIP(hipMemcpy(dev, host, sizeof(host), hipMemcpyHostToDevice));

  add2<<<1, 4>>>(dev);
  CHECK_HIP(hipGetLastError());
  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(hipMemcpy(host, dev, sizeof(host), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(dev));

  std::printf("out=%f,%f,%f,%f\n", host[0], host[1], host[2], host[3]);
  std::fflush(stdout);
  return host[0] == 3.0f && host[1] == 4.0f && host[2] == 5.0f && host[3] == 6.0f ? 0 : 30;
}
