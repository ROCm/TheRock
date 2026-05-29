// HIPRTC module-launch smoke that passes kernel parameters the way IREE's HIP
// stream command buffer does: an array of hipDeviceptr_t payloads and an array
// of pointers to those payload slots.

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <unistd.h>

#include <cstdio>
#include <string>
#include <vector>

#define CHECK_HIP(expr)                                                           \
  do {                                                                            \
    hipError_t e = (expr);                                                        \
    std::printf("%s -> %d %s\n", #expr, static_cast<int>(e), hipGetErrorString(e)); \
    std::fflush(stdout);                                                          \
    if (e != hipSuccess) _exit(10);                                               \
  } while (0)

#define CHECK_RTC(expr)                                                            \
  do {                                                                             \
    hiprtcResult e = (expr);                                                       \
    std::printf("%s -> %d %s\n", #expr, static_cast<int>(e), hiprtcGetErrorString(e)); \
    std::fflush(stdout);                                                           \
    if (e != HIPRTC_SUCCESS) _exit(20);                                            \
  } while (0)

int main(int argc, char **argv) {
  CHECK_HIP(hipSetDevice(0));

  const char *src = R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void add1(float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  x[i] += 1.0f;
}
)";
  hiprtcProgram prog = nullptr;
  CHECK_RTC(hiprtcCreateProgram(&prog, src, "add1.hip", 0, nullptr, nullptr));
  const char *opts[] = {
      "--gpu-architecture=gfx1201",
      "-I",
      "/Users/anush/github/TheRock/build-macos-egpu/core/clr/dist/include",
  };
  hiprtcResult compile_result = hiprtcCompileProgram(prog, 3, opts);
  std::printf("hiprtcCompileProgram -> %d %s\n", static_cast<int>(compile_result),
              hiprtcGetErrorString(compile_result));
  std::fflush(stdout);
  if (compile_result != HIPRTC_SUCCESS) _exit(21);

  size_t code_size = 0;
  CHECK_RTC(hiprtcGetCodeSize(prog, &code_size));
  std::vector<char> code(code_size);
  CHECK_RTC(hiprtcGetCode(prog, code.data()));

  hipModule_t mod = nullptr;
  hipFunction_t fun = nullptr;
  CHECK_HIP(hipModuleLoadData(&mod, code.data()));
  CHECK_HIP(hipModuleGetFunction(&fun, mod, "add1"));

  float host[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float *dev = nullptr;
  CHECK_HIP(hipMalloc(reinterpret_cast<void **>(&dev), sizeof(host)));
  CHECK_HIP(hipMemcpy(dev, host, sizeof(host), hipMemcpyHostToDevice));

  const char *mode = argc > 1 ? argv[1] : "deviceptr";
  void *args[1] = {};
  void *dev_void = dev;
  hipDeviceptr_t payload[1] = {reinterpret_cast<hipDeviceptr_t>(dev)};
  if (std::string(mode) == "typed") {
    args[0] = &dev;
  } else if (std::string(mode) == "void") {
    args[0] = &dev_void;
  } else {
    args[0] = &payload[0];
  }
  std::printf("mode=%s dev=%p dev_slot=%p void_slot=%p payload=%p payload[0]=%p\n",
              mode, dev, static_cast<void *>(&dev), static_cast<void *>(&dev_void),
              static_cast<void *>(payload), payload[0]);
  CHECK_HIP(hipModuleLaunchKernel(fun, 1, 1, 1, 4, 1, 1, 0, nullptr, args, nullptr));
  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(hipMemcpy(host, dev, sizeof(host), hipMemcpyDeviceToHost));
  std::printf("out=%f,%f,%f,%f\n", host[0], host[1], host[2], host[3]);
  std::fflush(stdout);
  _exit(host[0] == 2.0f && host[1] == 3.0f && host[2] == 4.0f &&
                host[3] == 5.0f
            ? 0
            : 30);
}
