// HIPRTC module-launch smoke for macOS eGPU normal-exit isolation.
//
// Build from the TheRock checkout with:
//   clang++ -std=c++17 -D__HIP_PLATFORM_AMD__ \
//     -I build-macos-egpu/core/clr/dist/include \
//     -L build-macos-egpu/core/clr/dist/lib \
//     -Wl,-rpath,$PWD/build-macos-egpu/core/clr/dist/lib \
//     -lamdhip64 -lhiprtc \
//     -o /tmp/hip_macos_smoke_normal \
//     userspace_driver/debug/hip_macos_smoke_normal.cpp
//
// Unlike hip_macos_smoke_fast_exit.cpp, this returns normally and can run HIP
// cleanup calls. Set HIP_MACOS_SMOKE_SKIP_CLEANUP=1 to return normally without
// hipFree/hipModuleUnload/hiprtcDestroyProgram.

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_HIP(expr)                                                              \
  do {                                                                               \
    hipError_t e = (expr);                                                           \
    std::printf("%s -> %d %s\n", #expr, static_cast<int>(e), hipGetErrorString(e)); \
    std::fflush(stdout);                                                             \
    if (e != hipSuccess) return 10;                                                  \
  } while (0)

#define CHECK_RTC(expr)                                                                  \
  do {                                                                                    \
    hiprtcResult e = (expr);                                                              \
    std::printf("%s -> %d %s\n", #expr, static_cast<int>(e), hiprtcGetErrorString(e));   \
    std::fflush(stdout);                                                                  \
    if (e != HIPRTC_SUCCESS) return 20;                                                   \
  } while (0)

int main() {
  int count = 0;
  CHECK_HIP(hipGetDeviceCount(&count));
  std::printf("count=%d\n", count);
  std::fflush(stdout);

  char name[256] = {};
  CHECK_HIP(hipDeviceGetName(name, sizeof(name), 0));
  std::printf("name=%s\n", name);
  std::fflush(stdout);

  const char* src = R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void add1(float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  x[i] += 1.0f;
}
)";
  hiprtcProgram prog = nullptr;
  CHECK_RTC(hiprtcCreateProgram(&prog, src, "add1.hip", 0, nullptr, nullptr));

  const char* opts[] = {
      "--gpu-architecture=gfx1201",
      "-I",
      "/Users/anush/github/TheRock/build-macos-egpu/core/clr/dist/include",
  };
  hiprtcResult compile_result = hiprtcCompileProgram(prog, 3, opts);
  std::printf("hiprtcCompileProgram -> %d %s\n", static_cast<int>(compile_result),
              hiprtcGetErrorString(compile_result));
  size_t log_size = 0;
  hiprtcGetProgramLogSize(prog, &log_size);
  if (log_size > 1) {
    std::vector<char> log(log_size);
    hiprtcGetProgramLog(prog, log.data());
    std::printf("RTC log:\n%s\n", log.data());
  }
  std::fflush(stdout);
  if (compile_result != HIPRTC_SUCCESS) return 21;

  size_t code_size = 0;
  CHECK_RTC(hiprtcGetCodeSize(prog, &code_size));
  std::vector<char> code(code_size);
  CHECK_RTC(hiprtcGetCode(prog, code.data()));
  std::printf("code_size=%zu magic=%02x %02x %02x %02x\n", code_size,
              static_cast<unsigned char>(code[0]), static_cast<unsigned char>(code[1]),
              static_cast<unsigned char>(code[2]), static_cast<unsigned char>(code[3]));
  std::fflush(stdout);

  hipModule_t mod = nullptr;
  hipFunction_t fun = nullptr;
  CHECK_HIP(hipModuleLoadData(&mod, code.data()));
  CHECK_HIP(hipModuleGetFunction(&fun, mod, "add1"));

  float host[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float* dev = nullptr;
  CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&dev), sizeof(host)));
  CHECK_HIP(hipMemcpy(dev, host, sizeof(host), hipMemcpyHostToDevice));

  void* args[] = {&dev};
  CHECK_HIP(hipModuleLaunchKernel(fun, 1, 1, 1, 4, 1, 1, 0, nullptr, args, nullptr));
  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(hipMemcpy(host, dev, sizeof(host), hipMemcpyDeviceToHost));
  std::printf("out=%f,%f,%f,%f\n", host[0], host[1], host[2], host[3]);
  std::fflush(stdout);

  const bool ok = host[0] == 2.0f && host[1] == 3.0f && host[2] == 4.0f && host[3] == 5.0f;
  if (!ok) return 30;

  if (std::getenv("HIP_MACOS_SMOKE_SKIP_CLEANUP") == nullptr) {
    CHECK_HIP(hipFree(dev));
    CHECK_HIP(hipModuleUnload(mod));
    CHECK_RTC(hiprtcDestroyProgram(&prog));
  } else {
    std::printf("normal-exit: skipping explicit cleanup calls\n");
    std::fflush(stdout);
  }

  std::printf("normal-exit: returning from main\n");
  std::fflush(stdout);
  return 0;
}
