/*
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * AMDGPU Code Object Version Backward Compatibility Test (hipRTC)
 *
 * Self-contained executable that uses hipRTC to compile a simple vector-add
 * kernel with the default code object version and then with the two prior
 * versions (n-1 and n-2).  For each variant it:
 *   1. Compiles the kernel source at runtime via hipRTC.
 *   2. Detects the actual code object version embedded in the compiled blob.
 *   3. Loads and launches the kernel through the HIP driver API.
 *   4. Verifies the computation results on the host.
 *
 * Output is one JSON object per line (JSONL) so the Python test harness can
 * parse results trivially.  Exit code is 0 only when all variants pass.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

static constexpr size_t N = 1024;
static constexpr unsigned THREADS_PER_BLOCK = 256;

static constexpr auto kernel_src = R"(
extern "C"
__global__ void vecadd(const float* A, const float* B, float* C, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
)";

/* ELF OS/ABI value for AMDGPU HSA (from LLVM / AMD ABI spec). */
static constexpr unsigned char ELFOSABI_AMDGPU_HSA = 64;

/*
 * Scan a compiled code blob for an AMDGPU HSA device ELF and return
 * the code object version (= EI_ABIVERSION + 2).  Returns -1 when no
 * AMDGPU ELF is found.
 */
static int detect_cov(const char* data, size_t size) {
    for (size_t i = 0; i + 16 <= size; ++i) {
        if (data[i]     != '\x7f' || data[i + 1] != 'E' ||
            data[i + 2] != 'L'    || data[i + 3] != 'F')
            continue;
        if (static_cast<unsigned char>(data[i + 7]) == ELFOSABI_AMDGPU_HSA) {
            int abi_ver = static_cast<unsigned char>(data[i + 8]);
            return abi_ver + 2;
        }
    }
    return -1;
}

static void emit_json(const char* variant, int requested_cov,
                       int detected_cov, const char* status,
                       const char* error = nullptr) {
    std::printf("{\"variant\":\"%s\",\"requested_cov\":%d,"
                "\"detected_cov\":%d,\"status\":\"%s\"",
                variant, requested_cov, detected_cov, status);
    if (error)
        std::printf(",\"error\":\"%s\"", error);
    std::printf("}\n");
    std::fflush(stdout);
}

/*
 * Compile the kernel source with hipRTC.  When use_default is true no
 * -mcode-object-version flag is passed; otherwise the supplied
 * cov_version is requested explicitly.
 *
 * On success the compiled code is appended to |code_out| and true is
 * returned.
 */
static bool rtc_compile(bool use_default, int cov_version,
                         std::vector<char>& code_out) {
    hiprtcProgram prog;
    if (hiprtcCreateProgram(&prog, kernel_src, "vecadd.hip",
                            0, nullptr, nullptr) != HIPRTC_SUCCESS)
        return false;

    std::vector<const char*> opts;
    char flag[64] = {};
    if (!use_default) {
        std::snprintf(flag, sizeof(flag),
                      "-mcode-object-version=%d", cov_version);
        opts.push_back(flag);
    }

    hiprtcResult comp = hiprtcCompileProgram(
        prog, static_cast<int>(opts.size()),
        opts.empty() ? nullptr : opts.data());

    if (comp != HIPRTC_SUCCESS) {
        size_t log_sz = 0;
        hiprtcGetProgramLogSize(prog, &log_sz);
        if (log_sz > 1) {
            std::vector<char> log(log_sz);
            hiprtcGetProgramLog(prog, log.data());
            std::fprintf(stderr, "hipRTC compile log: %s\n", log.data());
        }
        hiprtcDestroyProgram(&prog);
        return false;
    }

    size_t code_sz = 0;
    hiprtcGetCodeSize(prog, &code_sz);
    code_out.resize(code_sz);
    hiprtcGetCode(prog, code_out.data());
    hiprtcDestroyProgram(&prog);
    return true;
}

/*
 * Load a compiled code blob, launch the vecadd kernel, and verify results.
 * Returns true when the kernel produces correct output.
 */
static bool load_and_verify(const std::vector<char>& code) {
    hipModule_t mod = nullptr;
    hipFunction_t func = nullptr;

    if (hipModuleLoadData(&mod, code.data()) != hipSuccess)
        return false;
    if (hipModuleGetFunction(&func, mod, "vecadd") != hipSuccess) {
        hipModuleUnload(mod);
        return false;
    }

    constexpr size_t bytes = N * sizeof(float);
    std::vector<float> h_A(N), h_B(N), h_C(N);
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    hipDeviceptr_t d_A, d_B, d_C;
    bool ok = false;
    if (hipMalloc(reinterpret_cast<void**>(&d_A), bytes) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&d_B), bytes) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&d_C), bytes) != hipSuccess)
        goto cleanup;

    if (hipMemcpyHtoD(d_A, h_A.data(), bytes) != hipSuccess ||
        hipMemcpyHtoD(d_B, h_B.data(), bytes) != hipSuccess)
        goto cleanup;

    {
        size_t n_val = N;
        struct { hipDeviceptr_t a; hipDeviceptr_t b; hipDeviceptr_t c;
                 size_t n; } args{d_A, d_B, d_C, n_val};
        size_t arg_sz = sizeof(args);
        void* config[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
            HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_sz,
            HIP_LAUNCH_PARAM_END
        };
        unsigned blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (hipModuleLaunchKernel(func, blocks, 1, 1,
                                   THREADS_PER_BLOCK, 1, 1,
                                   0, nullptr, nullptr, config) != hipSuccess)
            goto cleanup;
    }

    if (hipDeviceSynchronize() != hipSuccess)
        goto cleanup;
    if (hipMemcpyDtoH(h_C.data(), d_C, bytes) != hipSuccess)
        goto cleanup;

    ok = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) + static_cast<float>(i * 2);
        if (h_C[i] != expected) {
            ok = false;
            std::fprintf(stderr, "Mismatch at %zu: got %f, expected %f\n",
                         i, h_C[i], expected);
            break;
        }
    }

cleanup:
    hipFree(reinterpret_cast<void*>(d_A));
    hipFree(reinterpret_cast<void*>(d_B));
    hipFree(reinterpret_cast<void*>(d_C));
    hipModuleUnload(mod);
    return ok;
}

int main() {
    int failures = 0;

    /* --- 1. Compile with default settings and detect version n. ---------- */
    std::vector<char> default_code;
    if (!rtc_compile(/*use_default=*/true, 0, default_code)) {
        emit_json("default", -1, -1, "FAIL", "hipRTC compilation failed");
        return 1;
    }

    int n = detect_cov(default_code.data(), default_code.size());
    if (n < 0) {
        emit_json("default", -1, -1, "FAIL",
                  "could not detect code object version from compiled blob");
        return 1;
    }

    if (!load_and_verify(default_code)) {
        emit_json("default", -1, n, "FAIL", "kernel execution/verification failed");
        return 1;
    }
    emit_json("default", -1, n, "PASS");

    /* --- 2. Backward-compat variants: n-1 and n-2. ---------------------- */
    for (int offset = 1; offset <= 2; ++offset) {
        int target = n - offset;
        const char* label = (offset == 1) ? "n_minus_1" : "n_minus_2";

        if (target < 2) {
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                          "version %d below minimum (derived from n=%d)",
                          target, n);
            emit_json(label, target, -1, "SKIP", msg);
            continue;
        }

        std::vector<char> code;
        if (!rtc_compile(/*use_default=*/false, target, code)) {
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                          "hipRTC compilation failed for -mcode-object-version=%d",
                          target);
            emit_json(label, target, -1, "FAIL", msg);
            ++failures;
            continue;
        }

        int detected = detect_cov(code.data(), code.size());
        if (detected != target) {
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                          "expected COV %d but detected %d", target, detected);
            emit_json(label, target, detected, "FAIL", msg);
            ++failures;
            continue;
        }

        if (!load_and_verify(code)) {
            emit_json(label, target, detected, "FAIL",
                      "kernel execution/verification failed");
            ++failures;
            continue;
        }
        emit_json(label, target, detected, "PASS");
    }

    return failures > 0 ? 1 : 0;
}
