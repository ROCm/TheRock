# ROCM SDK User Example

This example shows a CMake project with the supported way to depend on various
ROCM libraries and tools from C++. We attempt to keep this up to date and
build/test it as part of the overall project.

## gfx1250 HotSwap Smoke Test

The `gfx1250-hotswap-a0-smoke` test is opt-in because it requires a gfx1250
device. It builds a gfx1250 B0 code object with a small FP8 all-ones WMMA
matmul kernel that uses the 128-wide
`__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8` intrinsic. That intrinsic lowers
to the B0-only `v_wmma_f32_16x16x128_fp8_fp8` instruction. On A0 hardware, the
code object is expected to require the ROCm HotSwap B0-to-A0 rewrite path before
dispatch, and the test validates that each FP32 output element is 128.

Configure the example directly with:

```bash
cmake -GNinja -S examples/cpp-sdk-user -B build/cpp-sdk-user \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DENABLE_DEVICE_TEST=ON \
  -DENABLE_GFX1250_HOTSWAP_TEST=ON
cmake --build build/cpp-sdk-user --target gfx1250-hotswap-a0-smoke
ctest --test-dir build/cpp-sdk-user -R gfx1250-hotswap-a0-smoke --output-on-failure
```

When running through the top-level TheRock example test, set:

```bash
-DTHEROCK_EXAMPLES_ENABLE_DEVICE_TESTS=ON
-DTHEROCK_EXAMPLES_ENABLE_GFX1250_HOTSWAP_TEST=ON
```
