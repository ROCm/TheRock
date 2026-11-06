# PyTorch Smoke Tests

This repository contains a set of basic smoke tests for verifying fundamental PyTorch tensor operations. These tests ensure that basic matrix operations function correctly, but they are **not exhaustive PyTorch tests**.

## Disclaimer

⚠️ **These tests are not full PyTorch tests.** They are designed as quick checks for basic tensor operations and do not replace comprehensive testing of PyTorch functionalities.

## Test Overview

The following operations are covered in these smoke tests:

- **Matrix Multiplication (`torch.mm`)**
- **Batch Matrix Multiplication (`torch.bmm`)**
- **Matrix Multiplication using `@` operator**
- **Element-wise Multiplication (`*`)**
- **Matrix Transposition (`torch.t`)**
- **Dot Product (`torch.dot`)**
- **Matrix-Vector Multiplication (`torch.mv`)**
- **General Matrix Multiplication (`torch.matmul`)**
- **Convolution (`torch.conv2d` and `torch.nn.functional.conv_transpose2d`)**
- **Scaled Dot-Product Attention (`torch.nn.functional.scaled_dot_product_attention`)**

## Scaled Dot-Product Attention (SDPA)

`TestSDPA` is a representative smoketest for fused scaled dot-product attention,
a core operation for modern ML workloads. On ROCm, the fused FLASH_ATTENTION and
EFFICIENT_ATTENTION backends dispatch to AOTriton kernels whose usability depends
on how the wheel was built and packaged: a wheel that ships mismatched kernel
images can fail at kernel launch time with only a deferred error
([#7315](https://github.com/ROCm/TheRock/issues/7315),
[#7992](https://github.com/ROCm/TheRock/issues/7992)).

Each test forces a single fused backend with `torch.nn.attention.sdpa_kernel`, so
an unavailable kernel raises immediately instead of silently falling back to the
unfused MATH backend, and checks the BF16 output against an independent CPU FP32
reference computed from the same BF16-quantized inputs (`atol=rtol=0.03` covers
BF16 rounding on the tested shapes). Covered configurations, all BF16:

- FLASH_ATTENTION, Q/K/V `[1, 2, 16, 64]`, no mask
- EFFICIENT_ATTENTION, Q/K/V `[1, 2, 16, 64]`, additive bias
- EFFICIENT_ATTENTION, Q/K/V `[54, 12, 196, 64]`, additive bias (a larger
  vision-transformer-shaped input)

Not covered: the MATH backend, other dtypes, backward passes, performance, and
comprehensive attention configurations. Broader coverage is planned by running
pytorch's own `test_transformers.py`
([#9641](https://github.com/ROCm/TheRock/issues/9641)).

Run against an installed ROCm torch wheel with the standard runner (see
`run_pytorch_smoke_tests.py` for options):

```bash
python external-builds/pytorch/run_pytorch_smoke_tests.py --amdgpu-family gfx1151 -- -v -k fused_sdpa
```
