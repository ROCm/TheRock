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

## Windows AOTriton SDPA packaging

`TestSDPA` checks the installed wheel on a physical Windows gfx1151 GPU, addressing
the packaging-test gap described in [#7315](https://github.com/ROCm/TheRock/issues/7315#issuecomment-5274800279).
It forces FLASH without a mask and EFFICIENT with additive bias on BF16
`[1, 2, 16, 64]` inputs, plus EFFICIENT on vision inputs `[54, 12, 196, 64]`.
These shapes reproduce a historical runtime/image mismatch without model downloads.
The reference is independent CPU FP32 matmul/softmax using the same CPU-generated,
BF16-quantized inputs; `atol=rtol=0.03` is the validated BF16 acceptance threshold.

Each case has its own process, inherits the runner's GPU isolation, enables only
one fused backend, and observes the actual aten SDPA operator with a narrowly scoped
`TorchDispatchMode`. Non-clearing HIP error checks immediately after SDPA, after
synchronization and after a downstream tensor operation catch deferred launch errors.
Missing required backends or runtime libraries fail; other platforms/architectures
are explicitly outside scope, and no available GPU is reported as NOT RUN (skipped).
This covers BF16 forward execution, not backward, performance or other GPU families.

Run against an installed ROCm torch package (including the appropriate device extras
for multi-arch wheels, e.g. `torch[device-gfx1151]`), not a PyTorch source checkout:

```bash
python external-builds/pytorch/run_pytorch_smoke_tests.py --amdgpu-family gfx1151 -- -v -s -k fused_sdpa
```

The existing `test_pytorch_wheels.yml` smoke-test step discovers these tests
automatically. Selected Windows release jobs currently test the primary Python 3.11
row after wheel publication; a failure fails that test job, not publication.
This adds no Windows GPU presubmit scheduling or release gate. Ordinary hosted CPU
unit CI does not run these GPU smoke tests.
