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
