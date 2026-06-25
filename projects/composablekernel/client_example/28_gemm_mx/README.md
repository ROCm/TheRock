# Client Example: GEMM pipeline for microscaling (MX)

## How to Run


### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

```bash
cd composable_kernel/build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D DTYPES="fp8" ..
make -j
make install
```

### Build and run
```bash
/opt/rocm/bin/hipcc gemm_mx_fp8.cpp -o gemm_mx_fp8

# Example run
./gemm_mx_fp8
```

## Source Code Structure

### Directory Layout
```
client_example/28_gemm_mx/
├── gemm_mx_fp8.cpp       # GEMM MX (fp8)
├── CMakeLists.txt        # Build configuration for the example
```
---
[Back to Client Examples](../README.md)
