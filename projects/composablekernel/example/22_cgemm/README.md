# Complex General Matrix-Matrix Multiplication (CGEMM)

This example demonstrates a General Matrix-Matrix Multiplication for complex-valued tensors (CGEMM). This operation is a fundamental building block in many scientific and engineering domains, including signal processing, quantum computing, and electromagnetics, where computations are naturally expressed using complex numbers.

## Mathematical Formulation

A complex number `z` can be represented as `z = a + bi`, where `a` is the real part and `b` is the imaginary part. The multiplication of two complex numbers `z1 = a + bi` and `z2 = c + di` is:

$z_1 \cdot z_2 = (a+bi)(c+di) = (ac - bd) + (ad + bc)i$

A CGEMM operation, $D = \alpha \cdot (A \times B) + \beta \cdot C$, involves matrices where each element is a complex number. The core matrix multiplication $A \times B$ is defined as:

$C_{ik} = \sum_j A_{ij} \cdot B_{jk}$

Where each multiplication and addition is a complex operation. This can be broken down into four real-valued GEMM operations:

Let $A = A_r + iA_i$ and $B = B_r + iB_i$. Then the product $C = A \times B$ is:
$C = (A_r + iA_i) \times (B_r + iB_i) = (A_r B_r - A_i B_i) + i(A_r B_i + A_i B_r)$

This shows that one CGEMM can be decomposed into four real GEMMs and two real matrix additions/subtractions.

## Algorithmic Strategy: Fused Complex Arithmetic

A naive implementation would launch six separate real-valued kernels (4 GEMMs, 2 additions). A much more efficient approach, and the one used by Composable Kernel, is to implement CGEMM in a single, fused kernel.

1.  **Data Layout**: Complex numbers are typically stored in an interleaved format, where the real and imaginary parts of an element are adjacent in memory (e.g., `[r1, i1, r2, i2, ...]`). The kernel is designed to work efficiently with this layout.

2.  **Tiled CGEMM**: The kernel uses a standard tiled GEMM algorithm, but the fundamental operations are adapted for complex numbers.
    -   **Loading**: A thread block loads tiles of the complex-valued matrices A and B from global memory into shared memory.
    -   **Complex Multiply-Accumulate**: The core of the algorithm is the multiply-accumulate (MAC) operation. Instead of a single `fma` instruction, each complex MAC involves multiple real-valued `fma` instructions to compute the real and imaginary parts of the product, as shown in the mathematical formulation.
        -   `real_part = (a_r * b_r) - (a_i * b_i)`
        -   `imag_part = (a_r * b_i) + (a_i * b_r)`
    -   These operations are carefully scheduled to maximize instruction-level parallelism and hide latency. The accumulators for both the real and imaginary parts are held in private registers.

3.  **Storing**: After the tile is fully computed, the complex-valued result is written from registers back to the output matrix D in global memory.

By fusing the complex arithmetic directly into the GEMM kernel, we avoid launching multiple kernels and storing large intermediate real-valued matrices, which dramatically reduces kernel launch overhead and memory bandwidth requirements.

## Source Code Organization

-   [`cgemm_xdl.cpp`](./cgemm_xdl.cpp): The main example file. It defines complex-valued input matrices and instantiates the `DeviceGemm` operation, specialized for complex data types.
-   The standard `DeviceGemm` interface from [`../../include/ck/tensor_operation/gpu/device/device_gemm.hpp`](../../include/ck/tensor_operation/gpu/device/device_gemm.hpp) is used. Composable Kernel overloads this interface for complex types (`ck::complex<T>`).
-   The grid-wise GEMM kernel is specialized to handle complex types. When the template arguments for data types are `ck::complex`, the compiler instantiates a version of the kernel where the MAC operations are replaced with the sequence of real-valued operations required for complex multiplication.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/22_cgemm
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run the Example
```bash
# Run the example with default settings
./cgemm_xdl

# Run with verification, data initialization, and timing
./cgemm_xdl 1 2 1
```

## Applications

CGEMM is a critical kernel in many high-performance computing applications:

-   **Digital Signal Processing (DSP)**: The Fast Fourier Transform (FFT), a cornerstone of DSP, can be implemented using complex matrix multiplications. Filtering and convolution in the frequency domain also rely on complex arithmetic.
-   **Quantum Computing Simulation**: The state of a quantum system is described by a vector of complex numbers, and quantum gates are represented by unitary matrices (a special type of complex matrix). Simulating a quantum circuit involves a sequence of CGEMM operations.
-   **Electromagnetics and Wave Physics**: Simulating the propagation of electromagnetic or acoustic waves often involves solving systems of equations with complex numbers to represent the phase and amplitude of the waves.
-   **Communications**: Modern communication systems (like 5G and Wi-Fi) use complex modulation schemes (like QAM) where signals are represented by complex numbers.
