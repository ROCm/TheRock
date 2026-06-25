# Client Example: Parallel Reduction (NHWC)

## Theory

This client example demonstrates **parallel reduction operations** over NHWC tensors. Reduction is a fundamental operation in deep learning for computing statistics (such as batch mean/variance), loss aggregation, and normalization.

**Mathematical Formulation:**
Given a tensor $X[N, H, W, C]$ and a reduction axis (e.g., channel $C$):
- **Sum**: $Y_{n,h,w} = \sum_c X_{n,h,w,c}$
- **Max**: $Y_{n,h,w} = \max_c X_{n,h,w,c}$
- **Mean**: $Y_{n,h,w} = \frac{1}{C} \sum_c X_{n,h,w,c}$

**Algorithmic Background:**
- Reductions are implemented using parallel tree or segmented reduction algorithms.
- Efficient reductions require careful memory access, synchronization, and sometimes numerically stable algorithms.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/26_reduce
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (reduce over channel dimension)
./reduce_nhwc_c
```

## Source Code Structure

### Directory Layout
```
client_example/26_reduce/
├── reduce_nhwc_c.cpp         # Main client example: reduction over NHWC tensors (channel axis)
├── CMakeLists.txt            # Build configuration for the example
```

### Key Functions

- **main()** (in `reduce_nhwc_c.cpp`):  
  Sets up input tensors, configures reduction parameters, launches the reduction kernel, and verifies the result.
- **Reduction kernel invocation**:  
  Uses the Composable Kernel device API to launch the reduction operation.

---

## Additional Details

- Supports sum, max, mean, and other reductions over NHWC tensors.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [12_reduce](../../example/12_reduce/README.md): Parallel reduction in the main example directory

---
[Back to Client Examples](../README.md)
