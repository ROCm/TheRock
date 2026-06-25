# Client Example: BatchNorm with Instance ID Selection

## Theory

This client example demonstrates **batch normalization** using explicit instance ID selection. In Composable Kernel, "instance ID" refers to a specific kernel configuration (tile sizes, vectorization, etc.) chosen for a given workload. This allows users to benchmark or select the best-performing kernel for their data shape.

**Mathematical Formulation:**
See [BatchNorm Theory](../13_batchnorm/README.md) for the mathematical details of batch normalization.

**Algorithmic Background:**
- The example shows how to enumerate and select a specific kernel instance by its ID.
- Useful for performance tuning, benchmarking, and debugging.
- BatchNorm is performed in NHWC layout.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/14_instance_id
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (selects a specific kernel instance)
./batchnorm_fwd_instance_id
```

## Source Code Structure

### Directory Layout
```
client_example/14_instance_id/
├── batchnorm_fwd_instance_id.cpp         # Batchnorm forward with instance ID selection
├── CMakeLists.txt                        # Build configuration for the example
```

### Key Functions

- **main()** (in `batchnorm_fwd_instance_id.cpp`):  
  Sets up input tensors, enumerates available kernel instances, selects an instance by ID, launches the batchnorm kernel, and verifies the result.
- **Instance selection**:  
  Demonstrates how to use the Composable Kernel API to list and select kernel configurations.

---

## Additional Details

- Useful for kernel benchmarking and performance tuning.
- Example parameters and instance ID can be adjusted in the source.

---

## Related Examples

- [13_batchnorm](../13_batchnorm/README.md): Batch normalization client API
- [34_batchnorm](../../example/34_batchnorm/README.md): Batch normalization in the main example directory

---
[Back to Client Examples](../README.md)
