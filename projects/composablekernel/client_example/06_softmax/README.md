# Client Example: 4D Softmax

## Theory

This client example demonstrates **Softmax computation over 4D tensors**. Softmax is a key operation in deep learning, especially in attention mechanisms and classification, converting logits into normalized probabilities.

**Mathematical Formulation:**
Given input $X$ and axis $a$:
$$
\text{softmax}(X)_i = \frac{\exp(X_i)}{\sum_j \exp(X_j)}
$$

**Algorithmic Background:**
- Softmax is implemented using a numerically stable algorithm:
  1. Subtract the maximum value for numerical stability.
  2. Exponentiate and sum.
  3. Normalize by the sum.
- Efficient parallel Softmax requires careful reduction and memory access patterns.
- This example demonstrates Softmax over a 4D tensor, as used in attention and vision models.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/06_softmax
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./softmax4d
```

## Source Code Structure

### Directory Layout
```
client_example/06_softmax/
├── softmax4d.cpp         # Main client example: sets up, runs, and verifies 4D softmax
├── CMakeLists.txt        # Build configuration for the example
```

### Key Functions

- **main()** (in `softmax4d.cpp`):  
  Sets up input tensors, configures Softmax parameters, launches the Softmax kernel, and verifies the result.
- **Softmax kernel invocation**:  
  Uses the Composable Kernel device API to launch the Softmax operation.

This client example provides a demonstration of efficient, numerically stable Softmax for 4D tensors in deep learning models.
