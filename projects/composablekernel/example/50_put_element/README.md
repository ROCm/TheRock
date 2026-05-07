# Put Element Operation

This example demonstrates a **put element operation**, which scatters or places elements from a source tensor into specific positions of a destination tensor based on index arrays. This is a fundamental operation for implementing sparse updates, scatter operations, and advanced indexing patterns in deep learning and scientific computing.

## Mathematical Formulation

The put element operation updates specific positions in a destination tensor using values from a source tensor and position information from index tensors.

Given:
- **Destination tensor** `D` with shape `[D0, D1, ..., Dn]`
- **Source tensor** `S` with shape `[M, ...]` containing values to be placed
- **Index tensors** `I0, I1, ..., In` with shape `[M]` specifying destination coordinates
- **Update mode**: how to handle multiple updates to the same position

The operation performs:
$D[I0[i], I1[i], ..., In[i]] \leftarrow \text{Update}(D[I0[i], I1[i], ..., In[i]], S[i])$

For each element `i` from `0` to `M-1`.

**Update modes**:
- **Overwrite**: `D[idx] = S[i]`
- **Add**: `D[idx] += S[i]`
- **Multiply**: `D[idx] *= S[i]`
- **Max**: `D[idx] = max(D[idx], S[i])`
- **Min**: `D[idx] = min(D[idx], S[i])`

## Algorithmic Strategy: Parallel Scatter with Conflict Resolution

The implementation must handle parallel updates and potential conflicts when multiple source elements target the same destination position.

1.  **Grid Scheduling**: The operation is parallelized over the source elements. Each thread is assigned to process one or more elements from the source tensor.

2.  **Index Calculation**: For each source element, threads:
    -   Read the corresponding indices from the index tensors
    -   Validate that indices are within bounds
    -   Calculate the linear memory address in the destination tensor

3.  **Conflict Resolution**: When multiple threads attempt to update the same destination position:
    -   **Atomic Operations**: Use atomic functions for commutative operations (add, max, min)
    -   **Serialization**: For non-commutative operations, use locks or other synchronization
    -   **Deterministic Ordering**: Ensure consistent results across runs

4.  **Memory Access Optimization**:
    -   Coalesced reading from source and index tensors
    -   Efficient atomic operations on destination tensor
    -   Minimize memory bank conflicts

## Source Code Organization

-   [`put_element_xdl.cpp`](./put_element_xdl.cpp): The main example file. It sets up the destination tensor, source tensor, index arrays, and instantiates the `DevicePutElement` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_put_element.hpp`](../../include/ck/tensor_operation/gpu/device/device_put_element.hpp): The high-level device interface for put element operations.
-   [`../../include/ck/tensor_operation/gpu/grid/gridwise_put_element.hpp`](../../include/ck/tensor_operation/gpu/grid/gridwise_put_element.hpp): The grid-wise kernel implementing the parallel scatter algorithm with conflict resolution.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/50_put_element
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
./put_element_xdl

# Run with verification, data initialization, and timing
./put_element_xdl 1 2 1
```

## Applications

Put element operations are fundamental to many advanced algorithms and data structures.

-   **Sparse Neural Networks**: Updating specific weights or activations in sparse neural network architectures where only a subset of parameters are active.
-   **Graph Neural Networks**: Scatter operations for aggregating information from neighboring nodes to target nodes in graph structures.
-   **Embedding Updates**: Updating specific rows in embedding tables based on sparse input indices, common in recommendation systems and NLP models.
-   **Histogram Computation**: Accumulating counts or values into histogram bins based on computed indices.
-   **Sparse Linear Algebra**: Implementing sparse matrix operations where values are placed at specific coordinate positions.
-   **Advanced Indexing**: Supporting NumPy-style advanced indexing patterns for tensor manipulation.

## Performance Considerations

The performance of put element operations depends heavily on the access patterns:

-   **Random Access**: Scattered indices lead to poor memory locality and cache performance
-   **Atomic Contention**: High conflict rates (many updates to same positions) can severely impact performance
-   **Memory Bandwidth**: The operation is typically memory-bound, especially with good locality
-   **Load Balancing**: Uneven distribution of conflicts can cause load imbalance across threads
