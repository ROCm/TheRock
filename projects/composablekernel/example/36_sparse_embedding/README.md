# Sparse Embedding Lookup

This example demonstrates a **sparse embedding lookup**, a fundamental operation in deep learning models that process sparse, high-cardinality categorical features, such as words in a vocabulary or user IDs in a recommendation system. The operation gathers feature vectors from a large embedding table based on a set of sparse input indices.

## Mathematical Formulation

The operation can be described as a lookup or gather operation.

Given:
-   An **Embedding Table** `W`, a dense 2D tensor of shape `[VocabularySize, EmbeddingDim]`. Each row of `W` is a feature vector (an embedding) for a specific category.
-   A set of **Indices** `I`, a tensor of integer IDs (e.g., shape `[BatchSize, SequenceLength]`) that specify which embeddings to look up.
-   An optional **Sparsity-aware Optimizer** state, such as momentum vectors, which must also be looked up and updated.

The operation produces an **Output Tensor** `O` by gathering the rows from `W` corresponding to the indices in `I`.
$O_{bsj} = W_{I_{bs}, j}$

Where `b` is the batch index, `s` is the sequence index, and `j` is the embedding dimension index. The output tensor `O` will have a shape like `[BatchSize, SequenceLength, EmbeddingDim]`.

## Algorithmic Strategy: Parallel Gather

Unlike compute-bound operations like GEMM, an embedding lookup is almost entirely **memory-bound**. The primary challenge is to perform the gather operation from the potentially very large embedding table `W` as efficiently as possible.

1.  **Grid Scheduling**: The lookup problem is parallelized over the indices. The grid of threads is typically launched to match the shape of the index tensor `I`. Each thread is assigned to handle the lookup for a single index.

2.  **Gather Operation**:
    -   Each thread reads its assigned index `id = I[b, s]` from the index tensor.
    -   The thread then calculates the memory address of the start of the corresponding embedding vector in the table `W`. This is typically `address = base_address_W + id * EmbeddingDim * sizeof(DataType)`.
    -   The thread then reads the entire embedding vector of size `EmbeddingDim` from that address in global memory and writes it to the corresponding position in the output tensor `O`.

3.  **Memory Access Coalescing**: Performance is highly dependent on the memory access patterns.
    -   If multiple threads in a warp access indices that are close to each other, their memory reads from the embedding table `W` might also be close, leading to some coalescing and better memory bandwidth utilization.
    -   However, if the indices are random and scattered, the memory accesses will be random, leading to poor cache utilization and low memory bandwidth. This is often the bottleneck.

4.  **Fused Optimizer Update**: In training, the embedding lookup is part of a larger forward-backward-update cycle. For sparse features, only the embedding vectors that were actually used (the "hot" embeddings) need their gradients computed and their weights updated. High-performance implementations often fuse the backward pass (gradient accumulation) and the optimizer step (e.g., SGD or Adam update) for these hot embeddings directly into a specialized kernel to avoid multiple passes over the embedding table. This example focuses on the forward-pass lookup.

## Source Code Organization

-   [`sparse_embedding_xdl.cpp`](./sparse_embedding_xdl.cpp): The main example file. It sets up the embedding table `W`, the index tensor `I`, and instantiates the `DeviceSparseEmbedding` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_sparse_embedding.hpp`](../../include/ck/tensor_operation/gpu/device/device_sparse_embedding.hpp): The high-level device interface for the sparse embedding lookup.
-   The underlying grid-wise kernel is a straightforward gather kernel. Its performance is almost entirely dictated by the efficiency of its memory load and store operations.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/36_sparse_embedding
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
./sparse_embedding_xdl

# Run with verification, data initialization, and timing
./sparse_embedding_xdl 1 2 1
```

## Applications

Embedding layers are the first step in a vast number of deep learning models:

-   **Natural Language Processing (NLP)**: Models like BERT and GPT use embedding layers to convert integer token IDs from a vocabulary into dense vector representations.
-   **Recommender Systems**: Models use embeddings to represent users and items. The input to the model is often a set of sparse IDs (e.g., user ID, watched movie IDs), which are converted to dense vectors via embedding lookups. Embedding tables in these systems can be enormous (terabytes in size).
-   **Graph Neural Networks**: Nodes in a graph are often represented by feature vectors, which can be stored in an embedding table and looked up as needed.
-   **Any model with categorical features**: Whenever a model needs to process non-numeric categorical data (e.g., "product category", "day of the week"), it is typically first converted to an integer ID and then to a dense vector via an embedding layer.
