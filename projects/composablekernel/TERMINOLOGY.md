[Back to the main page](./README.md)

# Composable Kernel Terminology

This document provides a technical reference for terminology used in the Composable Kernel library, organized by conceptual progression from hardware to machine learning operations.

---

## Glossary Index (Alphabetical)

- [Add+Multiply](#addmultiply)
- [Bank Conflict](#bank-conflict)
- [Batched GEMM](#batched-gemm)
- [Benchmark](#benchmark)
- [Block Size](#block-size)
- [Block Tile](#block-tile)
- [Compute Unit (CU)](#compute-unit-cu)
- [Coordinate Transformation Primitives](#coordinate-transformation-primitives)
- [CUDA](#cuda)
- [Dense Tensor](#dense-tensor)
- [Descriptor](#descriptor)
- [Device](#device)
- [Elementwise](#elementwise)
- [Epilogue](#epilogue)
- [Fast Changing Dimension](#fast-changing-dimension)
- [GEMM](#gemm-general-matrix-multiply)
- [GEMV](#gemv)
- [Grouped GEMM](#grouped-gemm)
- [Global Memory](#global-memory)
- [Grid](#grid)
- [Host](#host)
- [HIP](#hip)
- [Inner Dimension](#inner-dimension)
- [Inner Product](#inner-product)
- [Input/Problem Shape](#inputproblem-shape)
- [Kernel](#kernel)
- [Launch Parameters](#launch-parameters)
- [Load Tile](#load-tile)
- [LDS Banks](#lds-banks)
- [Matrix Core](#matrix-core)
- [MFMA (Matrix Fused Multiply-Add)](#mfma-matrix-fused-multiply-add)
- [Occupancy](#occupancy)
- [Outer Dimension](#outer-dimension)
- [Outer Product](#outer-product)
- [Pinned Memory](#pinned-memory)
- [Pipeline](#pipeline)
- [Policy](#policy)
- [Problem](#problem)
- [Processing Units](#processing-units)
- [Reference Kernel](#reference-kernel)
- [Regression Test](#regression-test)
- [ROCm](#rocm)
- [Scalar General Purpose Register (SGPR)](#scalar-general-purpose-register-sgpr)
- [Shared Memory / LDS (Local Data Share)](#shared-memory--lds-local-data-share)
- [SIMT / SIMD](#simt--simd)
- [Smoke Test](#smoke-test)
- [Sparse Tensor](#sparse-tensor)
- [Split-K GEMM](#split-k-gemm)
- [Store Tile](#store-tile)
- [Thread / Work-item](#thread--work-item)
- [Thread Block / Work Group](#thread-block--work-group)
- [Vanilla GEMM](#vanilla-gemm)
- [Tile](#tile)
- [Tile Distribution](#tile-distribution)
- [Tile Partitioner](#tile-partitioner)
- [Tile Programming API](#tile-programming-api)
- [Tile Window](#tile-window)
- [User Customized Tile Pipeline](#user-customized-tile-pipeline)
- [User Customized Tile Pipeline Optimization](#user-customized-tile-pipeline-optimization)
- [Vector](#vector)
- [Vector General Purpose Register (VGPR)](#vector-general-purpose-register-vgpr)
- [Warp / Wavefront](#warp--wavefront)
- [Wave Tile](#wave-tile)
- [XDL Instructions](#xdl-instructions)

---

## 1. Hardware and Memory

### Processing Units
The GPU is composed of multiple hardware units ([compute units (CUs)](#compute-unit-cu) on AMD, [streaming multiprocessors (SMs)](#compute-unit-cu) on NVIDIA), each containing many cores that run threads in parallel. These units manage shared resources and coordinate execution at scale.

### Matrix Core
Specialized GPU units that accelerate matrix operations for AI and deep learning tasks. Modern GPUs contain multiple matrix cores.

### Compute Unit (CU)
AMD's parallel vector processor in a GPU with multiple ALUs. Each compute unit will run all the waves in a workgroup. _This is equivalent to NVIDIA's streaming multiprocessor (SM)_.

### Matrix Fused Multiply-Add (MFMA)
AMD's matrix core instruction for efficient GEMM operations. CK optimizes kernel designs to maximize MFMA utilization and performance.

### Registers
The fastest memory tier, registers are private to each thread/work-item and used for storing temporary variables during computation. AMD distinguishes between [vector (VGPR)](#vector-general-purpose-register-vgpr) and [scalar (SGPR)](#scalar-general-purpose-register-sgpr) registers, while NVIDIA uses a unified register file.

### Vector General Purpose Register (VGPR)
Per-thread registers that store individual thread data within a wave. Each thread has its own set of VGPRs for private variables and calculations.

### Scalar General Purpose Register (SGPR)
Wave-level registers shared by all threads in a wave. Used for constants, addresses, and control flow common across the entire wave.

### Shared Memory / Local Data Share (LDS)
AMD's high-bandwidth, low-latency on-chip memory accessible to all threads within a work group. This is equivalent to NVIDIA's shared memory. It enables fast data sharing and synchronization, but is limited in capacity and must be managed to avoid [bank conflicts](#bank-conflict).

### LDS Banks
Memory organization where consecutive addresses are distributed across multiple memory banks for parallel access. Prevents memory access conflicts ([bank conflicts](#bank-conflict)) and improves bandwidth.

### Global Memory
The main device memory accessible by all threads, offering high capacity but higher latency than shared memory.

### Pinned Memory
Host memory that is page-locked to accelerate transfers between CPU and GPU, reducing overhead for large data movements.

### Dense Tensor
A tensor in which most elements are nonzero, typically stored in a contiguous block of memory.

### Sparse Tensor
A tensor in which most elements are zero, allowing for memory and computation optimizations by storing only nonzero values and their indices.

### Host
CPU and main memory system that manages GPU execution. Launches kernels, transfers data, and coordinates overall computation.

### Device
GPU hardware that executes parallel kernels. Contains compute units, memory hierarchy, and specialized accelerators.

---

## 2. GPU Programming Model

### Thread / Work-item
AMD's work-item is the smallest unit of parallel execution, each running an independent instruction stream on a single data element. This is equivalent to NVIDIA's thread. Work-items/threads are grouped into [wavefronts (AMD)](#warp--wavefront) and [warps (NVIDIA)](#warp--wavefront) for efficient scheduling and resource sharing.

### Warp / Wavefront
AMD's wavefront is a group of threads that run instructions in lockstep, forming the SIMD group. This is equivalent to NVIDIA's warp.

### Thread Block / Work Group
AMD's work group is a collection of threads/work-items that can synchronize and share memory. This is equivalent to NVIDIA's thread block. Work groups/thread blocks are scheduled independently and mapped to hardware units for execution.

### Grid
The complete collection of all work groups (thread blocks) that execute a kernel. A grid spans the entire computational domain and is organized in 1D, 2D, or 3D dimensions. Each work group within the grid operates independently and can be scheduled on different compute units, enabling massive parallel execution across the entire GPU.

### Block Size
Number of work-items/threads in a compute unit (CU). Determines work group size and memory usage.

### Single-Instruction, Multi-Thread (SIMT) / Single-Instruction, Multi-Data (SIMD)
SIMT (Single-Instruction, Multi-Thread) allows threads in a warp to diverge, while SIMD (Single-Instruction, Multi-Data) enforces strict lockstep execution within wavefronts. These models define how parallelism is expressed and managed on different architectures.

### Occupancy
The ratio of active warps/wavefronts to the maximum number of warps/wavefronts supported by a hardware unit. Affects the ability to hide memory latency and maximize throughput.

---

## 3. Kernel Structure

### Kernel
A function executed on the GPU, typically written in [HIP](#hip) or [CUDA](#cuda), that performs parallel computations over input data. Kernels are launched with specific grid and block dimensions to map computation to hardware. In CK, kernels are composed from pipelines and require a pipeline, tile partitioner, and epilogue component.

### Pipeline
A CK Pipeline orchestrates the sequence of operations for a kernel, including data loading, computation, and storage phases. It consists of two core components: a [Problem](#problem) component that defines what to compute, and a [Policy](#policy) component that specifies how to move data around. 

### Tile Partitioner
Defines the mapping between problem dimensions (M, N, K) and GPU hierarchy. It specifies workgroup-level tile sizes (kM, kN, kK) and determines grid dimensions by dividing the problem size by tile sizes.

### Problem
Defines what to compute - input/output shapes, data types, and mathematical operations (e.g., GEMM, convolution).

### Policy
Defines memory access patterns and hardware-specific optimizations.

### User Customized Tile Pipeline
User-defined pipeline that combines custom problem and policy components for specialized computations. CK also provides prebuilt pipelines and policies for common operations that can be used as starting points.

### User Customized Tile Pipeline Optimization
Process of tuning tile sizes, memory access patterns, and hardware utilization for specific workloads. CK also provides prebuilt pipelines and policies for common operations that can be used as starting points.

### Tile Programming API
CK's high-level interface for defining tile-based computations with predefined hardware mapping for data load/store.

### Coordinate Transformation Primitives
CK utilities for converting between different coordinate systems (logical, physical, memory layouts).

### Reference Kernel
A baseline kernel implementation used to verify correctness and performance. CK has two reference kernel implementations: one for CPU and one for GPU.

### Launch Parameters
Configuration values (e.g., grid size, block size) that determine how a kernel is mapped to hardware resources. Proper tuning of these parameters is essential for optimal performance.

---

## 4. Memory Access and Data Layout

### Memory Coalescing
An optimization where consecutive threads access consecutive memory addresses, allowing a single memory transaction to serve multiple threads. Proper coalescing is vital for achieving peak memory bandwidth.

### Alignment
A memory management startegy for efficient memory access where data structures are stored at addresses that are multiples of a specific value.

### Bank Conflict
Occurs when multiple threads in a warp/wavefront access different addresses mapping to the same shared memory bank, causing serialization and reduced bandwidth.

### Padding
The addition of extra elements (often zeros) to tensor edges. This is used to control output size in convolution and pooling, or to align data for efficient memory access.

### Permute/Transpose
Operations that rearrange the order of tensor axes, often required to match kernel input formats or optimize memory access patterns.

### Host-Device Transfer
The process of moving data between CPU (host) and GPU (device) memory. Host-device transfers can be a performance bottleneck and are optimized using pinned memory and asynchronous operations.

### Stride
The step size to move from one element to the next in a particular dimension of a tensor or matrix. In convolution and pooling, stride determines how far the kernel moves at each step.

### Dilation
The spacing between kernel elements in convolution operations, allowing the receptive field to grow without increasing kernel size.

### Im2Col/Col2Im
Data transformation techniques that convert image data to column format (im2col) for efficient convolution and back (col2im) to reconstruct the original layout.

### Fast Changing Dimension
Innermost dimension that changes fastest in memory layout.

### Outer Dimension
Slower-changing dimension in memory layout.

### Inner Dimension
Faster-changing dimension in memory layout.

---

## 5. Tile-Based Computing and Data Structures

### Tile
A sub-region of a tensor or matrix processed by a block or thread. Tiles are used to improve memory locality and enable blocking strategies in kernels. Rectangular data blocks are the unit of computation and memory transfer in CK and the basis for tiled algorithms.

### Block Tile
Memory tile processed by a work group (thread block).

### Wave Tile
Sub-tile processed by a single wave within a work group. Represents the granularity of SIMD execution.

### Tile Distribution
Hierarchical data mapping from work-items to data in memory.

### Tile Window
Viewport into a larger tensor that defines the current tile's position and boundaries for computation.

### Load Tile
Operation that transfers data from global memory/LDS to per-thread registers using optimized memory access patterns.

### Store Tile
Operation that transfers data from per-thread registers to LDS/global memory using optimized memory access patterns.

### Descriptor
Metadata structure that defines tile properties, memory layouts, and coordinate transformations for CK operations.

### Input/Problem Shape
Dimensions and data types of input tensors that define the computational problem (e.g., M×K, K×N for GEMM).

### Vector
Smallest data unit processed by individual threads. Typically 4-16 elements depending on data type and hardware.

---

## 6. Kernel Operations and Optimization

### Elementwise
Operations applied independently to each tensor element, such as addition or multiplication. These are highly parallelizable and benefit from efficient memory access.

### Epilogue
The final stage of a kernel or operation, often applying activation functions, bias, or other post-processing steps. Epilogues are critical for integrating kernel outputs into larger computation graphs.

### Add+Multiply
A common fused operation in ML and linear algebra, where an elementwise addition is immediately followed by multiplication, often used for bias and scaling in neural network layers.

---

## 7. Linear Algebra and ML Operations

### General Matrix Multiply (GEMM)
Core matrix operation in linear algebra and deep learning. A GEMM is defined as C = αAB + βC for matrices A, B, and C. 

### "Vanilla" GEMM (Naive GEMM) Kernel
The **vanilla GEMM** is the simplest form of GEMM in CK. It:
- Takes input matrices **A** and **B**
- Multiplies them to produce output matrix **C**

This is the **baseline** or **building block** GEMM that all other complex versions expand upon.

### Grouped GEMM (GGEMMs)

A kernel which calls multiple VGEMMs. Each call can have a different input shape. Each input shape problem first finds its corresponding kernel and then data is mapped to the work-group (blocks) of that kernel. 

### Batched GEMM
A kernel which calls VGEMMs with different "batches" of data. All batches have the same input shape. 

### Split-K GEMM
A parallelization strategy that partitions the reduction dimension (K) across multiple compute units, increasing parallelism for large matrix multiplications.

### GEMV
The operation of multiplying a matrix by a vector, producing another vector. GEMV (General Matrix Vector Multiplication) is a core linear algebra primitive, widely used in neural networks and scientific computing.

### Inner Product
Also known as the dot product, it computes the sum of elementwise products of two vectors, yielding a scalar.

### Outer Product
The result of multiplying a column vector by a row vector, producing a matrix. Outer products are used in rank-1 updates and some ML algorithms.

### Norm
A function that measures the magnitude of a vector or matrix, such as L2 (Euclidean) or L1 norm. Norms are used in regularization, normalization, and optimization.

---

## 8. Testing, Build, and Infrastructure

### Regression Test
Tests that are part of CK's ctest suite and explicitly take more than 30s to finish on gfx942.

### Smoke Test
Tests that are part of CK's ctest suite and take less than or equal to 30 seconds to finish on gfx942.

---

## 9. Low-Level Instructions and Optimizations

### eXtensible Data Language (XDL) Instructions
eXtensible Data Language (XDL) instructions are a set of specialized, low-level instructions used to optimize data movement, memory access, and layout in high-performance computing, GPU programming, and deep learning tasks.

---

## 10. Miscellaneous

### HIP
AMD's Heterogeneous-Computing Interface for Portability, a C++ runtime API and programming language that enables developers to create portable applications for AMD and NVIDIA GPUs. HIP provides a familiar CUDA-like programming model while maintaining compatibility across different GPU architectures.

### CUDA
NVIDIA's Compute Unified Device Architecture, a parallel computing platform and programming model for NVIDIA GPUs. CUDA provides a C++ extension for writing GPU kernels and managing GPU resources.

### ROCm
AMD's Radeon Open Compute platform, an open-source software stack for GPU computing that includes [HIP](#hip), libraries, and tools for high-performance computing and machine learning workloads on AMD GPUs.

---

## Scientific Context and References

This terminology is grounded in parallel computing theory, numerical linear algebra, and computer architecture. For further reading, see:
- [Building Efficient GEMM Kernels with CK Tile](https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html)
- [CK Tile Flash](https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html)

This document assumes familiarity with parallel computing, linear algebra, and computer architecture principles.
