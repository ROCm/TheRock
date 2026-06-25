.. meta::
  :description: Composable Kernel glossary of terms
  :keywords: composable kernel, glossary

***************************************************
Composable Kernel glossary

***************************************************

.. glossary::
    :sorted:

    arithmetic logic unit
        The arithmetic logic unit (ALU) is the GPU component responsible for arithmetic and logic operations.

    compute unit
        The compute unit (CU) is the parallel vector processor in an AMD GPU with multiple :term:`ALUs<arithmetic logic unit>`. Each compute unit will run all the :term:`wavefronts<wavefront>` in a :term:`work group>`. A compute unit is equivalent to NVIDIA's streaming   multiprocessor.

    matrix core
        A matrix core is a specialized GPU unit that accelerate matrix operations for AI and deep learning tasks. A GPU contains multiple matrix cores.

    register
        Registers are the fastest tier of memory. They're used for storing temporary values during computations and are private to the :term:`work-items<work-item>` that use them.

    VGPR
        See :term:`vector general purpose register`.

    vector general purpose register 
        A vector general purpose register (VGPR) is a :term:`register` that stores individual thread data. Each thread in a :term:`wave<wavefront>` has its own set of VGPRs for private variables and calculations. 

    SGPR
        See :term:`scalar general purpose register`.

    scalar general purpose register
        A scalar general purpose register (SGPR) is a :term:`register` shared by all the :term:`work items<work item>` in a :term:`wave<wavefront>`. SGPRs are used for constants, addresses, and control flow common across the entire wave.

    LDS
        See :term:`local data share`.

    local data share
        Local data share (LDS) is high-bandwidth, low-latency on-chip memory accessible to all the :term:`work-items<work-item>` in a :term:`work group`. LDS is equivalent to NVIDIA's shared memory. 

    LDS banks
        LDS banks are a type of memory organization where consecutive addresses are distributed across multiple memory banks for parallel access. LDS banks are used to prevent memory access conflicts and improve bandwidth when LDS is used.

    global memory
        The main device memory accessible by all threads, offering high capacity but higher latency than shared memory.

    pinned memory
        Pinned memory is :term:`host` memory that is page-locked to accelerate transfers between the CPU and GPU.

    dense tensor
        A dense tensor is a tensor where most of its elements are non-zero. Dense tensors are typically stored in a contiguous block of memory.

    sparse tensor
        A sparse tensor is a tensor where most of its elements are zero. Typically only the non-zero elements of a sparse tensor and their indices are stored.

    host
        Host refers to the CPU and the main memory system that manages GPU execution. The host is responsible for launching kernels, transferring data, and coordinating overall computation.

    device
        Device refers to the GPU hardware that runs parallel kernels. The device contains the :term:`compute units<compute unit>`, memory hierarchy, and specialized accelerators.

    work-item
        A work-item is the smallest unit of parallel execution. A work-item runs a single independent instruction stream on a single data element. A work-item is equivalent to an NVIDIA thread.

    wavefront
        Also referred to as a wave, a wavefront is a group of :term:`work-items<work-item>` that run the same instruction. A wavefront is equivalent to an NVIDIA warp.

    work group
        A work group is a collection of :term:`work-items<work-item>` that can synchronize and share memory. A work group is equivalent to NVIDIA's thread block. 

    grid
        A grid is a collection of :term:`work groups<work group>` that run a kernel. Each work group within the grid operates independently and can be scheduled on a different :term:`compute unit`. A grid can be organized into one, two, or three dimensions. A grid is equivalent to an NVIDIA thread block.

    block Size
        The block size is the number of :term:`work-items<work-item>` in a :term:`compute unit`.

    SIMT
        See :term:`single-instruction, multi-thread`

    single-instruction, multi-thread 
        Single-instruction, multi-thread (SIMT) is a parallel computing model where all the :term:`work-items<work-item>` within a :term:`wavefront` run the same instruction on different data. 

    SIMD
        See :term:`single-instruction, multi-data`

    single-instruction, multi-data
        Single-instruction, multi-data (SIMD) is a parallel computing model where the same instruction is run with different data simultaneously. 

    occupancy
        The ratio of active :term:`wavefronts<wavefront>` to the maximum possible number of wavefronts.

    kernel
        A kernel is a function that runs an :term:`operation` or a collection of operations. A kernel will run in parallel on several :term:`work-items<work-item>` across the GPU. In Composable Kernel, kernels require :term:`pipelines<pipeline>`.

    operation
        An operation is a computation on input data. 
        
    pipeline
        A Composable Kernel pipeline schedules the sequence of operations for a :term:`kernel`, such as the data loading, computation, and storage phases. A pipeline consists of a :term:`problem` and a :term:`policy`. 

    tile partitioner
        The tile partitioner defines the mapping between the :term:`problem` dimensions and GPU hierarchy. It specifies :term:`workgroup`-level :term:`tile` sizes and determines :term:`grid` dimensions by dividing the problem size by the tile sizes.

    problem
        The problem is the part of the :term:`pipeline` that defines input and output shapes, data types, and mathematical :term:`operations<operation>`.

    policy
        The policy is the part of the :term:`pipeline` that defines memory access patterns and hardware-specific optimizations.

    user customized tile pipeline
        A customized :term:`tile` :term:`pipeline` that combines custom :term:`problem` and :term:`policy` components for specialized computations. 

    user customized tile pipeline optimization
        The process of tuning the :term:`tile` size, memory access pattern, and hardware utilization for specific workloads.

    tile programming API
        The :term:`tile` programming API is Composable Kernel's high-level interface for defining tile-based computations with predefined hardware mappings for data loading and storing.

    coordinate transformation primitives
        Coordinate transformation primitives are Composable Kernel utilities for converting between different coordinate systems.

    reference kernel
        A reference :term:`kernel` is a baseline kernel implementation used to verify correctness and performance. Composable Kernel makes two reference kernels, one for CPU and one for GPU, available.

    launch parameters
        Launch parameters are the configuration values, such as :term:`grid` and :term:`block size`, that determine how a :term:`kernel` is mapped to hardware resources.

    memory coalescing
        Memory coalescing is an optimization strategy where consecutive :term:`work-items<work-item>` access consecutive memory addresses in such a way that a single memory transaction serves multiple work-items.

    alignment
        Alignment is a memory management strategy where data structures are stored at addresses that are multiples of a specific value.


    bank conflict
        A bank conflict occurs when multiple :term:`work-items<work-item>` in a :term:`wavefront` access different addresses that map to the same shared memory bank.

    padding
        Padding is the addition of extra elements, often zeros, to tensor edges in order to control output size in convolution and pooling, or to align data for memory access.

    transpose
        Transpose is an :term:`operation` that rearranges the order of tensor axes, often for the purposes of matching :term:`kernel` input formats or optimize memory access patterns.

    permute
        Permute is an :term:`operation` that rearranges the order of tensor axes, often for the purposes of matching :term:`kernel` input formats or optimize memory access patterns.

    host-device transfer
        A host-device transfer is the process of moving data between :term:`host` and :term:`device` memory. 

    stride
        A stride is the step size to move from one element to the next in a specific dimension of a tensor or matrix. In convolution and pooling, the stride determines how far the :term:`kernel` moves at each step.

    dilation
        Dilation is the spacing between :term:`kernel` elements in convolution :term:`operations<operation>`, allowing the receptive field to grow without increasing kernel size.

    Im2Col
        Im2Col is a data transformation technique that converts image data to column format.

    Col2Im
        Col2Im is a data transformation technique that converts column data to image format.

    fast changing dimension
        The fast changing dimension is the innermost dimension in memory layout.

    outer dimension
        The outer dimension is the slower-changing dimension in memory layout.

    inner dimension
        The inner dimension is the faster-changing dimension in memory layout.

    tile
        A tile is a sub-region of a tensor or matrix that is processed by a :term:`work group` or :term:`work-item`. Rectangular data blocks are the unit of computation and memory transfer in Composable Kernel, and are the basis for tiled algorithms.

    block tile
        A block tile is a memory :term:`tile` processed by a :term:`work group`.

    wave tile
        A wave :term:`tile` is a sub-tile processed by a single :term:`wavefront` within a :term:`work group`. The wave tile is the base level granularity of a :term:`single-instruction, multi-thread (SIMD)<single-instruction, multi-thread>` model.

    tile distribution
        The tile distribution is the hierarchical data mapping from :term:`work-items<work-item>` to data in memory.

    tile window
        Viewport into a larger tensor that defines the current tile's position and boundaries for computation.

    load tile
        Load tile is an operation that transfers data from :term:`global memory` or the :term:`load data share` to :term:`vector general purpose registers<vector general purpose register>`.

    store tile
        Store tile is an operation that transfers data from  :term:`vector general purpose registers<vector general purpose register>` to :term:`global memory` or the :term:`load data share`.

    descriptor
        Metadata structure that defines :term:`tile` properties, memory layouts, and coordinate transformations for Composable Kernel :term:`operations<operation>`.

    input
        See :term:`problem shape`.

    problem shape
        The problem shape defines the dimensions and data types of input tensors that define the :term:`problem`.

    vector
        The vector is the smallest data unit processed by an individual :term:`work-item`. A vectors is typically four to sixteen elements, depending on data type and hardware.

    elementwise
        An elementwise :term:`operation` is an operation applied to each tensor element independently. 

    epilogue
        The epilogue is the final stage of a kernel. Activation functions, bias, and other post-processing steps are applied in the epilogue. 

    Add+Multiply
        See :term:`fused add multiply`.

    fused add multiply
        A common fused :term:`operation` in machine language and linear algebra, where an :term:`elementwise` addition is immediately followed by a multiplication. Fused add multiply is often used for bias and scaling in neural network layers.

    MFMA
        See :term:`matrix fused multiply-add`.

    matrix fused multiply-add
        Matrix fused multiply-add (MFMA) is a :term:`matrix core` instruction for GEMM :term:`operations<operation>`. 

    GEMM
        See :term:`general matrix multiply`.

    general matrix multiply 
        A general matrix multiply (GEMM) is a Core matrix :term:`operation` in linear algebra and deep learning. A GEMM is defined as :math:`C = {\alpha}AB + {\beta}C`, where :math:`A`, :math:`B`, and :math:`C` are matrices, and :math:`\alpha` and :math:`\beta` are scalars. 

    VGEMM
        See :term:`naive GEMM`.

    vanilla GEMM
        See :term:`naive GEMM`.

    naive GEMM 
        The naive GEMM, sometimes referred to as a vanilla GEMM or VGEMM, is the simplest form of :term:`GEMM` in Composable Kernel. The naive GEMM is defined as :math:`C = AB`, where :math:`A`, :math:`B`, and :math:`C` are matrices. The naive GEMM is the baseline GEMM that all other GEMM :term:`operations<operation>` build on.

    GGEMM
        See :term:`grouped GEMM`.

    grouped GEMM
        A :term:`kernel` that calls multiple :term:`VGEMMs<naive GEMM>`. Each call can have a different :term:`problem shape`. 

    batched GEMM
        A :term:`kernel` that calls :term:`VGEMMs<naive GEMM>` with different batches of data. All the data batches have the same :term:`problem shape`. 

    Split-K GEMM
        Split-K GEMM is a parallelization strategy that partitions the reduction dimension (K) of a :term:`GEMM` across multiple :term:`compute units<compute unit>`, increasing parallelism for large matrix multiplications.

    GEMV
        See :term:`general matrix vector multiplication`

    general matrix vector multiplication
        General matrix vector multiplication (GEMV) is an :term:`operation` where a matrix is multiplied by a vector, producing another vector. 

