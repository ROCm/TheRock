.. _ck_tile_terminology:

Terminology Reference - Key Concepts and Definitions
====================================================

Overview
--------

The Composable Kernel framework introduces concepts and abstractions that form the foundation of its approach to high-performance GPU computing. This terminology reference serves as a comprehensive guide to the language of CK, providing detailed explanations of each term along with practical examples of their usage in C++ code. 

The terminology of CK reflects its layered architecture, with concepts building upon one another in a logical progression. From the fundamental notion of tiles and distributions to the compile-time coordinate transformation systems, each term represents a carefully designed abstraction that serves a specific purpose in the overall framework. This reference is organized to mirror this conceptual hierarchy, starting with core concepts and progressing through increasingly specialized terminology.

As you explore this reference, you'll notice that many terms are interconnected, reflecting the holistic nature of the CK design. A tile is not just a block of data but a fundamental unit of work distribution. A distribution is not merely a pattern but a mathematical framework for optimal resource utilization. These interconnections are intentional and understanding them is crucial for effective use of the framework.

Core Concepts
-------------

Tile
~~~~
The concept of a tile represents the fundamental unit of data organization in the CK framework. A tile is a contiguous block of data that is processed as a cohesive unit by a coordinated group of threads. This abstraction serves multiple critical purposes in achieving high performance on GPU architectures. By organizing data into tiles, the framework ensures that memory accesses exhibit spatial locality, enabling efficient use of cache hierarchies. The tile size is chosen to balance several competing factors: it must be large enough to amortize the overhead of memory transactions, yet small enough to fit within the limited on-chip memory resources. Furthermore, tiles are designed to align with the :ref:`GPU's execution model <ck_tile_gpu_basics>`, ensuring that threads within a warp access contiguous memory locations for optimal bandwidth utilization.

**C++ Usage**: ``using TileShape = sequence<256, 256>;``

Distribution
~~~~~~~~~~~~
The distribution pattern represents one of the most compile-time abstractions in the CK framework, defining the precise mapping between logical data elements and the physical processing resources that will operate on them. A distribution is far more than an assignment scheme—it embodies a strategy for achieving optimal performance on GPU hardware. The distribution determines which threads access which data elements, how those accesses are ordered to maximize memory bandwidth, and how intermediate results are shared between cooperating threads. By encoding these decisions at compile time, distributions enable the generation of highly optimized code that respects hardware constraints while maintaining algorithmic clarity. For a detailed exploration of distribution concepts, see :ref:`ck_tile_distribution`.

**C++ Type**: ``tile_distribution<...>``

Encoding
~~~~~~~~
An encoding in CK represents a compile-time specification that captures the strategy for distributing tensor data across GPU processing elements. This specification is not merely a configuration but a mathematical description of the transformation between coordinate spaces. The encoding defines the hierarchical decomposition of work, the mapping between thread indices and data elements, and the patterns by which threads cooperate to process their assigned data. By expressing these concepts as compile-time constants, encodings enable aggressive compiler optimizations while ensuring that distribution strategies can be verified for correctness before execution.

**C++ Type**: ``tile_distribution_encoding<...>``

Coordinate Spaces
-----------------

For a comprehensive mathematical treatment of coordinate systems, see :ref:`ck_tile_coordinate_systems`.

P-Space (Partition Space)
~~~~~~~~~~~~~~~~~~~~~~~~~
The Partition Space, or P-space, represents the fundamental abstraction for identifying processing elements within the GPU's execution hierarchy. This coordinate space captures the multi-level organization of GPU computation, from individual threads to warps to thread blocks. P-space typically manifests as either a one-dimensional space containing only lane identifiers for simple distributions, or a two-dimensional space incorporating both warp and lane identifiers for more complex hierarchical distributions. The significance of P-space extends beyond mere thread identification—it forms the foundation for all work distribution decisions, determining which processing elements will collaborate on specific data tiles and how they will coordinate their efforts.

The dimensions of P-space directly reflect the hardware's execution model. In a one-dimensional P-space, threads are identified solely by their lane ID within a warp, suitable for algorithms where inter-warp coordination is minimal. Two-dimensional P-space adds warp-level coordination, enabling advanced tiling strategies that leverage both intra-warp and inter-warp parallelism. The values in P-space are always hardware thread indices, providing a direct mapping to the physical execution resources.

**C++ Example**:

.. code-block:: cpp

   // Get current thread's P coordinates
   auto p_idx = Distribution::_get_partition_index();

Y-Space (Yield Space)
~~~~~~~~~~~~~~~~~~~~~
The Yield Space, or Y-space, embodies the logical structure of computation within each tile, representing the pattern by which threads traverse their assigned data. Unlike P-space which identifies threads, Y-space defines what each thread does with its assigned work. This abstraction enables the expression of complex access patterns—from simple linear traversals to advanced space-filling curves—in a hardware-independent manner. The dimensionality of Y-space varies with the algorithm's requirements, typically ranging from two dimensions for matrix operations to four or more for complex tensor contractions.

Y-space serves as the primary iteration space for computational kernels. When a thread processes its assigned tile, it iterates through Y-space coordinates, with each coordinate mapping to specific data elements within the tile. This abstraction enables critical optimizations: the Y-space traversal order can be designed to maximize data reuse, minimize register pressure, or optimize for specific hardware characteristics, all without changing the fundamental algorithm.

**C++ Example**:

.. code-block:: cpp

   // Iterate over Y-space
   sweep_tile(tensor, [](auto y_idx) { /*...*/ });

X-Space (Physical Tensor Space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Physical Tensor Space, or X-space, represents the ground truth of data organization—the actual coordinates within the global tensor. This space directly corresponds to how data is laid out in memory, with dimensions matching those of the tensor being processed. For a matrix, X-space is two-dimensional with row and column coordinates. For a 4D convolution tensor, X-space encompasses batch, channel, height, and width dimensions. X-space serves as the target of the coordinate transformation pipeline, where abstract thread and pattern coordinates are converted into concrete memory addresses.

The relationship between X-space and physical memory is direct but not necessarily trivial. While X-space coordinates identify logical positions within a tensor, the actual memory layout may involve padding, striding, or other transformations for alignment and performance. The CK framework handles these low-level details transparently, allowing algorithms to work with logical X-space coordinates while ensuring efficient physical memory access.

**C++ Example**:

.. code-block:: cpp

   // Calculate X coordinates from P+Y
   auto x_idx = distribution.calculate_index(p_idx);

R-Space (Replication Space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Replication Space, or R-space, introduces a advanced mechanism for expressing redundant computation patterns that enhance performance through data sharing. Unlike the other coordinate spaces which map to unique data elements, R-space enables multiple processing elements to compute the same values, facilitating efficient communication patterns. This replication serves multiple purposes: it can reduce global memory traffic by computing values locally rather than loading them, enable efficient reduction operations by providing private workspace for each thread group, and facilitate complex data exchange patterns that would otherwise require expensive synchronization.

R-space dimensions are optional and algorithm-specific. A matrix multiplication might use R-space to replicate portions of the input matrices across thread groups, enabling each group to compute partial products independently. The framework automatically manages the complexities of replication, including the allocation of private storage and the coordination of replicated computations.

**C++ Example**:

.. code-block:: cpp

   // R-dimensions in encoding
   using Encoding = tile_distribution_encoding<
       sequence<2>,  // rs_lengths: 2-way replication
       /*...*/
   >;

D-Space (Data Space)
~~~~~~~~~~~~~~~~~~~~
The Data Space, or D-space, represents the final stage of the coordinate transformation pipeline—the linearization of multi-dimensional tile data for efficient storage in thread-local registers. This one-dimensional space serves a critical role in managing the GPU's most precious resource: register files. By transforming the potentially complex Y-space coordinates into a linear D-space index, the framework enables efficient register allocation and access patterns that minimize register bank conflicts and maximize instruction-level parallelism.

The transformation from Y-space to D-space is more than a simple flattening operation. It incorporates optimized strategies for register layout that consider the GPU's register file organization, the kernel's register pressure, and the access patterns of the computation. This transformation ensures that frequently accessed elements are kept in registers, that register bank conflicts are minimized, and that the compiler can generate efficient code for register access.

**C++ Example**:

.. code-block:: cpp

   // Y-to-D descriptor linearizes storage
   auto d_idx = ys_to_d_descriptor.calculate_offset(y_idx);

Dimension Types
---------------

H-Dimensions (Hierarchical Dimensions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The concept of Hierarchical Dimensions, or H-dimensions, represents one of the most key aspects of the CK framework's approach to work distribution. These dimensions encode a multi-level decomposition strategy that mirrors the hierarchical nature of GPU hardware, from individual vector operations up through threads, warps, and thread blocks. Each H-dimension group captures how a single tensor dimension is partitioned across these hardware levels, enabling fine-grained control over data access patterns and computational efficiency.

The structure of H-dimensions follows a specific pattern that reflects the GPU's execution hierarchy. Each H-dimension is expressed as a sequence of factors, where each factor corresponds to a specific level of the hierarchy. Consider the example ``sequence<4, 2, 8, 4>``. This seemingly simple sequence encodes a advanced distribution strategy: the rightmost factor (4) represents vector width, indicating that each memory operation processes 4 elements simultaneously. Moving left, the factor 8 indicates that 8 threads within a warp collaborate on the data. The factor 2 specifies that 2 warps within a block work together. Finally, the leftmost factor 4 indicates that each thread performs 4 iterations, enabling instruction-level parallelism and register reuse.

This hierarchical decomposition enables critical optimizations. By explicitly encoding the distribution strategy at compile time, the framework can generate code that perfectly matches the hardware's capabilities. The vector width aligns with the GPU's memory transaction size. The thread count per warp matches the hardware's SIMD width. The warp count per block balances parallelism with resource constraints. The repetition factor enables loop unrolling and software pipelining. Together, these factors create a distribution strategy that achieves near-optimal performance.

**C++ Example**:

.. code-block:: cpp

   using HsLengthss = tuple<
       sequence<4, 2, 8, 4>,  // H0: M dimension
       sequence<4, 2, 8, 4>   // H1: N dimension
   >;

RH-Dimensions (R + H Dimensions Combined)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The RH-dimensions represent the unified coordinate space that combines both replication (R) and hierarchical (H) dimensions into a single, coherent framework. This combined space serves as the internal representation used by the coordinate transformation machinery, enabling seamless handling of both replicated and non-replicated data patterns. The unification of these dimensions simplifies the mathematical framework while maintaining the flexibility to express complex distribution strategies.

Within the RH-dimension framework, coordinates are identified by two components: major and minor indices. The major index identifies which dimension group a coordinate belongs to, with 0 reserved for R-dimensions and subsequent values (1, 2, ...) identifying H-dimension groups. The minor index specifies the position within the identified group. This two-level addressing scheme enables efficient navigation through the combined coordinate space while maintaining clear separation between replication and hierarchical decomposition strategies.

The power of RH-dimensions becomes apparent when considering complex algorithms that require both data replication and hierarchical distribution. By providing a unified coordinate system, the framework can express transformations that simultaneously handle replicated data sharing and hierarchical work distribution, all within a single mathematical formalism. This unification is key to achieving both expressiveness and efficiency in the CK framework.

Transformations
---------------

Adaptor
~~~~~~~
An adaptor in the CK framework represents a advanced chain of coordinate transformations that bridges different coordinate spaces. Rather than simple one-to-one mappings, adaptors embody complex mathematical transformations that can involve permutations, embeddings, projections, and non-linear mappings. These transformations are composed at compile time, enabling the generation of highly optimized code that performs the complete transformation in a single step without intermediate representations. For detailed information about adaptors and their implementation, see :ref:`ck_tile_adaptors`.

The framework provides several specialized adaptor types, each serving a specific role in the coordinate transformation pipeline. The ``ps_ys_to_xs_adaptor`` performs the critical transformation from processing element and yield space coordinates to physical tensor coordinates, implementing the core logic of tile distribution. This adaptor encodes decisions about how threads are assigned to data, how data is traversed within each thread's assignment, and how these patterns map to the global tensor layout. Similarly, the ``ys_to_d_adaptor`` handles the transformation from multi-dimensional yield space to linearized data space, optimizing the layout of data in thread-local registers.

The power of adaptors lies in their composability. Complex transformations can be built by chaining simpler adaptors, with the framework automatically optimizing the composition. This design enables the expression of advanced access patterns—such as transposed access, strided access, or space-filling curves—through the composition of elementary transformations. The compile-time nature of this composition ensures zero runtime overhead while maintaining mathematical clarity.

**C++ Type**: ``tensor_adaptor<...>``

Descriptor
~~~~~~~~~~
A descriptor in CK provides a complete specification of tensor layout, encompassing not just the logical structure of the data but also all transformations and physical memory layout details. This comprehensive specification serves as the contract between different components of the system, ensuring that all parts of a kernel have a consistent view of how data is organized and accessed. Descriptors combine multiple aspects of tensor representation: the logical shape and dimensions, the physical memory layout including padding and alignment, the coordinate transformations for different access patterns, and optimization hints for the compiler. For comprehensive coverage of descriptors, see :ref:`ck_tile_descriptors`.

The sophistication of descriptors enables them to represent complex data layouts that arise in real-world applications. A descriptor might specify that a logically 4D tensor is physically stored with padding for alignment, uses a custom stride pattern for the channel dimension, and should be accessed using a space-filling curve for optimal cache utilization. All these details are encoded in the descriptor's type, enabling compile-time verification and optimization.

Descriptors play a crucial role in achieving performance portability. By abstracting the details of data layout behind a well-defined interface, descriptors enable algorithms to be written once and automatically adapted to different data layouts. This abstraction is particularly valuable when dealing with different hardware architectures that may have different alignment requirements, cache line sizes, or memory access patterns.

**C++ Type**: ``tensor_descriptor<...>``

Operations
----------

Load Tile
~~~~~~~~~
The load tile operation represents a fundamental building block of GPU kernel design in the CK framework, orchestrating the complex process of transferring data from global memory to thread-local registers. This operation is far more advanced than a simple memory copy—it implements the complete distribution strategy encoded in the tile distribution, ensuring that each thread loads exactly the data it needs for its portion of the computation. The load operation automatically handles memory coalescing to maximize bandwidth utilization, coordinates between threads to avoid redundant loads, manages boundary conditions for tiles that extend beyond tensor bounds, and optimizes the access pattern based on the specific distribution strategy.

The efficiency of the load tile operation stems from its deep integration with the distribution framework. By knowing at compile time exactly which threads will access which data elements, the operation can generate optimal memory access patterns that fully utilize the GPU's memory subsystem. For matrix multiplication, this might mean loading data in a pattern that ensures perfect coalescing. For convolution, it might involve complex patterns that minimize the number of redundant loads while respecting the GPU's cache hierarchy.

**C++ Function**: ``tile_window.load()``

Store Tile
~~~~~~~~~~
The store tile operation provides the complementary functionality to load tile, transferring computed results from thread-local registers back to global memory. Like its counterpart, the store operation implements optimized strategies that go beyond simple memory writes. It ensures that writes are coalesced for maximum bandwidth efficiency, coordinates between threads to handle overlapping write regions correctly, manages atomic operations when multiple threads write to the same location, and optimizes write patterns to minimize memory traffic.

The store operation must handle additional complexities compared to loads. While loads can often ignore synchronization issues (reading stale data is usually harmless), stores must ensure correctness when multiple threads write to overlapping regions. The framework provides different store modes for different scenarios: exclusive stores where each element is written by exactly one thread, atomic stores where multiple threads may update the same element, and reduction stores where partial results are accumulated. The choice of store mode is encoded in the distribution strategy and verified at compile time.

**C++ Function**: ``tile_window.store(tile)``

Sweep Tile
~~~~~~~~~~
The sweep tile operation embodies a key programming paradigm for distributed tensor computation, providing a high-level iteration abstraction over the complex distribution patterns. Rather than requiring manual index calculations and nested loops, sweep tile automatically visits each element in a distributed tensor exactly once, invoking a user-provided function with the appropriate coordinates. This abstraction hides the complexity of the distribution while enabling advanced optimizations such as automatic loop unrolling, software pipelining, and register rotation.

The implementation of sweep tile leverages the compile-time knowledge of the distribution pattern to generate highly optimized iteration code. For simple distributions, this might result in a single unrolled loop. For complex hierarchical distributions, it might generate nested loops with carefully chosen iteration orders that maximize data reuse and minimize register pressure. The beauty of the abstraction is that these optimizations happen transparently—the user simply provides the computation to perform on each element, and the framework handles the rest.

**C++ Function**: ``sweep_tile(tensor, lambda)``

Shuffle Tile
~~~~~~~~~~~~
The shuffle tile operation provides efficient intra-warp communication, enabling threads within a warp to exchange data without going through shared memory. This operation leverages the GPU's hardware shuffle instructions, which allow any thread in a warp to read registers from any other thread in the same warp. Shuffle operations are particularly valuable for reduction operations, transpose operations within a warp, and collaborative loading patterns where threads cooperate to load contiguous data and then redistribute it according to the computation pattern.

The framework provides various shuffle patterns optimized for different use cases. Butterfly shuffles enable efficient reductions and FFT-like operations. Broadcast shuffles allow one thread to share data with all others in the warp. Rotation shuffles enable cyclic data exchange patterns. The shuffle tile operation automatically selects the appropriate hardware instructions based on the data type and shuffle pattern, ensuring optimal performance while maintaining portability across different GPU architectures.

**C++ Function**: ``shuffle_tile(tensor, shuffle_pattern)``

Memory Concepts
---------------

Coalescing
~~~~~~~~~~
The property where adjacent threads access adjacent memory locations, maximizing memory bandwidth utilization.

Bank Conflict
~~~~~~~~~~~~~
A performance degradation that occurs when multiple threads in a warp access different addresses in the same memory bank. For detailed information about bank conflicts and mitigation strategies, see :ref:`ck_tile_lds_bank_conflicts`.

Vectorization
~~~~~~~~~~~~~
The technique of loading/storing multiple elements in a single memory transaction.

**C++ Example**:

.. code-block:: cpp

   // Vector load of 4 elements
   using float4 = vector_type<float, 4>::type;
   float4 data = tensor_view.template get_vectorized_elements<4>(x_idx);

Distribution Components
-----------------------

Window
~~~~~~
A view into a subset of a tensor that respects the distribution pattern. For detailed information about tile windows and their usage, see :ref:`ck_tile_tile_window`.

**C++ Type**: ``tile_window<...>``

Static Distributed Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~
A thread-local tensor stored in registers, distributed according to a tile distribution. For in-depth coverage of static distributed tensors, see :ref:`ck_tile_static_distributed_tensor`.

**C++ Type**: ``static_distributed_tensor<...>``

Spans
~~~~~
Iteration ranges over distributed dimensions, used by sweep operations.

**C++ Type**: ``tile_distributed_span<...>``

GPU Hardware Terms
------------------

Warp
~~~~
A group of threads (32 on AMD GPUs) that execute in lockstep.

Lane
~~~~
An individual thread within a warp (0-31).

Block
~~~~~
A group of warps that can cooperate through shared memory.

Grid
~~~~
The complete set of blocks launched for a kernel.

Template Parameters
-------------------

sequence<...>
~~~~~~~~~~~~~
A compile-time integer sequence used to specify dimensions and lengths.

**Example**: ``sequence<256, 256>`` for a 256×256 tile

tuple<...>
~~~~~~~~~~
A heterogeneous collection of types, often used for grouping sequences.

**Example**: ``tuple<sequence<4,4>, sequence<4,4>>``

number<N>
~~~~~~~~~
A compile-time integer constant.

**Example**: ``number<16>`` represents the value 16

Optimization Terms
------------------

Register Spilling
~~~~~~~~~~~~~~~~~
When a kernel uses more registers than available, causing data to spill to slower memory.

Occupancy
~~~~~~~~~
The ratio of active warps to maximum possible warps on a GPU multiprocessor.

Memory Bandwidth Utilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The percentage of theoretical memory bandwidth achieved by a kernel.

Instruction-Level Parallelism (ILP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ability to execute multiple independent instructions simultaneously.

Common Patterns
---------------

GEMM (General Matrix Multiplication)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A fundamental operation where C = αA×B + βC. For a complete optimization case study, see :ref:`ck_tile_gemm_optimization`.

Reduction
~~~~~~~~~
An operation that combines multiple values into a single result (e.g., sum, max).

Broadcast
~~~~~~~~~
An operation that replicates a value across multiple processing elements.

Transpose
~~~~~~~~~
An operation that swaps dimensions of a tensor.

Performance Metrics
-------------------

FLOPS (Floating-Point Operations Per Second)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Measure of computational throughput.

Bandwidth
~~~~~~~~~
Rate of data transfer, typically measured in GB/s.

Latency
~~~~~~~
Time delay between issuing an operation and its completion.

Throughput
~~~~~~~~~~
Rate of operation completion, often measured in operations per second.

Usage Examples
--------------

Creating a Distribution
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Define encoding
   using MyEncoding = tile_distribution_encoding<
       sequence<>,                        // No replication
       tuple<sequence<4,2,8,4>,          // M dimension
             sequence<4,2,8,4>>,         // N dimension
       tuple<sequence<1,2>, sequence<1,2>>, // P mappings
       tuple<sequence<1,1>, sequence<2,2>>, // P minor
       sequence<1,1,2,2>,                   // Y major
       sequence<0,3,0,3>                    // Y minor
   >;

   // Create distribution
   auto distribution = make_static_tile_distribution(MyEncoding{});

Using Tile Window
~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Create window
   auto window = make_tile_window(
       tensor_view,
       TileShape{},
       origin,
       distribution
   );

   // Load-compute-store pattern
   auto tile = window.load();
   sweep_tile(tile, compute_func);
   window.store(tile);

Related Documentation
---------------------

- :ref:`ck_tile_introduction` - Introduction and motivation
- :ref:`ck_tile_buffer_views` - Raw memory access
- :ref:`ck_tile_distribution` - Core distribution concepts


