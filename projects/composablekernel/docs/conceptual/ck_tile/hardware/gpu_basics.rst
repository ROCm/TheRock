.. meta::
   :description: Introduction to AMD CDNA Architecture for CK developers
   :keywords: CDNA, RDNA, ROCm, CK, Composable Kernel, GPU architecture, compute units

.. _ck_tile_gpu_basics:

********************************************************************
Intro to AMD CDNA Architecture
********************************************************************

The AMD CDNA architecture is a specialized GPU design for high-performance computing (HPC) and AI workloads. Unlike the RDNA architecture used in gaming GPUs, CDNA is optimized for data center tasks, prioritizing compute density, memory bandwidth, and scalability. This is achieved through several key architectural features.

For more information about the AMD GPU architecture, see the `GPU architecture documentation <https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch.html>`_.

Implications for CK Tile
========================

Understanding the CDNA architecture is crucial for effective use of CK Tile:

1. **Thread Organization**: CK Tile's hierarchical :ref:`ck_tile_thread_mapping` (blocks → warps → threads) directly maps to CDNA's hardware organization.

2. **Memory Hierarchy**: CK Tile's :ref:`ck_tile_buffer_views` and :ref:`ck_tile_tile_window` are designed to efficiently utilize the L2, Infinity Cache, and LDS hierarchy.

3. **Register Pressure**: CK Tile's compile-time optimizations help minimize VGPR usage, preventing spills to slower memory.

4. **Warp Execution**: CK Tile's :ref:`ck_tile_tile_distribution` ensures that threads within a warp access contiguous memory for optimal SIMD execution.

5. **LDS Utilization**: CK Tile's :ref:`ck_tile_static_distributed_tensor` and :ref:`ck_tile_tile_window` make effective use of the 64KB LDS per CU.

By understanding these architectural features, developers can better appreciate how CK Tile's abstractions map to hardware capabilities and why certain design decisions were made in the framework.

Related Topics

- :ref:`ck_tile_thread_mapping` - How threads are organized and mapped to hardware
- :ref:`ck_tile_coordinate_systems` - Mathematical foundation for data distribution
- :ref:`ck_tile_lds_bank_conflicts` - Optimizing shared memory access patterns
- :ref:`ck_tile_load_store_traits` - Memory access optimization strategies
- :ref:`ck_tile_gemm_optimization` - Practical application of architecture knowledge
