.. meta::
   :description: CK Tile Hardware-Specific Documentation
   :keywords: CDNA, GPU architecture, LDS, GEMM, CK, Composable Kernel

.. _ck_tile_hardware:

********************************************************************
CK Tile Hardware Documentation
********************************************************************

This section provides in-depth coverage of hardware-specific concepts and optimizations for CK Tile on AMD GPUs.

Overview
========

Understanding the underlying hardware architecture is crucial for achieving optimal performance with CK Tile. This documentation covers:

- AMD CDNA architecture fundamentals
- Memory hierarchy and optimization techniques
- Practical examples of high-performance kernels

Documentation Structure
=======================

.. toctree::
   :maxdepth: 2
   :caption: Hardware Topics

   gpu_basics
   lds_bank_conflicts
   gemm_optimization

GPU Architecture Basics
-----------------------

:ref:`ck_tile_gpu_basics` provides an introduction to AMD CDNA architecture.

LDS and Bank Conflicts
----------------------

:ref:`ck_tile_lds_bank_conflicts` explains Local Data Share (LDS) optimization.

GEMM Optimization Case Study
----------------------------

:ref:`ck_tile_gemm_optimization` demonstrates a complete optimization journey.


Key Hardware Considerations
===========================


Memory Hierarchy
----------------

1. **Global Memory**: High latency, high bandwidth
   
   - Optimize with coalesced access patterns
   - Use tile windows for automatic optimization

2. **L2/Infinity Cache**: Intermediate storage
   
   - Benefits from spatial and temporal locality
   - CK Tile's tiling naturally improves cache hit rates

3. **LDS**: Low latency, shared within CU
   
   - 64KB per CU, organized in 32 banks
   - CK Tile handles bank conflict avoidance

4. **Registers**: Lowest latency, per-thread storage
   
   - 512 VGPRs available per wavefront
   - CK Tile's compile-time optimization minimizes usage

Compute Resources
-----------------

1. **Wavefront Execution**: 64 threads in lockstep
   
   - CK Tile ensures coalesced memory access
   - Automatic warp-level synchronization

2. **Matrix Units**: Specialized MFMA instructions
   
   - 16x16x16 operations in 16 cycles
   - CK Tile can leverage these automatically

3. **Occupancy**: Balancing threads vs resources
   
   - Register pressure affects occupancy
   - CK Tile helps through efficient register use

Performance Guidelines
======================

To achieve optimal performance with CK Tile:

1. **Choose appropriate tile sizes**:
   
   - Match hardware capabilities (e.g., 256x256 for GEMM)
   - Consider LDS capacity and register pressure

2. **Align problem dimensions**:
   
   - Match CU count when possible (304 for MI300)
   - Use padding for non-aligned sizes

3. **Enable pipelining**:
   
   - Use double buffering for latency hiding
   - CK Tile supports async operations

4. **Profile and verify**:
   
   - Use rocprof to check for bottlenecks
   - Verify bank conflict avoidance
   - Monitor occupancy and register usage

Next Steps
==========

- Review :ref:`ck_tile_gpu_basics` for architecture fundamentals
- Study :ref:`ck_tile_lds_bank_conflicts` for shared memory optimization
- Explore :ref:`ck_tile_gemm_optimization` for a complete optimization example

For practical implementation, refer back to the main :ref:`ck_tile_conceptual` documentation to see how these hardware concepts integrate with CK Tile's abstractions.
