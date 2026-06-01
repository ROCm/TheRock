.. meta::
   :description: Understanding AMD GPU LDS and Bank Conflicts in CK Tile
   :keywords: LDS, bank conflicts, shared memory, CK, Composable Kernel, GPU optimization

.. _ck_tile_lds_bank_conflicts:

********************************************************************
Understanding AMD GPU LDS and Bank Conflicts
********************************************************************

Introduction
============

Local Data Share (**LDS**) is AMD's shared memory within a compute unit (see :ref:`ck_tile_gpu_basics` for architecture details). It is organized into **32 or 64 banks** depending on the hardware architecture, each bank has a 4 bytes width. Understanding how memory addresses map to banks is key to avoiding **bank conflicts**.

Bank Mapping
============

For AMD GCN architecture, the LDS bank mapping is typically:

.. math::

   \text{bank} = \left( \frac{\text{address in bytes}}{4} \right) \bmod 32

This means:

- Addresses that differ by multiples of ``bank numbers * 4 bytes`` map to the same bank.
- Conflicts occur when multiple threads in the same wave access the same bank **in the same cycle**.

Not all the lanes can produce bank conflicts. HW divides access to LDS from wavefront into phases. Which lanes would be considered in each phase depends on the width of the instruction. Let us consider ``ds_write_b128`` as an example as it is the instruction that has the largest granularity write with the highest performance. Here access will be divided into 8 phases for 64 lane wavefront. If in 1 phase there will not be two thread access the same bank, there will bot be bank conflict:

- lane0~lane7
- lane8~lane15
- lane16~lane23
- lane24~lane31
- lane32~lane39
- lane40~lane47
- lane48~lane55
- lane56~lane63

If within each group of lanes there is no conflict it is an LDS bank conflict free write access.

Bank Access Patterns
====================

LDS bank access can be simulated for a given set of thread addresses. With a 32 bank LDS with 4 bytes per bank, each thread will be writing 8 2-byte elements (16 bytes total), consuming 4 banks in LDS. fp16 or bf16 are the common formats GPU kernels have to deal with. With the phase access pattern like above by default it is a bank conflict free LDS write access.

Write Access Pattern
--------------------

For LDS write instructions like ``ds_write_b128``, the hardware provides conflict-free access when threads write to consecutive addresses. Each phase of 8 lanes writes to different banks, avoiding conflicts.

Read Access Pattern
-------------------

Similarly for LDS read instruction ``ds_read_b128``, when there is no bank conflict in these 8 lane groups:

- 0:3+20:23
- 4:7+16:19
- 8:11+28:31
- 12:15+24:27
- 32:35+52:55
- 36:39+48:51
- 40:43+60:63
- 44:47+56:59

then it's bank conflict-free for LDS reading.

The reason for accessing the data vertically is because in most LDS access the MFMA instruction in the next step and the MFMA are requirde to access the data vertically like above.

The LDS read access pattern illustrated below is typical for LDS usage in machine learning workloads. The read pattern can generate 4-way bank conflicts in every phase of access. You can experiment with ``row_padding`` (padding in a number of banks) to see if the problem can be solved this way, but also remember that in practice this will require additional LDS storage. The bigger the padding, the more additional storage is necessary.

XOR Preshuffle: An Alternative to Padding
=========================================

Another technique to reduce LDS bank conflicts is **XOR preshuffling** (see :ref:`ck_tile_lds_index_swapping` for detailed implementation). Instead of adding padding between rows, we can permute the column indices for each row using XOR. This method can help to avoid bank conflicts without allocating extra storage in LDS.

For a wavefront of 64 threads, if each thread writes a vector of 8 fp16 elements (16 bytes), and the row size is 64 elements, the column index for each element in a row is adjusted as follows:

- ``KTypeSize = 2``
- ``KPerBlock = 64``  // 64 elements per row
- ``KPack = 8``  // 8 elements per thread

The adjusted column position for element ``(x, y)`` is:

.. math::

   x' = \left( y \bmod \frac{\text{KPerBlock}}{\text{KPack}} \right) \oplus x

where :math:`\oplus` is the bitwise XOR, and :math:`x, y` are the original positions of a vector element with respect to the LDS banks.

C++ Implementation
==================

Here's how CK implements XOR preshuffling:

.. code-block:: cpp

    // XOR-based column index adjustment
    template <index_t KPerBlock, index_t KPack>
    __device__ constexpr index_t xor_preshuffle(index_t row, index_t col)
    {
        constexpr index_t num_cols = KPerBlock / KPack;
        return (row % num_cols) ^ col;
    }

    // LDS write with XOR preshuffle
    template <typename DataType, index_t RowStride>
    __device__ void lds_write_with_xor(DataType* lds_ptr,
                                       const DataType* src,
                                       index_t row,
                                       index_t col)
    {
        // Apply XOR preshuffle to column index
        index_t col_xor = xor_preshuffle<64, 8>(row, col);
        
        // Write to LDS with adjusted column
        index_t offset = row * RowStride + col_xor * 8;
        
        // Vectorized write (assuming 128-bit write)
        *reinterpret_cast<float4*>(lds_ptr + offset) = 
            *reinterpret_cast<const float4*>(src);
    }

    // LDS read with XOR preshuffle
    template <typename DataType, index_t RowStride>
    __device__ void lds_read_with_xor(DataType* dst,
                                      const DataType* lds_ptr,
                                      index_t row,
                                      index_t col)
    {
        // Apply same XOR preshuffle for read
        index_t col_xor = xor_preshuffle<64, 8>(row, col);
        
        // Read from LDS with adjusted column
        index_t offset = row * RowStride + col_xor * 8;
        
        // Vectorized read
        *reinterpret_cast<float4*>(dst) = 
            *reinterpret_cast<const float4*>(lds_ptr + offset);
    }

Integration with CK Tile
========================

CK Tile handles LDS bank conflict avoidance through its abstractions:

1. **TileWindow** (:ref:`ck_tile_tile_window`): Automatically applies XOR preshuffling when loading/storing to LDS
2. **StaticDistributedTensor** (:ref:`ck_tile_static_distributed_tensor`): Manages LDS allocation with proper alignment
3. **LoadStoreTraits** (:ref:`ck_tile_load_store_traits`): Selects optimal access patterns to minimize conflicts

Example usage in CK Tile:

.. code-block:: cpp

    // CK Tile automatically handles bank conflict avoidance
    template <typename TileDistribution>
    __device__ void gemm_kernel()
    {
        // Create tile window with automatic XOR preshuffle
        auto a_window = make_tile_window(
            a_tensor_view,
            tile_size,
            origin,
            tile_distribution);
        
        // Load to LDS - XOR preshuffle applied automatically
        auto a_lds_tensor = make_static_distributed_tensor<
            element_type,
            decltype(tile_distribution)>();
        
        a_window.load(a_lds_tensor);
        
        // Subsequent reads from LDS are conflict-free
        // See :ref:`ck_tile_sweep_tile` for sweep operations
        sweep_tile(a_lds_tensor, [](auto idx, auto& val) {
            // Process data...
        });
    }

Performance Impact
==================

Proper LDS bank conflict avoidance can have significant performance impact:

- **4-way conflicts**: Can reduce effective LDS bandwidth by 75%
- **XOR preshuffle**: Restores full bandwidth with zero storage overhead
- **Padding**: Also effective but requires 12.5-25% more LDS storage

Best Practices
==============

1. **Use CK Tile abstractions**: They automatically handle bank conflict avoidance
2. **Prefer XOR preshuffle**: No storage overhead compared to padding
3. **Verify with profiling**: Use rocprof to check for LDS bank conflicts
4. **Consider access patterns**: Design algorithms with bank-friendly patterns

By understanding LDS bank conflicts and using CK Tile's automatic conflict avoidance mechanisms, developers can achieve optimal shared memory performance without manual optimization.

Related Topics
==============

- :ref:`ck_tile_lds_index_swapping` - Detailed XOR preshuffle implementation
- :ref:`ck_tile_swizzling_example` - Morton ordering for memory swizzling
- :ref:`ck_tile_gpu_basics` - Understanding AMD GPU architecture
- :ref:`ck_tile_tile_window` - Automatic conflict avoidance in data access
- :ref:`ck_tile_static_distributed_tensor` - LDS memory management
- :ref:`ck_tile_gemm_optimization` - Practical application in GEMM kernels
- :ref:`ck_tile_transforms` - Coordinate transformations for conflict avoidance
