===================================
Cache Flushing for GPU Benchmarking
===================================

Overview
========

When benchmarking GPU kernels, accurate performance measurements require understanding and controlling cache behavior. Running a kernel multiple times with the same input data can lead to artificially fast results due to **cache hits**, where data and instructions are served from fast GPU cache rather than slow High Bandwidth Memory (HBM).

Composable Kernel provides two complementary mechanisms to ensure realistic "cold cache" performance measurements:

1. **Instruction Cache Flushing** - Invalidates cached GPU instructions
2. **Rotating Memory Buffers** - Cycles through multiple data buffer copies at different memory addresses

This document explains how these mechanisms work and how to use them in benchmarks.

The Problem: Hot vs. Cold Cache
================================

GPU Memory Hierarchy
--------------------

GPUs have a multi-level cache hierarchy:

.. code-block:: text

    Fast → Slow, Small → Large

    ┌─────────────────┐
    │  Register File  │  ~1 cycle
    ├─────────────────┤
    │  L1 I-Cache     │  ~4 cycles   ← Instruction cache
    ├─────────────────┤
    │  L1 Data Cache  │  ~4 cycles   ← Data cache
    ├─────────────────┤
    │  L2 Cache       │  ~50 cycles
    ├─────────────────┤
    │  HBM (VRAM)     │  ~400 cycles
    └─────────────────┘

Cache Behavior Without Flushing
--------------------------------

When running a kernel repeatedly without cache management:

.. code-block:: text

    Run 1:  [Cache MISS]  → Fetch from HBM → 400 cycles → 5.2ms
    Run 2:  [Cache HIT!]  → Read from L1/L2 → 4 cycles → 3.8ms ← Artificially fast!
    Run 3:  [Cache HIT!]  → Read from L1/L2 → 4 cycles → 3.8ms
    ...
    Average: 4.1ms (misleading - not representative of real-world performance)

This leads to:

- ✗ Inflated performance numbers
- ✗ Inconsistent timing between first and subsequent runs
- ✗ Unfair comparisons between different kernels
- ✗ Misleading optimization decisions

Solution 1: Instruction Cache Flushing
=======================================

What is Instruction Cache?
---------------------------

The **instruction cache (I-cache)** is a small, fast memory on each GPU compute unit that stores recently executed machine code instructions. When a thread needs to execute an instruction:

1. The **Program Counter (PC)** holds the instruction's memory address
2. The GPU checks if that address exists in the I-cache
3. **Cache HIT**: Instruction read instantly from I-cache (~4 cycles)
4. **Cache MISS**: Instruction fetched from HBM (~400 cycles), then cached

How It Works
------------

The GPU uses **address-based caching**: when you launch the same kernel multiple times, the kernel code resides at the same memory address, allowing the I-cache to serve cached instructions.

.. code-block:: text

    First Kernel Run:
    PC = 0x7F8A0000  →  I-Cache lookup  →  MISS  →  Fetch from HBM  →  Cache it
    
    Second Kernel Run (without flush):
    PC = 0x7F8A0000  →  I-Cache lookup  →  HIT!  →  Read from cache (fast!)
    
    Second Kernel Run (with flush):
    PC = 0x7F8A0000  →  I-Cache lookup  →  MISS  →  Fetch from HBM again

The ``flush_icache()`` Function
--------------------------------

Located in ``include/ck_tile/host/flush_icache.hpp``:

.. code-block:: cpp

    namespace ck_tile {
    // GPU kernel to invalidate instruction cache for accurate benchmarking.
    static __global__ void flush_cache()
    {
        asm __volatile__("s_icache_inv \n\t"    // Invalidate I-cache
                         "s_nop 0 \n\t"          // Wait cycles (16 NOPs)
                         "s_nop 0 \n\t"
                         // ... 14 more NOPs
                         "s_nop 0 \n\t" ::
                             :);
    }
    }

**Key Components:**

- ``s_icache_inv``: AMD GPU instruction that invalidates the L1 instruction cache on the current compute unit
- ``s_nop 0`` (×16): No-operation instructions (NOPs) that create a 16-cycle delay to ensure cache invalidation completes before the kernel exits

**Why 16 NOPs?**

The ``s_icache_inv`` instruction is **asynchronous**: it initiates cache invalidation but doesn't wait for completion. Without the NOPs, the kernel might exit before the flush finishes, leading to race conditions and incomplete cache invalidation.

Launching the Flush Kernel
---------------------------

From ``include/ck_tile/host/rotating_buffers.hpp``:

.. code-block:: cpp

    inline void flush_icache()
    {
        hipDeviceProp_t deviceProps;
        HIP_CHECK_ERROR(hipGetDeviceProperties(&deviceProps, 0));
        
        // Over-provision blocks to ensure all CUs execute the flush instruction.
        // With imperfect scheduling, launching exactly 1 block per CU doesn't guarantee coverage.
        // 60x over-provisioning provides statistical certainty that every CU gets at least one block.
        constexpr int32_t blocks_per_cu = 60;
        int32_t gpu_block3 = deviceProps.multiProcessorCount * blocks_per_cu;

        ck_tile::flush_cache<<<dim3(gpu_block3), dim3(64), 0, nullptr>>>();
        HIP_CHECK_ERROR(hipGetLastError());
    }

**Why 60× Over-provisioning?**

The I-cache is **per-compute-unit** (CU). To flush all CUs, we must ensure every CU executes at least one instance of ``s_icache_inv``. 

- Launching exactly 1 block per CU doesn't guarantee 1:1 mapping due to GPU scheduler behavior
- Launching 60 blocks per CU provides statistical certainty that every CU receives work
- For a 120-CU GPU: 120 × 60 = 7,200 blocks × 64 threads = 460,800 total threads

This ensures comprehensive instruction cache flushing across all compute units.

Solution 2: Rotating Memory Buffers
====================================

What is Data Cache?
-------------------

While I-cache stores instructions, **data cache** (L1 data, L2) stores matrix data (inputs A, B and output C). When a kernel reads the same matrix repeatedly, the data is served from cache rather than HBM.

The RotatingMemWrapper Struct
------------------------------

Located in ``include/ck_tile/host/rotating_buffers.hpp``:

.. code-block:: cpp

    template <typename ADataType, typename BDataType>
    struct RotatingMemWrapper
    {
        RotatingMemWrapper(const void* a_ptr_,
                          const void* b_ptr_,
                          std::size_t rotating_count_,
                          std::size_t size_a_,
                          std::size_t size_b_);
        
        void Next();  // Rotate to next buffer copy
        ~RotatingMemWrapper() noexcept;  // Cleanup
    };

**Purpose**: Prevents data cache reuse by cycling through multiple copies of input matrices at different memory addresses.

How It Works
------------

**Constructor: Create Buffer Copies**

.. code-block:: cpp

    RotatingMemWrapper(a_ptr, b_ptr, rotating_count=3, size_a, size_b)
    {
        // Store original buffer pointers as first entry
        p_a_grids.push_back(a_ptr);
        p_b_grids.push_back(b_ptr);
        
        // Create (rotating_count - 1) additional copies at different memory addresses
        for(size_t i = 1; i < rotating_count; i++)
        {
            void* pADeviceBuf;
            hipMalloc(&pADeviceBuf, size_a);
            hipMemcpy(pADeviceBuf, p_a_grids[0], size_a, hipMemcpyDeviceToDevice);
            p_a_grids.push_back(pADeviceBuf);
            
            // Same for B matrix...
        }
    }

Result:

.. code-block:: text

    GPU Memory:
    ┌─────────────────────────┐
    │ Matrix A (original)     │  Address: 0x1000
    │ Matrix A (copy 1)       │  Address: 0x2000
    │ Matrix A (copy 2)       │  Address: 0x3000
    │ Matrix B (original)     │  Address: 0x4000
    │ Matrix B (copy 1)       │  Address: 0x5000
    │ Matrix B (copy 2)       │  Address: 0x6000
    └─────────────────────────┘

**Next(): Rotate to Next Buffer**

.. code-block:: cpp

    void Next()
    {
        if(rotating_count > 1)
        {
            std::size_t idx = iter++ % rotating_count;  // Cycle: 0,1,2,0,1,2,...
            a_ptr = p_a_grids[idx];
            b_ptr = p_b_grids[idx];
        }
    }

Usage in benchmarking loop:

.. code-block:: text

    Iteration 1:  Next() → Use buffers at 0x1000, 0x4000 → Kernel reads → Cache miss
    Iteration 2:  Next() → Use buffers at 0x2000, 0x5000 → Kernel reads → Cache miss
    Iteration 3:  Next() → Use buffers at 0x3000, 0x6000 → Kernel reads → Cache miss
    Iteration 4:  Next() → Use buffers at 0x1000, 0x4000 → Kernel reads → Cache miss
    ...

By the time the buffers cycle back to the first copy, the cache has likely evicted the old data.

**Destructor: Cleanup**

.. code-block:: cpp

    ~RotatingMemWrapper() noexcept
    {
        // Restore original buffer pointers
        a_ptr = p_a_grids[0];
        b_ptr = p_b_grids[0];

        // Free extra buffer copies (index 0 is original, don't free it)
        for(size_t i = 1; i < rotating_count; i++)
        {
            hipFree(p_a_grids[i]);
            hipFree(p_b_grids[i]);
        }
    }

Using Cache Flushing in Practice
=================================

Command Line Argument
---------------------

The ``flush_cache`` command-line argument controls whether cache flushing is enabled:

.. code-block:: bash

    # Enable cache flushing (cold cache benchmarking)
    ./gemm_example --flush_cache=1 --rotating_count=3
    
    # Disable cache flushing (hot cache benchmarking)
    ./gemm_example --flush_cache=0

In ``run_gemm_quant_example.inc``:

.. code-block:: cpp

    bool flush_cache = arg_parser.get_bool("flush_cache");
    int rotating_count = arg_parser.get_int("rotating_count");
    
    // Pass to stream_config
    ck_tile::stream_config{
        nullptr,      // stream
        true,         // time_kernel
        1,            // log_level
        n_warmup,     // cold_niters (warmup iterations)
        n_repeat,     // nrepeat (timed iterations)
        true,         // is_gpu_timer
        flush_cache,  // flush_cache_ ← Controls cache flushing
        rotating_count // rotating_count_ ← Number of buffer copies
    }

Integration with Timing Loop
-----------------------------

The ``launch_kernel_time_mask`` function integrates both mechanisms:

.. code-block:: cpp

    // From include/ck_tile/host/kernel_launch.hpp
    template <typename PreprocessFunc, typename... Callables>
    float launch_kernel_time_mask(const stream_config& s, 
                                   PreprocessFunc preprocess,
                                   Callables&&... callables)
    {
        // Timing loop (simplified)
        for(int i = 0; i < s.nrepeat_; i++)
        {
            preprocess();      // 1. Flush I-cache + rotate buffers
            callables_func();  // 2. Launch kernel
        }
        
        return average_time;
    }

Complete Example
----------------

From ``example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc``:

.. code-block:: cpp

    // Setup rotating memory wrapper
    RotatingMemWrapper<ADataType, BDataType> rotating_mem(
        a_ptr, b_ptr, rotating_count, size_a, size_b);
    
    // Define preprocessing: flush I-cache + rotate buffers
    auto preprocess = [&]() {
        if(flush_cache) {
            flush_icache();      // Invalidate instruction cache
            rotating_mem.Next(); // Switch to next buffer copy
        }
    };
    
    // Define kernel launch
    auto kernel_launch = [&]() {
        gemm_kernel<<<grid, block>>>(a_ptr, b_ptr, c_ptr, M, N, K);
    };
    
    // Benchmark with cache control
    float avg_time = launch_kernel_time_mask(
        stream_config,   // Config with flush_cache and rotating_count
        preprocess,      // Flush + rotate before each iteration
        kernel_launch    // Kernel to benchmark
    );

Execution Flow
--------------

With ``flush_cache=true`` and ``rotating_count=3``, ``nrepeat=100``:

.. code-block:: text

    Warmup Phase (n_warmup iterations):
      - Run kernel without timing
      - Prime GPU, warm up scheduler
    
    Timed Phase (100 iterations):
      Iteration 1:  flush_icache() → rotating_mem.Next() → Use buffer copy 0 → kernel() → Measure
      Iteration 2:  flush_icache() → rotating_mem.Next() → Use buffer copy 1 → kernel() → Measure
      Iteration 3:  flush_icache() → rotating_mem.Next() → Use buffer copy 2 → kernel() → Measure
      Iteration 4:  flush_icache() → rotating_mem.Next() → Use buffer copy 0 → kernel() → Measure
      ...
      Iteration 100: flush_icache() → rotating_mem.Next() → Use buffer copy 1 → kernel() → Measure
    
    Return: Average time per iteration (excluding preprocess overhead)

References
==========

Related Files
-------------

- ``include/ck_tile/host/flush_icache.hpp`` - I-cache flush kernel implementation
- ``include/ck_tile/host/rotating_buffers.hpp`` - RotatingMemWrapper implementation
- ``include/ck_tile/host/kernel_launch.hpp`` - Timing loop integration

Conclusion
==========

Accurate GPU kernel benchmarking requires careful control of cache behavior. The combination of **instruction cache flushing** (``flush_icache``) and **rotating memory buffers** (``RotatingMemWrapper``) ensures realistic "cold cache" performance measurements that represent real-world application behavior.

By understanding and utilizing these mechanisms through the ``flush_cache`` command-line argument, you can obtain trustworthy performance data for optimization decisions and fair kernel comparisons.

