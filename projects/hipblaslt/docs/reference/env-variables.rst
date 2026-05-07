.. meta::
  :description: hipBLASLt environment variables reference
  :keywords: hipBLASLt, ROCm, API, environment variables, environment, reference

.. _environment-variables:

********************************************************************
hipBLASLt environment variables
********************************************************************

This section describes the important hipBLASLt environment variables,
which are grouped by functionality.

Logging and debugging
=====================

The logging and debugging environment variables for hipBLASLt are collected in the following table.
For more information, see :doc:`Use logging and heuristics <../how-to/use-logging-heuristics>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLASLT_LOG_LEVEL``
        | Controls the verbosity level of hipBLASLt logging output.
      - | 0: Off (logging disabled, default)
        | 1: Error (only errors are logged)
        | 2: Trace (API calls with kernel launches log parameters)
        | 3: Hints (performance improvement suggestions)
        | 4: Info (general library execution information)
        | 5: API trace (detailed API call parameters)

    * - | ``HIPBLASLT_LOG_MASK``
        | Controls logging output using bit mask flags (can be combined).
      - | 0: Off
        | 1: Error
        | 2: Trace
        | 4: Hints
        | 8: Info
        | 16: API trace
        | 32: Bench
        | 64: Profile
        | 128: Extended profile

    * - | ``HIPBLASLT_LOG_FILE``
        | Specifies path to logging file. Can contain ``%i`` for process ID replacement.
      - | Path to log file (for example, ``logfile_%i.log``)
        | If not defined: log messages printed to stdout

    * - | ``HIPBLASLT_ENABLE_MARKER``
        | Enables marker trace for ROCProfiler profiling.
      - | 0 or unset: Disable marker trace
        | 1: Enable marker trace


Offline tuning
===============

The offline tuning environment variables for hipBLASLt are collected in the following table.
For more information, see :doc:`Use hipBLASLt offline tuning <../how-to/how-to-use-hipblaslt-offline-tuning>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLASLT_TUNING_FILE``
        | Specifies file to store tuning results with best solution indices for GEMM problems.
      - | Path to tuning file (for example, ``tuning.txt``)
        | File stores optimal kernel indices for reuse

    * - | ``HIPBLASLT_TUNING_OVERRIDE_FILE``
        | Specifies file to load tuning results and override default kernel selection.
      - | Path to tuning file (for example, ``tuning.txt``)
        | Loads previously saved optimal kernel choices

    * - | ``HIPBLASLT_TUNING_USER_MAX_WORKSPACE``
        | Sets maximum workspace size constraint during tuning stage.
      - | Integer value in bytes (default: 128 * 1024 * 1024)
        | Limits workspace size for solution selection

Origami with Stream-K configuration
===================================

The Origami with Stream-K configuration environment variables for hipBLASLt are collected in the following table.
These variables apply to all GEMMs in an application.
For more information, see :doc:`Use Stream-K with hipBLASLt <../how-to/how-to-use-streamk>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``TENSILE_SOLUTION_SELECTION_METHOD``
        | Controls hipBLASLt kernel selection strategy for GEMM operations.
      - | 0: Default (standard tuned libraries, no Stream-K)
        | 2: Origami with Stream-K (enables Origami solution selection for consistent performance)
        | This variable has no effect on the AMD Instinct™ MI350 series. Stream-K is always used.

    * - | ``TENSILE_STREAMK_DYNAMIC_GRID``
        | Controls Stream-K dynamic grid size selection behavior.
      - | 0: Disable dynamic grid (use all available compute units)
        | 1: Only reduce compute units for small problems to the number of output tiles when ``num_tiles < CU count``
        | 2: Also reduce compute units for large sizes to improve the data-parallel portion and reduce power
        | 3: Analytically predict the best grid size by weighing the cost of the fix-up step and the cost of processing MACs 
        | 4: The Stream-K algorithm behaves like data parallel (Launch WGs =  number of CUs)
        | 5: The Stream-K algorithm uses the Origami ``select_best_grid_size`` function
        | 6: Default (automatically pick the optimal workgroup count)

    * - | ``TENSILE_STREAMK_FIXED_GRID``
        | Overrides default grid size with specified number of workgroups for Stream-K kernels.
      - | Integer value specifying number of workgroups
        | Example: 64 (limits GEMM kernels to 64 workgroups)

    * - | ``TENSILE_STREAMK_MAX_CUS``
        | Sets maximum number of compute units for Stream-K kernels.
      - | Integer value specifying maximum compute units
        | Example: 32 (limits GEMM kernels to 32 compute units)
        | Default: All available compute units

Type overrides
======================

Overrides for specific types.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32``
        | Overrides the compute type used for GEMMs which specify a compute type of ``XF32``.
      - | -1: Off (Default)
        | 0: F32
        | 1: XF32(eg TF32)
        | 2: F32_BF16
