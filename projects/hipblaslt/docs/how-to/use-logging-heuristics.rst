.. meta::
   :description: How to use the hipBLASLt logging and heuristic utilities
   :keywords: hipBLASLt, ROCm, library, API, logging, heuristics

.. _logging-heuristics:

=============================
Using logging and heuristics
=============================

This topic contains information about debugging and improving the application performance when using the hipBLASLt APIs.

Logging
==========

You can enable the hipBLASLt logging mechanism by setting the following environment variables before launching the target application:

*  ``HIPBLASLT_LOG_LEVEL=<level>``: The ``level`` can be one of the following values:

   .. csv-table::
      :header: "Value","Setting","Description"
      :widths: 5, 15, 80

      "``0``","Off","Logging is disabled (default)"
      "``1``","Error","Only errors are logged"
      "``2``","Trace","API calls that launch HIP kernels log their parameters and important information"
      "``3``","Hints","Hints that can potentially improve the application's performance"
      "``4``","Info","Provides general information about the library execution and can contain details about the heuristic status"
      "``5``","API trace","API calls log their parameters and important information"

*  ``HIPBLASLT_LOG_MASK=<mask>``: The ``mask`` is a combination of the following flags:

   .. csv-table::
      :header: "Value","Description"
      :widths: 10, 80

      "``0``","Off"
      "``1``","Error"
      "``2``","Trace"
      "``4``","Hints"
      "``8``","Info"
      "``16``","API trace"
      "``32``","Bench"
      "``64``","Profile"
      "``128``","Extended profile"

*  ``HIPBLASLT_LOG_FILE=<file_name>``: The ``file_name`` is a path to a logging file. The file name can contain ``%i``,
   which is replaced with the process ID, for example, ``<file_name>_%i.log``.
   If ``HIPBLASLT_LOG_FILE`` is not defined, the log messages are printed to stdout.

*  ``HIPBLASLT_ENABLE_MARKER=1``: Setting ``HIPBLASLT_ENABLE_MARKER`` to ``1`` enables marker trace for :doc:`ROCProfiler <rocprofiler:index>` profiling.

Heuristics cache
==================

hipBLASLt uses heuristics to pick the most suitable matmul kernel for execution based on the problem sizes,
GPU configuration, and other parameters. This requires performing some computations on the host CPU, which could take tens of microseconds.
To overcome this overhead, it's recommended that you query the heuristics once using :ref:`hipblasltmatmulalgogetheuristic`,
then reuse the result for subsequent computations using :ref:`hipblasltmatmul`.
