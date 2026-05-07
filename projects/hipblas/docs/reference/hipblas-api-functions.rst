.. meta::
  :description: hipBLAS documentation and API reference library
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, documentation, interface

.. _api_label:

*************
hipBLAS API
*************

The topic discusses technical aspects of the hipBLAS API and provides reference information about the API functions.

The hipBLAS interface
=====================

The hipBLAS interface is compatible with the rocBLAS and cuBLAS-v2 APIs. Porting a CUDA application which
originally called the cuBLAS API to an application calling the hipBLAS API should be relatively straightforward.

GEMV API
--------

For example, the hipBLAS SGEMV interface is:

.. code-block:: cpp

   hipblasStatus_t
   hipblasSgemv(hipblasHandle_t handle,
                hipblasOperation_t trans,
                int m, int n, const float *alpha,
                const float *A, int lda,
                const float *x, int incx, const float *beta,
                float *y, int incy );


Batched and strided GEMM API
----------------------------

hipBLAS GEMM can process matrices in batches with regular strides by using the strided-batched version of the API:

.. code-block:: cpp

   hipblasStatus_t
   hipblasSgemmStridedBatched(hipblasHandle_t handle,
                              hipblasOperation_t transa, hipblasOperation_t transb,
                              int m, int n, int k, const float *alpha,
                              const float *A, int lda, long long bsa,
                              const float *B, int ldb, long long bsb, const float *beta,
                              float *C, int ldc, long long bsc,
                              int batchCount);

hipBLAS assumes matrix ``A`` and vectors ``x`` and ``y`` are allocated in the GPU memory space for data.
You are responsible for copying data to and from the host and device memory.

Naming conventions
==================

hipBLAS follows the following naming conventions:

*  Upper case for a matrix, for example, matrix A, B, C   GEMM (C = A*B)
*  Lower case for a vector, for example, vector x, y    GEMV (y = A*x)


Notations
=========

hipBLAS function uses the following notations to denote precisions:

*  h  = half
*  bf = 16-bit brain floating point
*  s  = single
*  d  = double
*  c  = single complex
*  z  = double complex

.. _hipblas-backend:

hipBLAS backends
================

hipBLAS has multiple backends for different platforms: cuBLAS for the NVIDIA
platform and rocBLAS for the AMD platform. The cuBLAS backend does not support
all the functions and only returns the ``HIPBLAS_STATUS_NOT_SUPPORTED`` status
code.

The following level 1-3 and solver functions are not supported with the cuBLAS
backend:

* :ref:`AXPY <hipblas_axpy>` functions with half.
* :ref:`DOT <hipblas_dot>` functions with half and bfloat16.
* :ref:`SPR <hipblas_spr>` functions with ``std:complex<float>`` and
  ``std:complex<double>``.
* All the batched functions except for :ref:`TRSM <hipblas_trsm>`,
  :ref:`GEMV <hipblas_gemv>`, and :ref:`GEMM <hipblas_gemm>` and
  :ref:`solver functions <solver_api>`
  (:ref:`GETRF <hipblas_getrf>`, :ref:`GETRS <hipblas_getrs>`,
  :ref:`GEQRF <hipblas_geqrf>`, :ref:`GELS <hipblas_gels>`).
* All the strided_batched functions except for :ref:`GEMV <hipblas_gemv>` and
  :ref:`GEMM <hipblas_gemm>`.
* :ref:`TRTRI <hipblas_trtri>` and :ref:`TRSMEX <hipblas_trsmex>` functions.
* :ref:`GETRF <hipblas_getrf>`, :ref:`GETRS <hipblas_getrs>`,
  :ref:`GEQRF <hipblas_geqrf>`, and :ref:`GELS <hipblas_gels>` non-batched and
  strided_batched functions.

.. _ILP64 API:

ILP64 interfaces
================

The hipBLAS library Level-1 functions are also provided with ILP64 interfaces.
With these interfaces, all ``int`` arguments are replaced with the typename
``int64_t``. These ILP64 function names all end with the ``_64`` suffix.
The only output arguments that change are for
xMAX and xMIN, for which the index is now ``int64_t``. Function level documentation is not
repeated for these APIs because they are identical in behavior to the LP64 versions.
However functions that support this alternate API include the line:
``This function supports the 64-bit integer interface``.

The functionality of the ILP64 interfaces depends on the backend being used,
see the :doc:`rocBLAS <rocblas:index>` or NVIDIA CUDA cuBLAS documentation for more
information about support for ILP64 interfaces.

Atomic operations
=================

Some hipBLAS functions might use atomic operations to increase performance.
This can cause these functions to give results that are not bit-wise reproducible.
By default, the rocBLAS backend allows the use of atomics while the CUDA cuBLAS backend disallows their use.
To set the desired behavior, users can call
:any:`hipblasSetAtomicsMode`. See the :doc:`rocBLAS <rocblas:index>` or CUDA
cuBLAS documentation for more specific information about atomic operations in the backend library.

Graph support for hipBLAS
=========================

Graph support (also referred to as stream capture support) for hipBLAS depends on the backend being used.
If rocBLAS is the backend, see the :doc:`rocBLAS <rocblas:index>` documentation.
Similarly, if CUDA cuBLAS is the backend, see the cuBLAS documentation.

Custom data types
=================

hipBlas defines the ``hipblasBfloat16``. For more details, see
:ref:`custom_types`.

*************
hipBLAS types
*************

For information about the ``hipblasStatus_t``, ``hipblasComputeType_t``, and ``hipblasOperation_t`` enumerations,
see ``hipblas-common.h`` in the `hipBLAS-common GitHub <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas-common>`_ repository.

Definitions
===========

hipblasHandle_t
---------------
.. doxygentypedef:: hipblasHandle_t

hipblasHalf
------------
.. doxygentypedef:: hipblasHalf

hipblasInt8
------------
.. doxygentypedef:: hipblasInt8

hipblasStride
--------------
.. doxygentypedef:: hipblasStride

hipblasBfloat16
----------------
.. doxygenstruct:: hipblasBfloat16

Enums
=====

Enumeration constants have numbering that is consistent with CBLAS, ACML, and most standard C BLAS libraries.

hipblasStatus_t
-----------------

For information about ``hipblasStatus_t``,
see ``hipblas-common.h`` in the `hipBLAS-common GitHub <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas-common>`_ repository.

hipblasOperation_t
------------------

For information about ``hipblasOperation_t``,
see ``hipblas-common.h`` in the `hipBLAS-common GitHub <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas-common>`_ repository.


hipblasPointerMode_t
--------------------
.. doxygenenum:: hipblasPointerMode_t

hipblasFillMode_t
------------------
.. doxygenenum:: hipblasFillMode_t

hipblasDiagType_t
-----------------
.. doxygenenum:: hipblasDiagType_t

hipblasSideMode_t
-----------------
.. doxygenenum:: hipblasSideMode_t

hipblasComputeType_t
--------------------

For information about ``hipblasComputeType_t``,
see ``hipblas-common.h`` in the `hipBLAS-common GitHub <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas-common>`_ repository.


hipblasGemmAlgo_t
------------------
.. doxygenenum:: hipblasGemmAlgo_t

hipblasAtomicsMode_t
---------------------
.. doxygenenum:: hipblasAtomicsMode_t

*****************
hipBLAS functions
*****************

.. _level-1:

Level 1 BLAS
============

.. contents:: List of Level-1 BLAS functions
   :local:
   :backlinks: top

.. _hipblas_amax:

hipblasIXamax + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasIsamax
    :outline:
.. doxygenfunction:: hipblasIdamax
    :outline:
.. doxygenfunction:: hipblasIcamax
    :outline:
.. doxygenfunction:: hipblasIzamax

The ``amax`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasIsamaxBatched
    :outline:
.. doxygenfunction:: hipblasIdamaxBatched
    :outline:
.. doxygenfunction:: hipblasIcamaxBatched
    :outline:
.. doxygenfunction:: hipblasIzamaxBatched

The ``amaxBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasIsamaxStridedBatched
    :outline:
.. doxygenfunction:: hipblasIdamaxStridedBatched
    :outline:
.. doxygenfunction:: hipblasIcamaxStridedBatched
    :outline:
.. doxygenfunction:: hipblasIzamaxStridedBatched

The ``amaxStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_amin:

hipblasIXamin + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasIsamin
    :outline:
.. doxygenfunction:: hipblasIdamin
    :outline:
.. doxygenfunction:: hipblasIcamin
    :outline:
.. doxygenfunction:: hipblasIzamin

The ``amin`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasIsaminBatched
    :outline:
.. doxygenfunction:: hipblasIdaminBatched
    :outline:
.. doxygenfunction:: hipblasIcaminBatched
    :outline:
.. doxygenfunction:: hipblasIzaminBatched

The ``aminBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasIsaminStridedBatched
    :outline:
.. doxygenfunction:: hipblasIdaminStridedBatched
    :outline:
.. doxygenfunction:: hipblasIcaminStridedBatched
    :outline:
.. doxygenfunction:: hipblasIzaminStridedBatched

The ``aminStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_asum:

hipblasXasum + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSasum
    :outline:
.. doxygenfunction:: hipblasDasum
    :outline:
.. doxygenfunction:: hipblasScasum
    :outline:
.. doxygenfunction:: hipblasDzasum

The ``asum`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSasumBatched
    :outline:
.. doxygenfunction:: hipblasDasumBatched
    :outline:
.. doxygenfunction:: hipblasScasumBatched
    :outline:
.. doxygenfunction:: hipblasDzasumBatched

The ``asumBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSasumStridedBatched
    :outline:
.. doxygenfunction:: hipblasDasumStridedBatched
    :outline:
.. doxygenfunction:: hipblasScasumStridedBatched
    :outline:
.. doxygenfunction:: hipblasDzasumStridedBatched

The ``asumStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_axpy:

hipblasXaxpy + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasHaxpy
    :outline:
.. doxygenfunction:: hipblasSaxpy
    :outline:
.. doxygenfunction:: hipblasDaxpy
    :outline:
.. doxygenfunction:: hipblasCaxpy
    :outline:
.. doxygenfunction:: hipblasZaxpy

The ``axpy`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasHaxpyBatched
    :outline:
.. doxygenfunction:: hipblasSaxpyBatched
    :outline:
.. doxygenfunction:: hipblasDaxpyBatched
    :outline:
.. doxygenfunction:: hipblasCaxpyBatched
    :outline:
.. doxygenfunction:: hipblasZaxpyBatched

The ``axpyBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasHaxpyStridedBatched
    :outline:
.. doxygenfunction:: hipblasSaxpyStridedBatched
    :outline:
.. doxygenfunction:: hipblasDaxpyStridedBatched
    :outline:
.. doxygenfunction:: hipblasCaxpyStridedBatched
    :outline:
.. doxygenfunction:: hipblasZaxpyStridedBatched

The ``axpyStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_copy:

hipblasXcopy + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasScopy
    :outline:
.. doxygenfunction:: hipblasDcopy
    :outline:
.. doxygenfunction:: hipblasCcopy
    :outline:
.. doxygenfunction:: hipblasZcopy

The ``copy`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasScopyBatched
    :outline:
.. doxygenfunction:: hipblasDcopyBatched
    :outline:
.. doxygenfunction:: hipblasCcopyBatched
    :outline:
.. doxygenfunction:: hipblasZcopyBatched

The ``copyBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasScopyStridedBatched
    :outline:
.. doxygenfunction:: hipblasDcopyStridedBatched
    :outline:
.. doxygenfunction:: hipblasCcopyStridedBatched
    :outline:
.. doxygenfunction:: hipblasZcopyStridedBatched

The ``copyStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_dot:

hipblasXdot + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasHdot
    :outline:
.. doxygenfunction:: hipblasBfdot
    :outline:
.. doxygenfunction:: hipblasSdot
    :outline:
.. doxygenfunction:: hipblasDdot
    :outline:
.. doxygenfunction:: hipblasCdotc
    :outline:
.. doxygenfunction:: hipblasCdotu
    :outline:
.. doxygenfunction:: hipblasZdotc
    :outline:
.. doxygenfunction:: hipblasZdotu

The ``dot`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasHdotBatched
    :outline:
.. doxygenfunction:: hipblasBfdotBatched
    :outline:
.. doxygenfunction:: hipblasSdotBatched
    :outline:
.. doxygenfunction:: hipblasDdotBatched
    :outline:
.. doxygenfunction:: hipblasCdotcBatched
    :outline:
.. doxygenfunction:: hipblasCdotuBatched
    :outline:
.. doxygenfunction:: hipblasZdotcBatched
    :outline:
.. doxygenfunction:: hipblasZdotuBatched

The ``dotBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasHdotStridedBatched
    :outline:
.. doxygenfunction:: hipblasBfdotStridedBatched
    :outline:
.. doxygenfunction:: hipblasSdotStridedBatched
    :outline:
.. doxygenfunction:: hipblasDdotStridedBatched
    :outline:
.. doxygenfunction:: hipblasCdotcStridedBatched
    :outline:
.. doxygenfunction:: hipblasCdotuStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdotcStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdotuStridedBatched

The ``dotStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_nrm2:

hipblasXnrm2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSnrm2
    :outline:
.. doxygenfunction:: hipblasDnrm2
    :outline:
.. doxygenfunction:: hipblasScnrm2
    :outline:
.. doxygenfunction:: hipblasDznrm2

The ``nrm2`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSnrm2Batched
    :outline:
.. doxygenfunction:: hipblasDnrm2Batched
    :outline:
.. doxygenfunction:: hipblasScnrm2Batched
    :outline:
.. doxygenfunction:: hipblasDznrm2Batched

The ``nrm2Batched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSnrm2StridedBatched
    :outline:
.. doxygenfunction:: hipblasDnrm2StridedBatched
    :outline:
.. doxygenfunction:: hipblasScnrm2StridedBatched
    :outline:
.. doxygenfunction:: hipblasDznrm2StridedBatched

The ``nrm2StridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_rot:

hipblasXrot + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasSrot
    :outline:
.. doxygenfunction:: hipblasDrot
    :outline:
.. doxygenfunction:: hipblasCrot
    :outline:
.. doxygenfunction:: hipblasCsrot
    :outline:
.. doxygenfunction:: hipblasZrot
    :outline:
.. doxygenfunction:: hipblasZdrot

The ``rot`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSrotBatched
    :outline:
.. doxygenfunction:: hipblasDrotBatched
    :outline:
.. doxygenfunction:: hipblasCrotBatched
    :outline:
.. doxygenfunction:: hipblasCsrotBatched
    :outline:
.. doxygenfunction:: hipblasZrotBatched
    :outline:
.. doxygenfunction:: hipblasZdrotBatched

The ``rotBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasDrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasCrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasZrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdrotStridedBatched

The ``rotStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_rotg:

hipblasXrotg + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSrotg
    :outline:
.. doxygenfunction:: hipblasDrotg
    :outline:
.. doxygenfunction:: hipblasCrotg
    :outline:
.. doxygenfunction:: hipblasZrotg

The ``rotg`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSrotgBatched
    :outline:
.. doxygenfunction:: hipblasDrotgBatched
    :outline:
.. doxygenfunction:: hipblasCrotgBatched
    :outline:
.. doxygenfunction:: hipblasZrotgBatched

The ``rotgBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSrotgStridedBatched
    :outline:
.. doxygenfunction:: hipblasDrotgStridedBatched
    :outline:
.. doxygenfunction:: hipblasCrotgStridedBatched
    :outline:
.. doxygenfunction:: hipblasZrotgStridedBatched

The ``rotgStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_rotm:

hipblasXrotm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSrotm
    :outline:
.. doxygenfunction:: hipblasDrotm

The ``rotm`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSrotmBatched
    :outline:
.. doxygenfunction:: hipblasDrotmBatched

The ``rotmBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSrotmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDrotmStridedBatched

The ``rotmStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_rotmg:

hipblasXrotmg + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasSrotmg
    :outline:
.. doxygenfunction:: hipblasDrotmg

The ``rotmg`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSrotmgBatched
    :outline:
.. doxygenfunction:: hipblasDrotmgBatched

The ``rotmgBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSrotmgStridedBatched
    :outline:
.. doxygenfunction:: hipblasDrotmgStridedBatched

The ``rotmgStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_scal:

hipblasXscal + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSscal
    :outline:
.. doxygenfunction:: hipblasDscal
    :outline:
.. doxygenfunction:: hipblasCscal
    :outline:
.. doxygenfunction:: hipblasCsscal
    :outline:
.. doxygenfunction:: hipblasZscal
    :outline:
.. doxygenfunction:: hipblasZdscal

The ``scal`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSscalBatched
    :outline:
.. doxygenfunction:: hipblasDscalBatched
    :outline:
.. doxygenfunction:: hipblasCscalBatched
    :outline:
.. doxygenfunction:: hipblasZscalBatched
    :outline:
.. doxygenfunction:: hipblasCsscalBatched
    :outline:
.. doxygenfunction:: hipblasZdscalBatched

The ``scalBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasDscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasCscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasZscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdscalStridedBatched

The ``scalStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_swap:

hipblasXswap + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSswap
    :outline:
.. doxygenfunction:: hipblasDswap
    :outline:
.. doxygenfunction:: hipblasCswap
    :outline:
.. doxygenfunction:: hipblasZswap

The ``swap`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSswapBatched
    :outline:
.. doxygenfunction:: hipblasDswapBatched
    :outline:
.. doxygenfunction:: hipblasCswapBatched
    :outline:
.. doxygenfunction:: hipblasZswapBatched

The ``swapBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSswapStridedBatched
    :outline:
.. doxygenfunction:: hipblasDswapStridedBatched
    :outline:
.. doxygenfunction:: hipblasCswapStridedBatched
    :outline:
.. doxygenfunction:: hipblasZswapStridedBatched

The ``swapStridedBatched`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _level-2:

Level 2 BLAS
============
.. contents:: List of Level-2 BLAS functions
   :local:
   :backlinks: top

.. _hipblas_gbmv:

hipblasXgbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgbmv
    :outline:
.. doxygenfunction:: hipblasDgbmv
    :outline:
.. doxygenfunction:: hipblasCgbmv
    :outline:
.. doxygenfunction:: hipblasZgbmv

The ``gbmv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSgbmvBatched
    :outline:
.. doxygenfunction:: hipblasDgbmvBatched
    :outline:
.. doxygenfunction:: hipblasCgbmvBatched
    :outline:
.. doxygenfunction:: hipblasZgbmvBatched

The ``gbmvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSgbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgbmvStridedBatched

The ``gbmvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_gemv:

hipblasXgemv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgemv
    :outline:
.. doxygenfunction:: hipblasDgemv
    :outline:
.. doxygenfunction:: hipblasCgemv
    :outline:
.. doxygenfunction:: hipblasZgemv

The ``gemv``` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSgemvBatched
    :outline:
.. doxygenfunction:: hipblasDgemvBatched
    :outline:
.. doxygenfunction:: hipblasCgemvBatched
    :outline:
.. doxygenfunction:: hipblasZgemvBatched

The ``gemvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSgemvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgemvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgemvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgemvStridedBatched

The ``gemvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_ger:

hipblasXger + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSger
    :outline:
.. doxygenfunction:: hipblasDger
    :outline:
.. doxygenfunction:: hipblasCgeru
    :outline:
.. doxygenfunction:: hipblasCgerc
    :outline:
.. doxygenfunction:: hipblasZgeru
    :outline:
.. doxygenfunction:: hipblasZgerc

The ``ger`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSgerBatched
    :outline:
.. doxygenfunction:: hipblasDgerBatched
    :outline:
.. doxygenfunction:: hipblasCgeruBatched
    :outline:
.. doxygenfunction:: hipblasCgercBatched
    :outline:
.. doxygenfunction:: hipblasZgeruBatched
    :outline:
.. doxygenfunction:: hipblasZgercBatched

The ``gerBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSgerStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgerStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgeruStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgercStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgeruStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgercStridedBatched

The ``gerStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_hbmv:

hipblasXhbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChbmv
    :outline:
.. doxygenfunction:: hipblasZhbmv

The ``hbmv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChbmvBatched
    :outline:
.. doxygenfunction:: hipblasZhbmvBatched

The ``hbmvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhbmvStridedBatched

The ``hbmvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_hemv:

hipblasXhemv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChemv
    :outline:
.. doxygenfunction:: hipblasZhemv

The ``hemv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChemvBatched
    :outline:
.. doxygenfunction:: hipblasZhemvBatched

The ``hemvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChemvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhemvStridedBatched

The ``hemvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_her:

hipblasXher + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasCher
    :outline:
.. doxygenfunction:: hipblasZher

The ``her`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCherBatched
    :outline:
.. doxygenfunction:: hipblasZherBatched

The ``herBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCherStridedBatched
    :outline:
.. doxygenfunction:: hipblasZherStridedBatched

The ``herStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_her2:

hipblasXher2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasCher2
    :outline:
.. doxygenfunction:: hipblasZher2

The ``her2`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCher2Batched
    :outline:
.. doxygenfunction:: hipblasZher2Batched

The ``her2Batched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCher2StridedBatched
    :outline:
.. doxygenfunction:: hipblasZher2StridedBatched

The ``her2StridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_hpmv:

hipblasXhpmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChpmv
    :outline:
.. doxygenfunction:: hipblasZhpmv

The ``hpmv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChpmvBatched
    :outline:
.. doxygenfunction:: hipblasZhpmvBatched

The ``hpmvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChpmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhpmvStridedBatched

The ``hpmvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_hpr:

hipblasXhpr + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasChpr
    :outline:
.. doxygenfunction:: hipblasZhpr

The ``hpr`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChprBatched
    :outline:
.. doxygenfunction:: hipblasZhprBatched

The ``hprBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChprStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhprStridedBatched

The ``hprStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_hpr2:

hipblasXhpr2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChpr2
    :outline:
.. doxygenfunction:: hipblasZhpr2

The ``hpr2`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChpr2Batched
    :outline:
.. doxygenfunction:: hipblasZhpr2Batched

The ``hpr2Batched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChpr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasZhpr2StridedBatched

The ``hpr2StridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_sbmv:

hipblasXsbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsbmv
    :outline:
.. doxygenfunction:: hipblasDsbmv

The ``sbmv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsbmvBatched
    :outline:
.. doxygenfunction:: hipblasDsbmvBatched

The ``sbmvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsbmvStridedBatched

The ``sbmvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_spmv:

hipblasXspmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSspmv
    :outline:
.. doxygenfunction:: hipblasDspmv

The ``spmv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSspmvBatched
    :outline:
.. doxygenfunction:: hipblasDspmvBatched

The ``spmvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSspmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDspmvStridedBatched

The ``spmvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_spr:

hipblasXspr + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSspr
    :outline:
.. doxygenfunction:: hipblasDspr
    :outline:
.. doxygenfunction:: hipblasCspr
    :outline:
.. doxygenfunction:: hipblasZspr

The ``spr`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsprBatched
    :outline:
.. doxygenfunction:: hipblasDsprBatched
    :outline:
.. doxygenfunction:: hipblasCsprBatched
    :outline:
.. doxygenfunction:: hipblasZsprBatched

The ``sprBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsprStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsprStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsprStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsprStridedBatched

The ``sprStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_spr2:

hipblasXspr2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSspr2
    :outline:
.. doxygenfunction:: hipblasDspr2

The ``spr2`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSspr2Batched
    :outline:
.. doxygenfunction:: hipblasDspr2Batched

The ``spr2Batched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSspr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasDspr2StridedBatched

The ``spr2StridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_symv:

hipblasXsymv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsymv
    :outline:
.. doxygenfunction:: hipblasDsymv
    :outline:
.. doxygenfunction:: hipblasCsymv
    :outline:
.. doxygenfunction:: hipblasZsymv

The ``symv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsymvBatched
    :outline:
.. doxygenfunction:: hipblasDsymvBatched
    :outline:
.. doxygenfunction:: hipblasCsymvBatched
    :outline:
.. doxygenfunction:: hipblasZsymvBatched

The ``symvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsymvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsymvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsymvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsymvStridedBatched

The ``symvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_syr:

hipblasXsyr + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsyr
    :outline:
.. doxygenfunction:: hipblasDsyr
    :outline:
.. doxygenfunction:: hipblasCsyr
    :outline:
.. doxygenfunction:: hipblasZsyr

The ``syr`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyrBatched
    :outline:
.. doxygenfunction:: hipblasDsyrBatched
    :outline:
.. doxygenfunction:: hipblasCsyrBatched
    :outline:
.. doxygenfunction:: hipblasZsyrBatched

The ``syrBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyrStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyrStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyrStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyrStridedBatched

The ``syrStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_syr2:

hipblasXsyr2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsyr2
    :outline:
.. doxygenfunction:: hipblasDsyr2
    :outline:
.. doxygenfunction:: hipblasCsyr2
    :outline:
.. doxygenfunction:: hipblasZsyr2

The ``syr2`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyr2Batched
    :outline:
.. doxygenfunction:: hipblasDsyr2Batched
    :outline:
.. doxygenfunction:: hipblasCsyr2Batched
    :outline:
.. doxygenfunction:: hipblasZsyr2Batched

The ``syr2Batched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyr2StridedBatched

The ``syr2StridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_tbmv:

hipblasXtbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStbmv
    :outline:
.. doxygenfunction:: hipblasDtbmv
    :outline:
.. doxygenfunction:: hipblasCtbmv
    :outline:
.. doxygenfunction:: hipblasZtbmv

The ``tbmv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStbmvBatched
    :outline:
.. doxygenfunction:: hipblasDtbmvBatched
    :outline:
.. doxygenfunction:: hipblasCtbmvBatched
    :outline:
.. doxygenfunction:: hipblasZtbmvBatched

The ``tbmvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtbmvStridedBatched

The ``tbmvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_tbsv:

hipblasXtbsv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStbsv
    :outline:
.. doxygenfunction:: hipblasDtbsv
    :outline:
.. doxygenfunction:: hipblasCtbsv
    :outline:
.. doxygenfunction:: hipblasZtbsv

The ``tbsv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStbsvBatched
    :outline:
.. doxygenfunction:: hipblasDtbsvBatched
    :outline:
.. doxygenfunction:: hipblasCtbsvBatched
    :outline:
.. doxygenfunction:: hipblasZtbsvBatched

The ``tbsvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStbsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtbsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtbsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtbsvStridedBatched

The ``tbsvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_tpmv:

hipblasXtpmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStpmv
    :outline:
.. doxygenfunction:: hipblasDtpmv
    :outline:
.. doxygenfunction:: hipblasCtpmv
    :outline:
.. doxygenfunction:: hipblasZtpmv

The ``tpmv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStpmvBatched
    :outline:
.. doxygenfunction:: hipblasDtpmvBatched
    :outline:
.. doxygenfunction:: hipblasCtpmvBatched
    :outline:
.. doxygenfunction:: hipblasZtpmvBatched

The ``tpmvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStpmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtpmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtpmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtpmvStridedBatched

The ``tpmvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_tpsv:

hipblasXtpsv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStpsv
    :outline:
.. doxygenfunction:: hipblasDtpsv
    :outline:
.. doxygenfunction:: hipblasCtpsv
    :outline:
.. doxygenfunction:: hipblasZtpsv

The ``tpsv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStpsvBatched
    :outline:
.. doxygenfunction:: hipblasDtpsvBatched
    :outline:
.. doxygenfunction:: hipblasCtpsvBatched
    :outline:
.. doxygenfunction:: hipblasZtpsvBatched

The ``tpsvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStpsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtpsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtpsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtpsvStridedBatched

The ``tpsvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_trmv:

hipblasXtrmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrmv
    :outline:
.. doxygenfunction:: hipblasDtrmv
    :outline:
.. doxygenfunction:: hipblasCtrmv
    :outline:
.. doxygenfunction:: hipblasZtrmv

The ``trmv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStrmvBatched
    :outline:
.. doxygenfunction:: hipblasDtrmvBatched
    :outline:
.. doxygenfunction:: hipblasCtrmvBatched
    :outline:
.. doxygenfunction:: hipblasZtrmvBatched

The ``trmvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStrmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrmvStridedBatched

The ``trmvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_trsv:

hipblasXtrsv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrsv
    :outline:
.. doxygenfunction:: hipblasDtrsv
    :outline:
.. doxygenfunction:: hipblasCtrsv
    :outline:
.. doxygenfunction:: hipblasZtrsv

The ``trsv`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStrsvBatched
    :outline:
.. doxygenfunction:: hipblasDtrsvBatched
    :outline:
.. doxygenfunction:: hipblasCtrsvBatched
    :outline:
.. doxygenfunction:: hipblasZtrsvBatched

The ``trsvBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStrsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrsvStridedBatched

The ``trsvStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _level-3:

Level 3 BLAS
============
.. contents:: List of Level-3 BLAS functions
   :local:
   :backlinks: top

.. _hipblas_gemm:

hipblasXgemm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasHgemm
    :outline:
.. doxygenfunction:: hipblasSgemm
    :outline:
.. doxygenfunction:: hipblasDgemm
    :outline:
.. doxygenfunction:: hipblasCgemm
    :outline:
.. doxygenfunction:: hipblasZgemm

The ``gemm`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasHgemmBatched
    :outline:
.. doxygenfunction:: hipblasSgemmBatched
    :outline:
.. doxygenfunction:: hipblasDgemmBatched
    :outline:
.. doxygenfunction:: hipblasCgemmBatched
    :outline:
.. doxygenfunction:: hipblasZgemmBatched

The ``gemmBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasHgemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasSgemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgemmStridedBatched

The ``gemmStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_herk:

hipblasXherk + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasCherk
    :outline:
.. doxygenfunction:: hipblasZherk

The ``herk`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCherkBatched
    :outline:
.. doxygenfunction:: hipblasZherkBatched

The ``herkBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCherkStridedBatched
    :outline:
.. doxygenfunction:: hipblasZherkStridedBatched

The ``herkStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_herkx:

hipblasXherkx + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasCherkx
    :outline:
.. doxygenfunction:: hipblasZherkx

The ``herkx`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCherkxBatched
    :outline:
.. doxygenfunction:: hipblasZherkxBatched

The ``herkxBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCherkxStridedBatched
    :outline:
.. doxygenfunction:: hipblasZherkxStridedBatched

The ``herkxStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_her2k:

hipblasXher2k + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasCher2k
    :outline:
.. doxygenfunction:: hipblasZher2k

The ``her2k`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCher2kBatched
    :outline:
.. doxygenfunction:: hipblasZher2kBatched

The ``her2kBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasCher2kStridedBatched
    :outline:
.. doxygenfunction:: hipblasZher2kStridedBatched

The ``her2kStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_symm:

hipblasXsymm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsymm
    :outline:
.. doxygenfunction:: hipblasDsymm
    :outline:
.. doxygenfunction:: hipblasCsymm
    :outline:
.. doxygenfunction:: hipblasZsymm

The ``symm`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsymmBatched
    :outline:
.. doxygenfunction:: hipblasDsymmBatched
    :outline:
.. doxygenfunction:: hipblasCsymmBatched
    :outline:
.. doxygenfunction:: hipblasZsymmBatched

The ``symmBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsymmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsymmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsymmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsymmStridedBatched

The ``symmStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_syrk:

hipblasXsyrk + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsyrk
    :outline:
.. doxygenfunction:: hipblasDsyrk
    :outline:
.. doxygenfunction:: hipblasCsyrk
    :outline:
.. doxygenfunction:: hipblasZsyrk

The ``syrk`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyrkBatched
    :outline:
.. doxygenfunction:: hipblasDsyrkBatched
    :outline:
.. doxygenfunction:: hipblasCsyrkBatched
    :outline:
.. doxygenfunction:: hipblasZsyrkBatched

The ``syrkBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyrkStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyrkStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyrkStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyrkStridedBatched

The ``syrkStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_syr2k:

hipblasXsyr2k + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasSsyr2k
    :outline:
.. doxygenfunction:: hipblasDsyr2k
    :outline:
.. doxygenfunction:: hipblasCsyr2k
    :outline:
.. doxygenfunction:: hipblasZsyr2k

The ``syr2k`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyr2kBatched
    :outline:
.. doxygenfunction:: hipblasDsyr2kBatched
    :outline:
.. doxygenfunction:: hipblasCsyr2kBatched
    :outline:
.. doxygenfunction:: hipblasZsyr2kBatched

The ``syr2kBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyr2kStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyr2kStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyr2kStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyr2kStridedBatched

The ``syr2kStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_syrkx:

hipblasXsyrkx + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasSsyrkx
    :outline:
.. doxygenfunction:: hipblasDsyrkx
    :outline:
.. doxygenfunction:: hipblasCsyrkx
    :outline:
.. doxygenfunction:: hipblasZsyrkx

The ``syrkx`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyrkxBatched
    :outline:
.. doxygenfunction:: hipblasDsyrkxBatched
    :outline:
.. doxygenfunction:: hipblasCsyrkxBatched
    :outline:
.. doxygenfunction:: hipblasZsyrkxBatched

The ``syrkxBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSsyrkxStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyrkxStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyrkxStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyrkxStridedBatched

The ``syrkxStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_geam:

hipblasXgeam + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgeam
    :outline:
.. doxygenfunction:: hipblasDgeam
    :outline:
.. doxygenfunction:: hipblasCgeam
    :outline:
.. doxygenfunction:: hipblasZgeam

The ``geam`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSgeamBatched
    :outline:
.. doxygenfunction:: hipblasDgeamBatched
    :outline:
.. doxygenfunction:: hipblasCgeamBatched
    :outline:
.. doxygenfunction:: hipblasZgeamBatched

The ``geamBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSgeamStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgeamStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgeamStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgeamStridedBatched

The ``geamStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_hemm:

hipblasXhemm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChemm
    :outline:
.. doxygenfunction:: hipblasZhemm

The ``hemm`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChemmBatched
    :outline:
.. doxygenfunction:: hipblasZhemmBatched

The ``hemmBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasChemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhemmStridedBatched

The ``hemmStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_trmm:

hipblasXtrmm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrmm
    :outline:
.. doxygenfunction:: hipblasDtrmm
    :outline:
.. doxygenfunction:: hipblasCtrmm
    :outline:
.. doxygenfunction:: hipblasZtrmm

The ``trmm`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStrmmBatched
    :outline:
.. doxygenfunction:: hipblasDtrmmBatched
    :outline:
.. doxygenfunction:: hipblasCtrmmBatched
    :outline:
.. doxygenfunction:: hipblasZtrmmBatched

The ``trmmBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStrmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrmmStridedBatched

The ``trmmStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_trsm:

hipblasXtrsm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrsm
    :outline:
.. doxygenfunction:: hipblasDtrsm
    :outline:
.. doxygenfunction:: hipblasCtrsm
    :outline:
.. doxygenfunction:: hipblasZtrsm

The ``trsm`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStrsmBatched
    :outline:
.. doxygenfunction:: hipblasDtrsmBatched
    :outline:
.. doxygenfunction:: hipblasCtrsmBatched
    :outline:
.. doxygenfunction:: hipblasZtrsmBatched

The ``trsmBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasStrsmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrsmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrsmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrsmStridedBatched

The ``trsmStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_trtri:

hipblasXtrtri + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasStrtri
    :outline:
.. doxygenfunction:: hipblasDtrtri
    :outline:
.. doxygenfunction:: hipblasCtrtri
    :outline:
.. doxygenfunction:: hipblasZtrtri

.. doxygenfunction:: hipblasStrtriBatched
    :outline:
.. doxygenfunction:: hipblasDtrtriBatched
    :outline:
.. doxygenfunction:: hipblasCtrtriBatched
    :outline:
.. doxygenfunction:: hipblasZtrtriBatched

.. doxygenfunction:: hipblasStrtriStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrtriStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrtriStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrtriStridedBatched

.. _hipblas_dgmm:

hipblasXdgmm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSdgmm
    :outline:
.. doxygenfunction:: hipblasDdgmm
    :outline:
.. doxygenfunction:: hipblasCdgmm
    :outline:
.. doxygenfunction:: hipblasZdgmm

The ``dgmm`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSdgmmBatched
    :outline:
.. doxygenfunction:: hipblasDdgmmBatched
    :outline:
.. doxygenfunction:: hipblasCdgmmBatched
    :outline:
.. doxygenfunction:: hipblasZdgmmBatched

The ``dgmmBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasSdgmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDdgmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCdgmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdgmmStridedBatched

The ``dgmmStridedBatched`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_extension:

BLAS extensions
===============
.. contents:: List of BLAS extension functions
   :local:
   :backlinks: top

.. _hipblas_gemmex:

hipblasGemmEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasGemmEx
.. doxygenfunction:: hipblasGemmBatchedEx
.. doxygenfunction:: hipblasGemmStridedBatchedEx

The ``gemmEx``, ``gemmBatchedEx``, and ``gemmStridedBatchedEx`` functions support the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_syrkex:

hipblasSyrkEx
------------------------------------------
.. doxygenfunction:: hipblasSyrkEx

.. _hipblas_herkex:
   
hipblasHerkEx
------------------------------------------
.. doxygenfunction:: hipblasHerkEx

.. _hipblas_trsmex:

hipblasTrsmEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasTrsmEx
.. doxygenfunction:: hipblasTrsmBatchedEx
.. doxygenfunction:: hipblasTrsmStridedBatchedEx

.. _hipblas_axpyex:

hipblasAxpyEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasAxpyEx

The ``axpyEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasAxpyBatchedEx

The ``axpyBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasAxpyStridedBatchedEx

The ``axpyStridedBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_dotex:

hipblasDotEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasDotEx

The ``dotEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasDotBatchedEx

The ``dotBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasDotStridedBatchedEx

The ``dotStridedBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_dotcex:

hipblasDotcEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasDotcEx

The ``dotcEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasDotcBatchedEx

The ``dotcBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasDotcStridedBatchedEx

The ``dotcStridedBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_nrm2ex:

hipblasNrm2Ex + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasNrm2Ex

The ``nrm2Ex`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasNrm2BatchedEx

The ``nrm2BatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasNrm2StridedBatchedEx

The ``nrm2StridedBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_rotex:

hipblasRotEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasRotEx

The ``rotEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasRotBatchedEx

The ``rotBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasRotStridedBatchedEx

The ``rotStridedBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _hipblas_scalex:

hipblasScalEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasScalEx

The ``scalEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasScalBatchedEx

The ``scalBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. doxygenfunction:: hipblasScalStridedBatchedEx

The ``scalStridedBatchedEx`` function supports the 64-bit integer interface. See the :ref:`ILP64 API` section.

.. _solver_api:

SOLVER API
===========
.. contents:: List of SOLVER APIs
   :local:
   :backlinks: top

.. _hipblas_getrf:

hipblasXgetrf + Batched, stridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgetrf
    :outline:
.. doxygenfunction:: hipblasDgetrf
    :outline:
.. doxygenfunction:: hipblasCgetrf
    :outline:
.. doxygenfunction:: hipblasZgetrf

.. doxygenfunction:: hipblasSgetrfBatched
    :outline:
.. doxygenfunction:: hipblasDgetrfBatched
    :outline:
.. doxygenfunction:: hipblasCgetrfBatched
    :outline:
.. doxygenfunction:: hipblasZgetrfBatched

.. doxygenfunction:: hipblasSgetrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgetrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgetrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgetrfStridedBatched

.. _hipblas_getrs:

hipblasXgetrs + Batched, stridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgetrs
    :outline:
.. doxygenfunction:: hipblasDgetrs
    :outline:
.. doxygenfunction:: hipblasCgetrs
    :outline:
.. doxygenfunction:: hipblasZgetrs

.. doxygenfunction:: hipblasSgetrsBatched
    :outline:
.. doxygenfunction:: hipblasDgetrsBatched
    :outline:
.. doxygenfunction:: hipblasCgetrsBatched
    :outline:
.. doxygenfunction:: hipblasZgetrsBatched

.. doxygenfunction:: hipblasSgetrsStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgetrsStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgetrsStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgetrsStridedBatched

.. _hipblas_getri:

hipblasXgetri + Batched, stridedBatched
----------------------------------------

.. doxygenfunction:: hipblasSgetriBatched
    :outline:
.. doxygenfunction:: hipblasDgetriBatched
    :outline:
.. doxygenfunction:: hipblasCgetriBatched
    :outline:
.. doxygenfunction:: hipblasZgetriBatched

.. _hipblas_geqrf:

hipblasXgeqrf + Batched, stridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgeqrf
    :outline:
.. doxygenfunction:: hipblasDgeqrf
    :outline:
.. doxygenfunction:: hipblasCgeqrf
    :outline:
.. doxygenfunction:: hipblasZgeqrf

.. doxygenfunction:: hipblasSgeqrfBatched
    :outline:
.. doxygenfunction:: hipblasDgeqrfBatched
    :outline:
.. doxygenfunction:: hipblasCgeqrfBatched
    :outline:
.. doxygenfunction:: hipblasZgeqrfBatched

.. doxygenfunction:: hipblasSgeqrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgeqrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgeqrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgeqrfStridedBatched

.. _hipblas_gels:

hipblasXgels + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgels
    :outline:
.. doxygenfunction:: hipblasDgels
    :outline:
.. doxygenfunction:: hipblasCgels
    :outline:
.. doxygenfunction:: hipblasZgels

.. doxygenfunction:: hipblasSgelsBatched
    :outline:
.. doxygenfunction:: hipblasDgelsBatched
    :outline:
.. doxygenfunction:: hipblasCgelsBatched
    :outline:
.. doxygenfunction:: hipblasZgelsBatched

.. doxygenfunction:: hipblasSgelsStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgelsStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgelsStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgelsStridedBatched

Auxiliary
=========

hipblasCreate
--------------
.. doxygenfunction:: hipblasCreate

hipblasDestroy
---------------
.. doxygenfunction:: hipblasDestroy

hipblasSetStream
-----------------
.. doxygenfunction:: hipblasSetStream

hipblasGetStream
------------------
.. doxygenfunction:: hipblasGetStream

hipblasSetPointerMode
----------------------
.. doxygenfunction:: hipblasSetPointerMode

hipblasGetPointerMode
----------------------
.. doxygenfunction:: hipblasGetPointerMode

hipblasSetVector
----------------
.. doxygenfunction:: hipblasSetVector

hipblasGetVector
-----------------
.. doxygenfunction:: hipblasGetVector

hipblasSetMatrix
-----------------
.. doxygenfunction:: hipblasSetMatrix

hipblasGetMatrix
------------------
.. doxygenfunction:: hipblasGetMatrix

hipblasSetVectorAsync
----------------------
.. doxygenfunction:: hipblasSetVectorAsync

hipblasGetVectorAsync
----------------------
.. doxygenfunction:: hipblasGetVectorAsync

hipblasSetMatrixAsync
-----------------------
.. doxygenfunction:: hipblasSetMatrixAsync

hipblasGetMatrixAsync
---------------------
.. doxygenfunction:: hipblasGetMatrixAsync

hipblasSetAtomicsMode
----------------------
.. doxygenfunction:: hipblasSetAtomicsMode

hipblasGetAtomicsMode
----------------------
.. doxygenfunction:: hipblasGetAtomicsMode

hipblasStatusToString
----------------------
.. doxygenfunction:: hipblasStatusToString
