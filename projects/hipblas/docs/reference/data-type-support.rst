.. meta::
  :description: hipBLAS library data type support
  :keywords: hipBLAS, ROCm, API, Linear Algebra, documentation, data type support

.. _data-types-support:

********************************************************************
Data type support
********************************************************************

This topic lists the data type support for the hipBLAS library on AMD GPUs for
different levels of BLAS operations :ref:`Level 1 <level-1>`,
:ref:`Level 2 <level-2>`, and :ref:`Level 3 <level-3>`.

The hipBLAS library functions are also available with ILP64 interfaces. With
these interfaces, all ``int`` arguments are replaced by the type name
``int64_t``. For more information on these ``_64`` functions, see the
:ref:`ILP64 API` section.

The icons representing different levels of support are explained in the
following table.

.. list-table::
    :header-rows: 1

    *
      -  Icon
      - Definition

    *
      - NA
      - Not applicable

    *
      - ❌
      - Not supported

    *
      - ⚠️
      - Partial support

    *
      - ✅
      - Full support


For more information about data type support for the other ROCm libraries, see
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _custom_types:

Custom types
============

hipBlas defines the ``hipblasBfloat16`` and uses the HIP types ``hipComplex``
and ``hipDoubleComplex``.

.. _hipblas_bfloat16:

The bfloat 16 data type
-----------------------

hipBLAS defines a ``hipblasBfloat16`` data type. This type is exposed as a
struct containing 16 bits of data. There is also a C++ ``hipblasBfloat16`` class
defined which provides slightly more functionality, including conversion to and
from a 32-bit float data type.
This class can be used in C++11 or newer by defining ``HIPBLAS_BFLOAT16_CLASS``
before including the header file ``<hipblas.h>``.

There is also an option to interpret the API as using the ``hip_bfloat16`` data
type. This is provided to avoid casting when using the ``hip_bfloat16`` data
type. To expose the API using ``hip_bfloat16``, define
``HIPBLAS_USE_HIP_BFLOAT16`` before including the header file ``<hipblas.h>``.

.. note::

   The ``hip_bfloat16`` data type is only supported on AMD platforms.

.. _hipblas_complex:

Complex data types
------------------

hipBLAS uses the HIP types ``hipComplex`` and ``hipDoubleComplex`` in its API.

Functions data type support
=======================================

This section collects the data type support tables of BLAS functions and
solver functions. The cuBLAS backend does not support all the functions and all
the types. For more information, see :ref:`cuBLAS backend <hipblas-backend>`.

Level 1 functions - vector operations
-------------------------------------

Level-1 functions perform scalar, vector, and vector-vector operations.

.. tab-set::

  .. tab-item:: Float types
    :sync: float-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 45, 10, 10, 10, 10

        *
          - Function
          - Description
          - float16
          - bfloat16
          - float
          - double

        *
          - :ref:`AMax <hipblas_amax>`, :ref:`AMin <hipblas_amin>`, :ref:`ASum <hipblas_asum>`
          - Finds the first index of the element of minimum or maximum magnitude of a vector x or computes the sum of the magnitudes of elements of a real vector x.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`AXPY <hipblas_axpy>`
          - Scales a vector and adds it to another: :math:`y = \alpha x + y`
          - ✅
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Copy <hipblas_copy>`
          - Copies vector x to y: :math:`y = x`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Dot <hipblas_dot>`
          - Computes the dot product: :math:`result = x^T y`
          - ✅
          - ✅
          - ✅
          - ✅

        *
          - :ref:`NRM2 <hipblas_nrm2>`
          - Computes the Euclidean norm of a vector.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Rot <hipblas_rot>`, :ref:`Rotg <hipblas_rotg>`
          - Applies and generates a Givens rotation matrix.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Rotm <hipblas_rotm>`, :ref:`Rotmg <hipblas_rotmg>`
          - Applies and generates a modified Givens rotation matrix.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Scal <hipblas_scal>`
          - Scales a vector by a scalar: :math:`x = \alpha x`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Swap <hipblas_swap>`
          - Swaps corresponding elements of two vectors x and y: :math:`x_i \leftrightarrow y_i \quad \text{for} \quad i = 0, 1, 2, \ldots, n - 1`
          - ❌
          - ❌
          - ✅
          - ✅

  .. tab-item:: Complex types
    :sync: complex-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 55, 10, 20

        *
          - Function
          - Description
          - complex
          - double complex

        *
          - :ref:`AMax <hipblas_amax>`, :ref:`AMin <hipblas_amin>`, :ref:`ASum <hipblas_asum>`
          - Finds the first index of the element of minimum or maximum magnitude of a vector x or computes the sum of the magnitudes of elements of a real vector x.
          - ✅
          - ✅

        *
          - :ref:`AXPY <hipblas_axpy>`
          - Scales a vector and adds it to another: :math:`y = \alpha x + y`
          - ✅
          - ✅

        *
          - :ref:`Copy <hipblas_copy>`
          - Copies vector x to y: :math:`y = x`
          - ✅
          - ✅

        *
          - :ref:`Dot <hipblas_dot>`
          - Computes the dot product: :math:`result = x^T y`
          - ✅
          - ✅

        *
          - :ref:`NRM2 <hipblas_nrm2>`
          - Computes the Euclidean norm of a vector.
          - ✅
          - ✅

        *
          - :ref:`Rot <hipblas_rot>`, :ref:`Rotg <hipblas_rotg>`
          - Applies and generates a Givens rotation matrix.
          - ✅
          - ✅

        *
          - :ref:`Rotm <hipblas_rotm>`, :ref:`Rotmg <hipblas_rotmg>`
          - Applies and generates a modified Givens rotation matrix.
          - ❌
          - ❌

        *
          - :ref:`Scal <hipblas_scal>`
          - Scales a vector by a scalar: :math:`x = \alpha x`
          - ✅
          - ✅

        *
          - :ref:`Swap <hipblas_swap>`
          - Swaps corresponding elements of two vectors x and y: :math:`x_i \leftrightarrow y_i \quad \text{for} \quad i = 0, 1, 2, \ldots, n - 1`
          - ✅
          - ✅

Level 2 functions - matrix-vector operations
--------------------------------------------

Level-2 functions perform matrix-vector operations.

.. tab-set::

  .. tab-item:: Float types
    :sync: float-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 45, 10, 10, 10, 10

        *
          - Function
          - Description
          - float16
          - bfloat16
          - float
          - double

        *
          - :ref:`GBMV <hipblas_gbmv>`
          - General band matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`GEMV <hipblas_gemv>`
          - General matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`GER <hipblas_ger>`
          - Generalized rank-1 update: :math:`A = \alpha x y^T + A`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`GERU and GERC <hipblas_ger>`
          - Generalized rank-1 update for unconjugated or conjugated complex numbers: :math:`A = \alpha x y^T + A`
          - NA
          - NA
          - NA
          - NA

        *
          - :ref:`SBMV <hipblas_sbmv>`, :ref:`SPMV <hipblas_spmv>`
          - Symmetric Band matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SPR <hipblas_spr>`
          - Symmetric packed rank-1 update.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SPR2 <hipblas_spr2>`
          - Symmetric packed rank-2 update.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYMV <hipblas_symv>`
          - Symmetric matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYR <hipblas_syr>`, :ref:`SYR2 <hipblas_syr2>`
          - Symmetric matrix rank-1 or rank-2 update.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TBMV <hipblas_tbmv>`, :ref:`TBSV <hipblas_tbsv>`
          - Triangular band matrix-vector multiplication, triangular band solve.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TPMV <hipblas_tpmv>`, :ref:`TPSV <hipblas_tpsv>`
          - Triangular packed matrix-vector multiplication, triangular packed solve.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TRMV <hipblas_trmv>`, :ref:`TRSV <hipblas_trsv>`
          - Triangular matrix-vector multiplication, triangular solve.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`HEMV <hipblas_hemv>`, :ref:`HBMV <hipblas_hbmv>`, :ref:`HPMV <hipblas_hpmv>`
          - Hermitian matrix-vector multiplication.
          - NA
          - NA
          - NA
          - NA

        *
          - :ref:`HER <hipblas_her>`, :ref:`HER2 <hipblas_her2>`
          - Hermitian rank-1 and rank-2 update.
          - NA
          - NA
          - NA
          - NA


        *
          - :ref:`HPR <hipblas_hpr>`, :ref:`HPR2 <hipblas_hpr2>`
          - Hermitian packed rank-1 and rank-2 update of packed.
          - NA
          - NA
          - NA
          - NA



  .. tab-item:: Complex types
    :sync: complex-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 55, 10, 20

        *
          - Function
          - Description
          - complex
          - double complex

        *
          - :ref:`GBMV <hipblas_gbmv>`
          - General band matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ✅
          - ✅

        *
          - :ref:`GEMV <hipblas_gemv>`
          - General matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ✅
          - ✅

        *
          - :ref:`GER <hipblas_ger>`
          - Generalized rank-1 update: :math:`A = \alpha x y^T + A`
          - NA
          - NA

        *
          - :ref:`GERU and GERC <hipblas_ger>`
          - Generalized rank-1 update for unconjugated or conjugated complex numbers: :math:`A = \alpha x y^T + A`
          - ✅
          - ✅

        *
          - :ref:`SBMV <hipblas_sbmv>`, :ref:`SPMV <hipblas_spmv>`
          - Symmetric Band matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ❌
          - ❌

        *
          - :ref:`SPR <hipblas_spr>`
          - Symmetric packed rank-1 update.
          - ✅
          - ✅

        *
          - :ref:`SPR2 <hipblas_spr2>`
          - Symmetric packed rank-2 update.
          - ❌
          - ❌

        *
          - :ref:`SYMV <hipblas_symv>`
          - Symmetric matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ✅
          - ✅

        *
          - :ref:`SYR <hipblas_syr>`, :ref:`SYR2 <hipblas_syr2>`
          - Symmetric matrix rank-1 or rank-2 update.
          - ✅
          - ✅

        *
          - :ref:`TBMV <hipblas_tbmv>`, :ref:`TBSV <hipblas_tbsv>`
          - Triangular band matrix-vector multiplication, triangular band solve.
          - ✅
          - ✅

        *
          - :ref:`TPMV <hipblas_tpmv>`, :ref:`TPSV <hipblas_tpsv>`
          - Triangular packed matrix-vector multiplication, triangular packed solve.
          - ✅
          - ✅

        *
          - :ref:`TRMV <hipblas_trmv>`, :ref:`TRSV <hipblas_trsv>`
          - Triangular matrix-vector multiplication, triangular solve.
          - ✅
          - ✅

        *
          - :ref:`HEMV <hipblas_hemv>`, :ref:`HBMV <hipblas_hbmv>`, :ref:`HPMV <hipblas_hpmv>`
          - Hermitian matrix-vector multiplication.
          - ✅
          - ✅

        *
          - :ref:`HER <hipblas_her>`, :ref:`HER2 <hipblas_her2>`
          - Hermitian rank-1 and rank-2 update.
          - ✅
          - ✅

        *
          - :ref:`HPR <hipblas_hpr>`, :ref:`HPR2 <hipblas_hpr2>`
          - Hermitian packed rank-1 and rank-2 update.
          - ✅
          - ✅

Level 3 functions - matrix-matrix operations
--------------------------------------------

Level-3 functions perform matix-matrix operations. hipBLAS calls the AMD
:doc:`Tensile <tensile:src/index>` and :doc:`hipBLASLt <hipblaslt:index>`
libraries for Level-3 GEMMs (matrix matrix multiplication).

.. tab-set::

  .. tab-item:: Float types
    :sync: float-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 45, 10, 10, 10, 10

        *
          - Function
          - Description
          - float16
          - bfloat16
          - float
          - double

        *
          - :ref:`GEMM <hipblas_gemm>`
          - General matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
          - ✅
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYMM <hipblas_symm>`
          - Symmetric matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYRK <hipblas_syrk>`, :ref:`SYR2K <hipblas_syr2k>`
          - Update symmetric matrix with one matrix product or by using two matrices.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYRKX <hipblas_syrkx>`
          - SYRKX adds an extra matrix multiplication step before updating the symmetric matrix.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TRMM <hipblas_trmm>`
          - Triangular matrix-matrix multiplication.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TRSM <hipblas_trsm>`
          - Triangular solve with multiple right-hand sides.
          - ❌
          - ❌
          - ✅
          - ✅
        *
          - :ref:`HEMM <hipblas_hemm>`
          - Hermitian matrix-matrix multiplication.
          - NA
          - NA
          - NA
          - NA

        *
          - :ref:`HERK <hipblas_herk>`, :ref:`HER2K <hipblas_her2k>`
          - Update Hermitian matrix with one matrix product or by using two matrices.
          - NA
          - NA
          - NA
          - NA

        *
          - :ref:`HERKX <hipblas_herkx>`
          - HERKX adds an extra matrix multiplication step before updating the Hermitian matrix.
          - NA
          - NA
          - NA
          - NA

        *
          - :ref:`TRTRI <hipblas_trtri>`
          - Triangular matrix inversion.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`DGMM <hipblas_dgmm>`
          - Diagonal matrix matrix multiplication.
          - ❌
          - ❌
          - ✅
          - ✅

  .. tab-item:: Complex types
    :sync: complex-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 55, 10, 20

        *
          - Function
          - Description
          - complex
          - double complex

        *
          - :ref:`GEMM <hipblas_gemm>`
          - General matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
          - ✅
          - ✅

        *
          - :ref:`SYMM <hipblas_symm>`
          - Symmetric matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
          - ✅
          - ✅

        *
          - :ref:`SYRK <hipblas_syrk>`, :ref:`SYR2K <hipblas_syr2k>`
          - Update symmetric matrix with one matrix product or by using two matrices.
          - ✅
          - ✅

        *
          - :ref:`SYRKX <hipblas_syrkx>`
          - SYRKX adds an extra matrix multiplication step before updating the symmetric matrix.
          - ✅
          - ✅

        *
          - :ref:`TRMM <hipblas_trmm>`
          - Triangular matrix-matrix multiplication.
          - ✅
          - ✅

        *
          - :ref:`TRSM <hipblas_trsm>`
          - Triangular solve with multiple right-hand sides.
          - ✅
          - ✅
        *
          - :ref:`HEMM <hipblas_hemm>`
          - Hermitian matrix-matrix multiplication.
          - ✅
          - ✅

        *
          - :ref:`HERK <hipblas_herk>`, :ref:`HER2K <hipblas_her2k>`
          - Update Hermitian matrix with one matrix product or by using two matrices.
          - ✅
          - ✅

        *
          - :ref:`HERKX <hipblas_herkx>`
          - HERKX adds an extra matrix multiplication step before updating the Hermitian matrix.
          - ✅
          - ✅

        *
          - :ref:`TRTRI <hipblas_trtri>`
          - Triangular matrix inversion.
          - ✅
          - ✅

        *
          - :ref:`DGMM <hipblas_dgmm>`
          - Diagonal matrix matrix multiplication.
          - ✅
          - ✅


Extensions
----------

The :ref:`extension functions <hipblas_extension>` data type support is listed
separately for every function for the different backends in the
:ref:`rocBLAS extensions <rocblas:extension>` and
`cuBLAS extensions <https://docs.nvidia.com/cuda/cublas/index.html#blas-like-extension>`_
documentation.

SOLVER API
----------

:ref:`Solver API <solver_api>` is for solving linear systems, computing matrix
inverses, and performing matrix factorizations.

.. tab-set::

  .. tab-item:: Float types
    :sync: float-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 45, 10, 10, 10, 10

        *
          - Function
          - Description
          - float16
          - bfloat16
          - float
          - double

        *
          - :ref:`GETRF <hipblas_getrf>`
          - Compute the LU factorization of a general matrix using partial pivoting with row interchanges.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`GETRS <hipblas_getrs>`
          - Solve a system of linear equations :math:`AxX=B` after performing an LU factorization using GETRF.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`GETRI <hipblas_getri>`
          - Compute the inverse of a matrix using its LU factorization.
          - ❌
          - ❌
          - ⚠️ [#getri]_
          - ⚠️ [#getri]_

        *
          - :ref:`GEQRF <hipblas_geqrf>`
          - QR factorization of a general matrix.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`GELS <hipblas_gels>`
          - Solve overdetermined or underdetermined linear systems using the QR factorization of a matrix.
          - ❌
          - ❌
          - ✅
          - ✅

  .. tab-item:: Complex types
    :sync: complex-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 55, 10, 20

        *
          - Function
          - Description
          - complex
          - double complex

        *
          - :ref:`GETRF <hipblas_getrf>`
          - Compute the LU factorization of a general matrix using partial pivoting with row interchanges.
          - ✅
          - ✅

        *
          - :ref:`GETRS <hipblas_getrs>`
          - Solve a system of linear equations :math:`AxX=B` after performing an LU factorization using GETRF.
          - ✅
          - ✅

        *
          - :ref:`GETRI <hipblas_getri>`
          - Compute the inverse of a matrix using its LU factorization.
          - ⚠️ [#getri]_
          - ⚠️ [#getri]_

        *
          - :ref:`GEQRF <hipblas_geqrf>`
          - QR factorization of a general matrix.
          - ✅
          - ✅

        *
          - :ref:`GELS <hipblas_gels>`
          - Solve overdetermined or underdetermined linear systems using the QR factorization of a matrix.
          - ✅
          - ✅

.. rubric:: Footnotes

.. [#getri] Only the batched GETRI functions are supported.
