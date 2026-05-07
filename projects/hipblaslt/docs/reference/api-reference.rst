.. meta::
   :description: hipBLASLt API reference
   :keywords: hipBLASLt, ROCm, library, API, reference

.. _api-reference:

***********************
hipBLASLt API reference
***********************

hipblasLtCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtCreate

hipblasLtDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtDestroy

hipblasLtMatrixLayoutCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutCreate

hipblasLtMatrixLayoutDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutDestroy

hipblasLtMatrixLayoutSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutSetAttribute

hipblasLtMatrixLayoutGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutGetAttribute

hipblasLtMatmulDescCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescCreate

hipblasLtMatmulDescDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescDestroy

hipblasLtMatmulDescSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescSetAttribute

hipblasLtMatmulDescGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescGetAttribute

hipblasLtMatmulPreferenceCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceCreate

hipblasLtMatmulPreferenceDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceDestroy

hipblasLtMatmulPreferenceSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceSetAttribute

hipblasLtMatmulPreferenceGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceGetAttribute

.. _hipblasltmatmulalgogetheuristic:

hipblasLtMatmulAlgoGetHeuristic()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulAlgoGetHeuristic

.. _hipblasltmatmul:

hipblasLtMatmul()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmul

Supported data types
------------------------------------------

``hipblasLtMatmul`` supports the following computeType, scaleType, Bias type, Atype/Btype, and Ctype/Dtype:

============================= =================== =============== ===============
computeType                   scaleType/Bias type Atype/Btype     Ctype/Dtype
============================= =================== =============== ===============
HIPBLAS_COMPUTE_32F           HIP_R_32F           HIP_R_32F       HIP_R_32F
HIPBLAS_COMPUTE_32F_FAST_TF32 HIP_R_32F           HIP_R_32F       HIP_R_32F
HIPBLAS_COMPUTE_32F           HIP_R_32F           HIP_R_16F       HIP_R_16F
HIPBLAS_COMPUTE_32F           HIP_R_32F           HIP_R_16F       HIP_R_32F
HIPBLAS_COMPUTE_32F           HIP_R_32F           HIP_R_16BF      HIP_R_16BF
============================= =================== =============== ===============

For ``FP8`` type Matmul, hipBLASLt supports the type combinations shown in the following table:

* This table uses simpler abbrieviations:

  *  **FP16** means **HIP_R_16F**
  *  **BF16** means **HIP_R_16BF**
  *  **FP32** means **HIP_R_32F**
  *  **FP8** means **HIP_R_8F_E4M3**
  *  **BF8** means **HIP_R_8F_E5M2**
  *  **FP8_FNUZ** means **HIP_R_8F_E4M3_FNUZ** and
  *  **BF8_FNUZ** means **HIP_R_8F_E5M2_FNUZ**

*  The table applies to all transpose types (NN/NT/TT/TN).
*  **Default bias type** indicates the type when the bias type is not explicitly specified.

+-------+-------+-------+-------+-------------+----------+----------+------------+-----------+
| Atype | Btype | Ctype | Dtype | computeType | scaleA,B | scaleC,D | Bias type  | Default   |
|       |       |       |       |             |          |          |            | bias type |
+=======+=======+=======+=======+=============+==========+==========+======+=====+===========+
| FP8   | FP8   | FP16  | FP16  | FP32        | Yes      | No       | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF16  | BF16  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | FP32  | FP32  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +----------+------------+-----------+
|       |       | FP8   | FP8   |             |          | Yes      | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF8   | BF8   |             |          |          | FP32, FP16 | FP16      |
|       +-------+-------+-------+             +          +----------+------------+-----------+
|       | BF8   | FP16  | FP16  |             |          | No       | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF16  | BF16  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | FP32  | FP32  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +----------+------------+-----------+
|       |       | FP8   | FP8   |             |          | Yes      | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF8   | BF8   |             |          |          | FP32, FP16 | FP16      |
+-------+-------+-------+-------+             +          +----------+------------+-----------+
| BF8   | FP8   | FP16  | FP16  |             |          | No       | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF16  | BF16  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | FP32  | FP32  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +----------+------------+-----------+
|       |       | FP8   | FP8   |             |          | Yes      | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF8   | BF8   |             |          |          | FP32, FP16 | FP16      |
|       +-------+-------+-------+             +          +----------+------------+-----------+
|       | BF8   | FP16  | FP16  |             |          | No       | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF16  | BF16  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | FP32  | FP32  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +----------+------------+-----------+
|       |       | FP8   | FP8   |             |          | Yes      | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF8   | BF8   |             |          |          | FP32, FP16 | FP16      |
+-------+-------+-------+-------+-------------+----------+----------+------------+-----------+

To use special data ordering for ``HIPBLASLT_ORDER_COL16_4R8`` and ``HIPBLASLT_ORDER_COL16_4R16`` in ``hipblasLtMatmul`` for the gfx94x architecture, choose one of these valid combinations of transposes and orders of input and output matrices:

+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
| Atype | Btype | CType | DType |  opA |  opB  |  orderA                     |  orderB                     |  orderC             |   orderD            |
+=======+=======+=======+=======+======+=======+=============================+=============================+=====================+=====================+
|  FP8  | FP8   |  FP16 | FP16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R16 |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP8  | FP8   |  BF16 | BF16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R16 |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP16 | FP16  |  FP32 | FP32  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R8  |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP16 | FP16  |  FP16 | FP16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R8  |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  BF16 | BF16  |  BF16 | BF16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R8  |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP8  | FP8   |  FP16 | FP16  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R16 | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP8  | FP8   |  BF16 | BF16  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R16 | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP16 | FP16  |  FP32 | FP32  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R8  | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP16 | FP16  |  FP16 | FP16  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R8  | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  BF16 | BF16  |  BF16 | BF16  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R8  | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+

There are restrictions on the supported problem sizes for the ``HIP_R_4F_E2M1``, ``HIP_R_6F_E2M3``, ``HIP_R_6F_E3M2``,
``HIP_R_8F_E4M3``, and ``HIP_R_8F_E5M2`` data types.
When ``HIPBLASLT_MATMUL_DESC_A_SCALE_MODE`` and ``HIPBLASLT_MATMUL_DESC_B_SCALE_MODE`` are both set
to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0``, the following restrictions apply:

*  Atype and Btype can be any combination of: ``HIP_R_8F_E4M3``, ``HIP_R_8F_E5M2``, ``HIP_R_6F_E2M3``, ``HIP_R_6F_E3M2``, and ``HIP_R_4F_E2M1``.
*  Ctype must be the same as Dtype.
*  Dtype can be any of the following: ``HIP_R_32F``, ``HIP_R_16F``, or ``HIP_R_16BF``.
*  ``m % 16`` must be equal to ``0``.
*  ``n % 16`` must be equal to ``0``.
*  ``K % 128`` must be equal to ``0``.
*  ``B`` must be equal to ``1``.
*  ``opA`` must be equal to ``T``.
*  ``opB`` must be equal to ``N``.
*  Epilogues are not supported.
*  The scaling data pointed to by ``HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`` must be stored in the same order as ``A``.
*  The scaling data pointed to by ``HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER`` must be stored in the same order as ``B``.

hipblasLtMatrixTransformDescCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescCreate

hipblasLtMatrixTransformDescDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescDestroy

hipblasLtMatrixTransformDescSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescSetAttribute

hipblasLtMatrixTransformDescGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescGetAttribute

hipblasLtMatrixTransform()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransform

``hipblasLtMatrixTransform`` supports the following Atype/Btype/Ctype and scaleType:

======================= ===================
Atype/Btype/Ctype       scaleType
======================= ===================
HIP_R_32F               HIP_R_32F
HIP_R_16F               HIP_R_32F/HIP_R_16F
HIP_R_16BF              HIP_R_32F
HIP_R_8I                HIP_R_32F
HIP_R_32I               HIP_R_32F
======================= ===================
