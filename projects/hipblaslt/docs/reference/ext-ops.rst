.. meta::
   :description: hipBLASLtExt operation API reference
   :keywords: hipBLASLt, ROCm, library, tool

.. _ext-ops:

hipBLASLtExt operation API reference
======================================

hipBLASLt has the following extension operation APIs that are independent of GEMM operations.
These extensions support the following:

*  ``hipblasltExtSoftmax``

   Softmax for 2D tensor. It performs softmax on the second dimension of input tensor and assumes the
   input is contiguous on the second dimension.
   For sample code, see :ref:`client_extop_softmax`.

*  ``hipblasltExtLayerNorm``

   Converts a 2D tensor using LayerNorm to generate a new 2D normalized tensor.
   This is an independent function used to call and get the result.
   For sample code, see :ref:`sample_hipblaslt_ext_op_layernorm`.

*  ``hipblasltExtAMax``

   Determines the absolute maximum value of a 2D tensor.
   This is an independent function used to call and get the result.
   For sample code, see :ref:`sample_hipblaslt_ext_op_amax`.

These APIs are explained in detail below.

hipblasltExtSoftmax()
------------------------------------------
.. doxygenfunction:: hipblasltExtSoftmax


hipblasltExtLayerNorm()
------------------------------------------
.. doxygenfunction:: hipblasltExtLayerNorm


hipblasltExtAMax()
------------------------------------------
.. doxygenfunction:: hipblasltExtAMax
