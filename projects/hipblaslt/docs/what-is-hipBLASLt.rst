.. meta::
   :description: An introduction to the hipBLASLt library that provides GEMM operations with flexible APIs and extends functionalities beyond the traditional BLAS library
   :keywords: hipBLASLt, ROCm, library, API, tool

.. _what-is-hipblaslt:

What is hipBLASLt?
====================

hipBLASLt is a library that provides GEMM operations with flexible APIs and extends functionality beyond the traditional BLAS library.
hipBLASLt provides APIs in the :doc:`HIP programming language<hip:index>` with an underlying optimized generator as a backend kernel provider.

The library adds flexibility for matrix data layouts, input types, and compute types and
for choosing the algorithmic implementations and heuristics through parameter programmability.
After identifying a set of options for the intended GEMM operations, you can repeatedly use these options for different inputs.

The GEMM operation of hipBLASLt is performed by :ref:`hipblasltmatmul` using this equation:

.. math::

 D = Activation(alpha \cdot op(A) \cdot op(B) + beta \cdot op(C) + bias)

where :math:`op(A)/op(B)` refers to in-place operations such as transpose/non-transpose and :math:`alpha` and :math:`beta` are the scalars.
The :math:`Activation` function supports Gelu, Relu, Swish (SiLU), and Clamp. The :math:`Bias` vector matches matrix :math:`D` rows and broadcasts to all :math:`D` columns.
