.. meta::
  :description: hipBLAS documentation about how to use the hipBLAS clients
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, clients, test, testbench, documentation

.. _hipblas-clients:

********************************************************************
Using hipBLAS clients
********************************************************************

There are two client executables that can be used with hipBLAS. They are:

*  hipblas-bench
*  hipblas-test

These two clients can be built by following the instructions in :doc:`../install/Linux_Install_Guide`.
After building, the hipBLAS clients can be found in the ``hipBLAS/build/release/clients/staging`` directory.
See the next two sections for a brief explanation and usage notes for each hipBLAS client.

hipblas-bench
=============

hipblas-bench is used to measure performance and verify the correctness of hipBLAS functions.

It has a command line interface. For usage information, run this command:

.. code-block:: bash

   ./hipblas-bench --help

This example measures the performance of SGEMM:

.. code-block:: bash

   ./hipblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 4096 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 4096

On a system using an AMD Vega 20 GPU, the previous command displays a performance of 11941.5 Gflops,
as shown in the output below:

.. code-block:: bash

   transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,hipblas-Gflops,us
   N,N,4096,4096,4096,1,4096,4096,0,4096,11941.5,11509.4

A helpful way of finding the parameters that can be used with ``./hipblas-bench -f gemm`` is to turn on logging
by setting the environment variable ``ROCBLAS_LAYER=2``. For example, if you run this command:

.. code-block:: bash

   ROCBLAS_LAYER=2 ./hipblas-bench -f gemm -i 1 -j 0

It logs the following information:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 128 -n 128 -k 128 --alpha 1 --lda 128 --ldb 128 --beta 0 --ldc 128

You can copy and change the above command. For example, to change the
datatype to IEEE 64-bit and the size to 2048, follow this example:

.. code-block:: bash

   ./hipblas-bench -f gemm -r f64_r --transposeA N --transposeB N -m 2048 -n 2048 -k 2048 --alpha 1 --lda 2048 --ldb 2048 --beta 0 --ldc 2048

Logging affects performance, so only use it to log the command of interest,
then run the command without logging enabled to measure performance.

.. note::

   hipblas-bench also provides the flag ``-v 1`` for correctness checks.

If multiple arguments or functions need to be benchmarked,
hipblas-bench supports data-driven benchmarks using a YAML-format specification file.

.. code-block:: bash

   ./hipblas-bench --yaml <file>.yaml

For example, ``hipblas_smoke.yaml`` is a YAML file used to run a smoke test.
However, other examples can be found in the `rocBLAS <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas>`_ GitHub repository.

hipblas-test
============

hipblas-test is used to perform hipBLAS unit tests. It uses the GoogleTest framework.

To run the hipBLAS tests, use this command:

.. code-block:: bash

   ./hipblas-test

To run a subset of tests, provide an optional filter. For example,
to run only the ``axpy`` function tests from the command line, use:

.. code-block:: bash

   ./hibblas-test --gtest_filter=*axpy*

The pattern for ``--gtest_filter`` is:

.. code-block:: bash

   --gtest_filter=POSTIVE_PATTERNS[-NEGATIVE_PATTERNS]

If specific function arguments or multiple functions need to be tested,
hipblas-test provides support for data-driven testing using a YAML-format test specification file.

.. code-block:: bash

   ./hipblas-test --yaml <file>.yaml

As an example, ``hipblas_smoke.yaml`` is a YAML file that is used to run a smoke test.
Other examples can be found in the `rocBLAS <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas>`_ GitHub repository.
YAML-based tests list function parameter values in the test name, which can be also used for
test filtering using the ``gtest_filter`` argument.
To run the provided smoke test, use this command:

.. code-block:: bash

   ./hipblas-test --yaml hipblas_smoke.yaml
