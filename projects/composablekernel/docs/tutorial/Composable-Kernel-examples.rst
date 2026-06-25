.. meta::
  :description: Composable Kernel examples and tests
  :keywords: composable kernel, CK, ROCm, API, examples, tests

********************************************************************
Composable Kernel examples and tests
********************************************************************

After :doc:`building and installing Composable Kernel <../install/Composable-Kernel-install>`, the examples and tests will be moved to ``/opt/rocm/bin/``.

All tests have the prefix ``test`` and all examples have the prefix ``example``.

Use ``ctest`` with no arguments to run all examples and tests, or use ``ctest -R`` to run a single test. For example:

.. code:: shell

    ctest -R test_gemm_fp16

Examples can be run individually as well. For example:

.. code:: shell

    ./bin/example_gemm_xdl_fp16 1 1 1

For instructions on how to run individual examples and tests, see their README files in the |example|_ and |test|_ GitHub folders.

To run smoke tests, use ``make smoke``.

To run regression tests, use ``make regression``.

In general, tests that run for under thirty seconds are included in the smoke tests and tests that run for over thirty seconds are included in the regression tests. 

.. |example| replace:: ``example``
.. _example: https://github.com/ROCm/composable_kernel/tree/develop/example

.. |client_example| replace:: ``client_example``
.. _client_example: https://github.com/ROCm/composable_kernel/tree/develop/client_example

.. |test| replace:: ``test``
.. _test: https://github.com/ROCm/composable_kernel/tree/develop/test