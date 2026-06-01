.. meta::
  :description: index page for the hipBLAS documentation and API reference library
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, documentation

.. _hipblas:

********************************************************************
hipBLAS documentation
********************************************************************

hipBLAS is a Basic Linear Algebra Subprograms (BLAS) marshaling library that
supports multiple backends. It lies between the application and a "worker" BLAS library,
marshalling inputs into the backend library and results back to the application.
hipBLAS exports an interface that does not require client changes, regardless of the
chosen backend. Currently, it supports :doc:`rocBLAS <rocblas:index>` and
NVIDIA CUDA `cuBLAS <https://developer.nvidia.com/cublas>`_ as backends.

The hipBLAS public repository is located at
`<https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas>`_.

.. note::

   The hipBLAS repository for ROCm 6.4.3 and earlier is located at `<https://github.com/ROCm/hipBLAS>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`hipBLAS prerequisites <./install/prerequisites>`
    * :doc:`Install hipBLAS on Linux <./install/Linux_Install_Guide>`
    * :doc:`Install hipBLAS on Windows <./install/Windows_Install_Guide>`

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :doc:`Library source code organization <./conceptual/library-source-code-organization>`

  .. grid-item-card:: How to

    * :doc:`Use hipBLAS clients <./how-to/using-hipblas-clients>`

  .. grid-item-card:: Examples

    * `hipBLAS client examples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas/clients/samples>`_

  .. grid-item-card:: API Reference

    * :doc:`Data type support <./reference/data-type-support>`
    * :doc:`hipBLAS API <./reference/hipblas-api-functions>`
    * :doc:`Deprecations by version <./reference/deprecation>`

To contribute to the documentation, see
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.
For information on contributing to the hipBLAS code base, see :doc:`Contribute to hipBLAS <./contribute>`.


You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
