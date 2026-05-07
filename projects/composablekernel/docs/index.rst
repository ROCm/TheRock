.. meta::
  :description: Composable Kernel documentation and API reference library
  :keywords: composable kernel, CK, ROCm, API, documentation

.. _composable-kernel:

********************************************************************
Composable Kernel User Guide
********************************************************************

The Composable Kernel library provides a programming model for writing performance critical kernels for machine learning workloads across multiple architectures including GPUs and CPUs, through general purpose kernel languages such as `HIP C++ <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_.

The Composable Kernel repository is located at `https://github.com/ROCm/composable_kernel <https://github.com/ROCm/composable_kernel>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Composable Kernel prerequisites <./install/Composable-Kernel-prerequisites>`
    * :doc:`Build and install Composable Kernel <./install/Composable-Kernel-install>`
    * :doc:`Build and install Composable Kernel on a Docker image <./install/Composable-Kernel-Docker>`

  .. grid-item-card:: Conceptual

    * :doc:`Composable Kernel structure <./conceptual/Composable-Kernel-structure>`
    * :doc:`Composable Kernel mathematical basis <./conceptual/Composable-Kernel-math>`
    * :doc:`CK Tile conceptual documentation <./conceptual/ck_tile/index>`

  .. grid-item-card:: Tutorials

    * :doc:`Composable Kernel examples and tests <./tutorial/Composable-Kernel-examples>`

  .. grid-item-card:: Reference

    * :doc:`Composable Kernel supported scalar types <./reference/Composable_Kernel_supported_scalar_types>`
    * :doc:`Composable Kernel custom types <./reference/Composable_Kernel_custom_types>`
    * :doc:`Composable Kernel vector utilities <./reference/Composable_Kernel_vector_utilities>`
    * :ref:`wrapper`    
    * :doc:`Composable Kernel API reference <./doxygen/html/namespace_c_k>`
    * :doc:`CK Tile API reference <./doxygen/html/namespaceck__tile>`
    * :doc:`Composable Kernel complete API class list <./doxygen/html/annotated>`
    * :doc:`Composable Kernel glossary <./reference/Composable-Kernel-Glossary>`
    
To contribute to the documentation refer to `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
