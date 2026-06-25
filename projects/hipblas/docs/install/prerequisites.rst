.. meta::
  :description: hipBLAS installation prerequisites for Linux and Windows
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, install, prerequisites

.. _prerequisites:

***********************************************************
Prerequisites for hipBLAS installation
***********************************************************

The following prerequisites are required to install hipBLAS, whether by using a package manager
or building the application from the source code.

Prerequisites for Linux
=========================

The hipBLAS prerequisites are different than the :doc:`rocBLAS <rocblas:index>` and NVIDIA CUDA `cuBLAS <https://developer.nvidia.com/cublas>`_ backend prerequisites.

*  The prerequisites required to use the rocBLAS backend with AMD components are as follows:

   * A ROCm-enabled platform. For more information, see the :doc:`Linux system requirements <rocm-install-on-linux:reference/system-requirements>`.
   * A compatible version of `hipblas-common <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas-common>`_.
   * A compatible version of rocBLAS.
   * For full functionality, optionally install a compatible version of :doc:`rocSOLVER <rocsolver:index>` and its :doc:`rocSPARSE <rocsparse:index>`
     and :doc:`rocPRIM <rocprim:index>` dependencies.

*  The prerequisites required to use the cuBLAS backend with NVIDIA components are as follows:

   * A HIP-enabled platform. For more information, see :doc:`HIP installation <hip:install/install>`.
   * A working CUDA toolkit, including cuBLAS. For more information, see `CUDA toolkit <https://developer.nvidia.com/accelerated-computing-toolkit/>`_.

Prerequisites for Microsoft Windows
===================================

*  Here are the prerequisites required to use the rocBLAS backend with AMD components:

   * An AMD HIP SDK-enabled platform. For more information, see :doc:`Windows system requirements <rocm-install-on-windows:reference/system-requirements>`.
   * A compatible version of `hipblas-common <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas-common>`_.
   * A compatible version of rocBLAS.
   * For full functionality, a compatible version of :doc:`rocSOLVER <rocsolver:index>` and its :doc:`rocSPARSE <rocsparse:index>`
     and :doc:`rocPRIM <rocprim:index>` dependencies.
   * hipBLAS is supported on the same Windows versions and toolchains that HIP SDK supports.

* hipBLAS does not support the cuBLAS backend on Windows.
