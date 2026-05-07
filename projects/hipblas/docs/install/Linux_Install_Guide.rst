.. meta::
  :description: Installing and Building hipBLAS for Linux
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, documentation, Linux installation, build

.. _linux-install:

***********************************
Installing and building for Linux
***********************************

This topic discusses how to install hipBLAS on Linux from a package and how to build and install it from the source code.
For a list of installation prerequisites, see :doc:`hipBLAS prerequisites <prerequisites>`.

Installing prebuilt packages
=============================

You can manually download the prebuilt hipBLAS packages from the :doc:`ROCm native package manager <rocm-install-on-linux:install/quick-start>`.

To download the prebuilt package, use this command:

.. code-block:: shell

   sudo apt update && sudo apt install hipblas

Building hipBLAS from source
============================

When building hipBLAS from source, you can choose to build only the library and its dependencies or include the client and its
dependencies.

Download hipBLAS
----------------

The hipBLAS source code is available from the `hipBLAS folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas>`_
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.

To download hipBLAS, including all projects in the rocm-libraries repository, use the following commands.

.. code-block:: shell

   git clone -b release/rocm-rel-x.y https://github.com/ROCm/rocm-libraries.git
   cd  rocm-libraries/projects/hipblas


To limit your local checkout to only the hipBLAS project, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
Use the following commands for a sparse checkout:

.. note::

   To include the hipBLAS dependencies, set the projects for the sparse checkout using
   ``git sparse-checkout set projects/hipblas projects/rocsolver projects/rocblas projects/hipblas-common``.

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipblas # add projects/rocsolver projects/rocblas projects/hipblas-common to include dependencies
   git checkout develop # or use the branch you want to work with

.. note::

   To build ROCm 6.4.3 and older, use the hipBLAS repository at `<https://github.com/ROCm/hipBLAS>`_.
   For more information, see the documentation associated with the release you want to build.

The hipBLAS source code is found in the ``projects/hipblas`` directory.

Building the library and library dependencies
---------------------------------------------

The root directory of this repository contains the helper Python script ``rmake.py`` which lets you build and install
hipBLAS with a single command. It accepts several options but has a hard-coded configuration
that you can override by invoking ``cmake`` directly. However, it's a great way to get started quickly and
serves as an example of how to build and install hipBLAS.
A few commands in the script require ``sudo`` access, which might prompt you for a password.

.. note::

   You can run the ``rmake.py`` script from the ``projects/hipblas`` directory.

The install script determines the build platform by querying ``hipconfig --platform``. This value can be explicitly defined
by setting the environment variable ``HIP_PLATFORM`` to ``HIP_PLATFORM=amd`` or ``HIP_PLATFORM=nvidia``.

Common examples showing how to use ``rmake.py`` to build the library dependencies and library are listed
in this table.

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``python3 rmake.py -h``", "Help information."
   "``python3 rmake.py -d``", "Build the library dependencies and library in your local directory. The ``-d`` flag only has to be used once. For subsequent invocations of ``rmake.py``, it is not necessary to rebuild the dependencies."
   "``python3 rmake.py``", "Build the library in your local directory. It is assumed the dependencies have been built."
   "``python3 rmake.py -i``", "Build the library, then build and  install the hipBLAS package in  ``/opt/rocm/hipblas``. You will be prompted for sudo access. This installs it for all users. To restrict hipBLAS to your local directory, do not use the  ``-i`` flag. "
   "``python3 rmake.py -n``", "Build the library without the functionality provided by rocSOLVER. The rocSOLVER, rocSPARSE, and rocPRIM dependencies will not be required. This flag has no effect when building with a NVIDIA CUDA backend."

Building the library, client, and all dependencies
-------------------------------------------------------------------

The client contains the executables listed in the table below.

================= =======================================================
Executable name   Description
================= =======================================================
hipblas-test      Runs GoogleTest tests to validate the library
hipblas-bench     An executable to benchmark or test individual functions
hipblas-example-* Various examples showing how to use hipBLAS
================= =======================================================

Common ways to use ``rmake.py`` to build the dependencies, library, and client are
listed in this table.

.. csv-table::
   :header: "Command","Description"
   :widths: 33, 97

   "``python3 rmake.py -dc``", "Build the library dependencies, client dependencies, library, and client in your local directory. The ``-d`` flag only has to be used once. For subsequent invocations of  ``rmake.py``, it is not necessary to rebuild the dependencies."
   "``python3 rmake.py -c``", "Build the library and client in your local directory. It is assumed the  dependencies have been built."
   "``python3 rmake.py -idc``", "Build the library dependencies, client dependencies, library, and client, then build and install the hipBLAS package. You will be prompted for sudo access. To install hipBLAS for all users, use the ``-i`` flag. To restrict hipBLAS to your local directory, do not use the ``-i`` flag."
   "``python3 rmake.py -ic``", "Build and install the hipBLAS package and build the client. You will be prompted for sudo access. This installs it for all users. To restrict hipBLAS to your local directory, do not use the ``-i`` flag."

Dependencies for building the library
=====================================

Use ``rmake.py`` with the ``-d`` option to install the dependencies required to build the library.
This does not install the hipblas-common, rocBLAS, rocSOLVER, rocSPARSE, and rocPRIM dependencies.
When building hipBLAS, it is important to take note of the version dependencies for other libraries. The hipblas-common,
rocBLAS, and rocSOLVER versions required to build for an AMD backend are listed in the top-level ``CMakeLists.txt`` file.
rocSPARSE and rocPRIM are currently dependencies for rocSOLVER. To build these libraries from
source, see the :doc:`rocBLAS <rocblas:index>`,
:doc:`rocSOLVER <rocsolver:index>`, :doc:`rocSPARSE <rocsparse:index>`,
and :doc:`rocPRIM <rocprim:index>` documentation.

The minimum version of CMake is currently 3.16.8. See the ``--cmake_install`` flag in ``rmake.py`` to
upgrade automatically.

To use the test and benchmark clients' host reference functions, you must manually download and install
AMD's `ILP64 version of the AOCL libraries <https://www.amd.com/en/developer/aocl.html>`_ version 4.2.
The ``aocl-linux-*`` packages include AOCL-BLAS (``aocl-blis``) and AOCL-LAPACK (``aocl-libflame``).
If you download and install the full AOCL packages to the default location, then these reference
functions should be found by the clients` ``CMakeLists.txt`` file.

.. note::

   If you only use the ``rmake.py -d`` dependency script and change the default CMake option ``LINK_BLIS=ON``,
   you might experience ``hipblas-test`` stress test failures due to a 32-bit integer overflow
   on the host. To resolve this issue, exclude the stress tests using the command line argument ``--gtest_filter=-*stress*``.

Manual build
=======================================

This section provides information on how to configure CMake and manually build on all supported platforms.

Build the library using individual commands
-------------------------------------------

.. code-block:: bash

   mkdir -p [HIPBLAS_BUILD_DIR]/release
   cd [HIPBLAS_BUILD_DIR]/release
   # Default install location is in /opt/rocm, define -DCMAKE_INSTALL_PREFIX=<path> to specify other
   # Default build config is 'Release', define -DCMAKE_BUILD_TYPE=<config> to specify other
   CXX=/opt/rocm/bin/amdclang++ ccmake [HIPBLAS_SOURCE]
   make -j$(nproc)
   sudo make install # sudo required if installing into system directory such as /opt/rocm

Build the library, tests, benchmarks, and samples using individual commands
----------------------------------------------------------------------------

The repository contains source code for clients that serve as samples, tests, and benchmarks. These source code files can be
found in the `clients subdirectory <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas/clients>`_ of the hipBLAS GitHub.

Dependencies (only necessary for hipBLAS clients)
-------------------------------------------------

The hipBLAS samples have no external dependencies, but the unit test and benchmarking applications do.
These clients have the following dependencies:

* `LAPACK <https://github.com/Reference-LAPACK>`_: LAPACK itself adds a dependency on a Fortran compiler
* `GoogleTest <https://github.com/google/googletest>`_

GoogleTest and LAPACK are more difficult to install. Many distributions
do not provide a GoogleTest package with pre-compiled libraries,
and the LAPACK packages do not have the necessary CMake config files for CMake to link to the library.
hipBLAS provides a CMake script that builds these dependencies from source.
This is an optional step. Users can provide their own builds of these dependencies and configure CMake to find them
by setting the ``CMAKE_PREFIX_PATH`` definition. The following steps demonstrate how to build dependencies and
install them to the default CMake ``/usr/local`` directory.

.. note::

   The following steps are optional and only need to be run once.

.. code-block:: bash

   mkdir -p [HIPBLAS_BUILD_DIR]/release/deps
   cd [HIPBLAS_BUILD_DIR]/release/deps
   ccmake -DBUILD_BOOST=OFF [HIPBLAS_SOURCE]/deps   # assuming boost is installed through package manager as above
   make -j$(nproc) install

After the dependencies are available on the system, configure the clients to build.
This involves passing a few extra CMake flags to the library CMake configure script. If the dependencies are not
installed into the default system locations, such as ``/usr/local``, pass the ``CMAKE_PREFIX_PATH`` to CMake so it can find them.

.. code-block:: bash

   -DCMAKE_PREFIX_PATH="<semicolon separated paths>"
   # Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
   CXX=/opt/rocm/bin/amdclang++ ccmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON [HIPBLAS_SOURCE]
   make -j$(nproc)
   sudo make install   # sudo required if installing into system directory such as /opt/rocm
