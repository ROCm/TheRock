.. meta::
  :description: Composable Kernel build and install
  :keywords: composable kernel, CK, ROCm, API, documentation, install

******************************************************
Building and installing Composable Kernel with CMake
******************************************************

Before you begin, clone the `Composable Kernel GitHub repository <https://github.com/ROCm/composable_kernel.git>`_ and create a ``build`` directory in its root:

.. code:: shell

  git clone https://github.com/ROCm/composable_kernel.git
  cd composable_kernel
  mkdir build

Change directory to the ``build`` directory and generate the makefile using the ``cmake`` command. Two build options are required:

* ``CMAKE_PREFIX_PATH``: The ROCm installation path. ROCm is installed in ``/opt/rocm`` by default.
* ``CMAKE_CXX_COMPILER``: The path to the Clang compiler. Clang is found at ``/opt/rocm/llvm/bin/clang++`` by default.


.. code:: shell

  cd build
  cmake ../. -D CMAKE_PREFIX_PATH="/opt/rocm" -D CMAKE_CXX_COMPILER="/opt/rocm/llvm/bin/clang++" [-D<OPTION1=VALUE1> [-D<OPTION2=VALUE2>] ...]


Other build options are:

* ``DISABLE_DL_KERNELS``: Set this to "ON" to not build deep learning (DL) and data parallel primitive (DPP) instances. 

  .. note::

      DL and DPP instances are useful on architectures that don't support XDL or WMMA.

* ``CK_USE_FP8_ON_UNSUPPORTED_ARCH``: Set to ``ON`` to build FP8 data type instances on gfx90a without native FP8 support.
* ``GPU_TARGETS``: Target architectures. Target architectures in this list must all be different versions of the same architectures. Enclose the list of targets in quotation marks. Separate multiple targets with semicolons (``;``). For example, ``cmake -D GPU_TARGETS="gfx908;gfx90a"``. This option is required to build tests and examples.
* ``GPU_ARCHS``: Target architectures. Target architectures in this list are not limited to different versions of the same architectures. Enclose the list of targets in quotation marks. Separate multiple targets with semicolons (``;``). For example, ``cmake -D GPU_TARGETS="gfx908;gfx1100"``.
* ``CMAKE_BUILD_TYPE``: The build type. Can be ``None``, ``Release``, ``Debug``, ``RelWithDebInfo``, or ``MinSizeRel``. CMake will use ``Release`` by default.

.. Note::

  If neither ``GPU_TARGETS`` nor ``GPU_ARCHS`` is specified, Composable Kernel will be built for all targets supported by the compiler.

Build Composable Kernel using the generated makefile. This will build the library, the examples, and the tests, and save them to ``bin``.

.. code:: shell

    make -j20

The ``-j`` option speeds up the build by using multiple threads in parallel. For example, ``-j20`` uses twenty threads in parallel. On average, each thread will use 2GB of memory. Make sure that the number of threads you use doesn't exceed the available memory in your system.

Using ``-j`` alone will launch an unlimited number of threads and is not recommended.

Install the Composable Kernel library:

.. code:: shell
  
  make install

After running ``make install``, the Composable Kernel files will be saved to the following locations:

* Library files: ``/opt/rocm/lib/``
* Header files: ``/opt/rocm/include/ck/`` and ``/opt/rocm/include/ck_tile/``
* Examples, tests, and ckProfiler: ``/opt/rocm/bin/``

For information about ckProfiler, see `the ckProfiler readme file <https://github.com/ROCm/composable_kernel/blob/develop/profiler/README.md>`_.

For information about running the examples and tests, see :doc:`Composable Kernel examples and tests <../tutorial/Composable-Kernel-examples>`.


