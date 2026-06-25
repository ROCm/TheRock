.. meta::
   :description: Source code organization and structure for the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, API, source code, structure, organization

.. _source-code-organization:

******************************************
hipBLASLt library source code organization
******************************************

The hipBLASLt source code resides in the following two directories:

*  The ``library`` directory contains all source code for the library.
*  The ``clients`` directory contains all test code and the code to build the clients.

The library directory
-----------------------

Here are the subdirectories within the ``library`` directory:

*  ``library/include``

   Contains C98 include files for the external API. These files also contain Doxygen
   comments that document the API.

*  ``library/src/amd_detail``

   Contains the implementation of the hipBLASLt interface that is compatible with the rocBLASLt APIs.

*  ``library/src/include``

   Contains internal include files for converting C++ exceptions to hipBLAS statuses.

The clients directory
-----------------------

Here is the subdirectory within the ``clients`` directory:

*  ``clients/samples``

   Contains sample code for calling the hipBLASLt functions.

Infrastructure
--------------

*  ``CMake`` is used to build and package hipBLASLt. There are ``CMakeLists.txt`` files throughout the code.
*  Doxygen, Breathe, Sphinx, and ReadTheDocs generate the documentation
   based on these sources:

   *  Doxygen comments in the ``include`` files in the ``library/include`` directory.
   *  Files in the ``docs/source`` directory.

*  Jenkins is used to automate continuous integration testing.
*  ``clang-format`` is used to format the C++ code.
