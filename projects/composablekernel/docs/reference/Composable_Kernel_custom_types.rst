.. meta::
  :description: Composable Kernel supported custom types
  :keywords: composable kernel, custom, data types, support, CK, ROCm

******************************************************
Composable Kernel custom data types
******************************************************

Composable Kernel supports the use of custom types that provide a way to implement specialized numerical formats.

To use custom types, a C++ type that implements the necessary operations for tensor computations needs to be created. These should include:

* Constructors and initialization methods
* Arithmetic operators if the type will be used in computational operations
* Any conversion functions needed to interface with other parts of an application

For example, to create a complex half-precision type:

.. code:: cpp

    struct complex_half_t
    {
        half_t real;
        half_t img;
    };

    struct complex_half_t
    {
        using type = half_t;
        type real;
        type img;

        complex_half_t() : real{type{}}, img{type{}} {}
        complex_half_t(type real_init, type img_init) : real{real_init}, img{img_init} {}
    };

Custom types can be particularly useful for specialized applications such as complex number arithmetic,
custom quantization schemes, or domain-specific number representations.

