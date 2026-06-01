.. meta::
  :description: Composable Kernel supported precision types and custom type support
  :keywords: composable kernel, precision, data types, ROCm

******************************************************
Composable Kernel vector template utilities
******************************************************

Composable Kernel includes template utilities for creating vector types with customizable widths. These template utilities also flatten nested vector types into a single, wider vector, preventing the creation of vectors of vectors.

Vectors composed of supported scalar and custom types can be created with the ``ck::vector_type`` template.

For example, ``ck::vector_type<float, 4>`` creates a vector composed of four floats and ``ck::vector_type<ck::half_t, 8>`` creates a vector composed of eight half-precision scalars.

For vector operations to be valid, the underlying types must be either a :doc:`supported scalar type <Composable_Kernel_supported_scalar_types>` or :doc:`a custom type <Composable_Kernel_custom_types>` that implements the required operations.

