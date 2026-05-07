.. meta::
  :description: Composable Kernel supported scalar types
  :keywords: composable kernel, scalar, data types, support, CK, ROCm

***************************************************
Composable Kernel supported scalar data types
***************************************************

The Composable Kernel library provides support for the following scalar data types:

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Type
      - Bit Width
      - Description

    * - ``double``
      - 64-bit
      - Standard IEEE 754 double precision floating point

    * - ``float``
      - 32-bit
      - Standard IEEE 754 single precision floating point

    * - ``int32_t``
      - 32-bit
      - Standard signed 32-bit integer

    * - ``int8_t``
      - 8-bit
      - Standard signed 8-bit integer

    * - ``uint8_t``
      - 8-bit
      - Standard unsigned 8-bit integer

    * - ``bool``
      - 1-bit
      - Boolean type

    * - ``ck::half_t``
      - 16-bit
      - IEEE 754 half precision floating point with 5 exponent bits, 10 mantissa bits, and 1 sign bit

    * - ``ck::bhalf_t``
      - 16-bit
      - Brain floating point with 8 exponent bits, 7 mantissa bits, and 1 sign bit

    * - ``ck::f8_t``
      - 8-bit
      - 8-bit floating point (E4M3 format) with 4 exponent bits, 3 mantissa bits, and 1 sign bit

    * - ``ck::bf8_t``
      - 8-bit
      - 8-bit brain floating point (E5M2 format) with 5 exponent bits, 2 mantissa bits, and 1 sign bit

    * - ``ck::f4_t``
      - 4-bit
      - 4-bit floating point format (E2M1 format) with 2 exponent bits, 1 mantissa bit, and 1 sign bit

    * - ``ck::f6_t``
      - 6-bit
      - 6-bit floating point format (E2M3 format) with 2 exponent bits, 3 mantissa bits, and 1 sign bit

    * - ``ck::bf6_t``
      - 6-bit
      - 6-bit brain floating point format (E3M2 format) with 3 exponent bits, 2 mantissa bits, and 1 sign bit