.. meta::
   :description: Code sample demonstrating the use of the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, API, code sample

.. _client_extop_softmax:

***********************************
Softmax for a 2D tensor
***********************************

This code sample from ``clients/bench/src/client_extop_softmax.cpp`` demonstrates how to implement softmax for a 2D tensor. It performs softmax on the second dimension of input tensor and assumes the
input is contiguous on the second dimension.

.. literalinclude:: ../../clients/bench/src/client_extop_softmax.cpp
   :language: c++
