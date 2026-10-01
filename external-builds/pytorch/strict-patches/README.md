# Native MI45x integration overlay

CI release/2.12 is pinned to b97872d55cf78ed4a8dd9f71ce7767e35b39f2fc;
release/2.13 is pinned to 8e0af7fa4388411493540afa152777537bbbfc1c.
ROCm/pytorch #3667 merged as 991b967f6c9966d1a93ce2515613d55d4d171e85,
but its exact patch is absent and applies cleanly to both pinned release sources.

The additional CK-only empty-target guard is a local backport/adaptation of
#3639's missing Dependencies.cmake intent. It disables CK GEMM when its filtered
architecture list is empty, preserving the overall PyTorch target list. The
whole overlapping #3639 patch is not applied (it conflicts with these release
sources and duplicates #3667). Source patches are applied before HIPIFY.

These overlays do not replace or qualify embedded AOTriton/Triton payloads;
framework compilation and physical native-target qualification remain separate.
