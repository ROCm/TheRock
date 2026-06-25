[Back to the main page](../../README.md)
# Composable Kernel Tile
## concept
`ck_tile` provides a programming model with templated abstractions to enable users to implement performance-critical kernels for machine learning workloads. introduces following basic concepts to help users building your own operator
 - tensor coordinate transformation, this is the core concept of layout/index transform abstraction in both compiler time and run time.
 - tile-based programming model, including tile-level api and the concept of distributed tensor.

`ck_tile` is independently from the old ck, located under [/include/ck_tile](/include/ck_tile). You don't need to include anything from old CK, `ck_tile` has similiar (indeed almost the same) implementations for users to build operators. We will have a transition period to pull everything from old ck into `ck_tile`, stay tuned.

## component
`ck_tile` is splitted into several componenets including `core`, `host`, `ops/gemm`, `ops/fmha`... each component you only need to include a single header (e.g `#include "ck_tile/core.hpp"`, `#include "ck_tile/ops/fmha.hpp"`) then you are able to use the function/structure inside (different from old `ck`)  

**[core]**  
`ck_tile/core` contains all the basic data structure and function to build the kernel, you can only include this header and build your own operators that utilizing all the basic building blocks introduced in ck.

`core/container`
 - array, store runtime variables with fixed length (tensor index, register buffer, etc...)
 - tuple, same as std::tuple, hold different type of data, and one of the solution to achieve multiple buffer. 
 - sequence, compile time integer sequence used to build various internal structures, or to describe tile size
 - other convenient structure build on top of above 3

`core/numeric`
 - gpu data type like `fp16_t`, `bf16_t`, `fp8_t`... and the conversion between each other
 - constexpr integer similiar to std::integral_constant to be used as compile time integer.
 - math functions and numeric utilities

`core/algorithm`
 - coordinate transformation system, used to build tensor transform and compile time indexing. This is the core idea introduced in old `ck` to describe how a tensor is build by several basic transform primitives like `merge`/`unmerge`/`embed` etc... and how we indexing into a ND tensor that finally mapped to 1D memory offset.

`core/tensor`
 - tensor descriptor, to describe how a ND tensor 
 - distributed tensor, describe the storage of this tensor, and the distribution of how a collection of threads collaborately work for this tensor.
 - tile level API, including `load_tile`, `store_tile`, `shuffle_tile`, `slice_tile`, etc...

**[host]**  
`ck_tile/host` contains all the host side utilities to launch a kernel, create the device buffer, and some reference implementations. This can be used to create examples (like that under ck_tile example folder) and simple executable to invoke this kernel, so if you only need `ck_tile` to build your own device library then it's OK to not include this. Based on this, it is recommended to include the specific header you needed under this folder to avoid including unwanted headers (e.g, only include `ck_tile/host/kernel_launch.hpp`), unless you are writing a host executable.

**[ops/gemm, ops/fmha, ops/reduce...]**  
our implementation of different device operators. 
 - warp, warp tile level operator
 - block, block tile level operator
 - pipeline, pipeline that can achieve a customized tile level mainloop (or epilogue). By switching different pipeline to the kernel template you can have different kind of pipeline optimizations.
 - kernel, template interface for users to instantiate a particular kernel

**[ops/epilogue]**  
epilogue part of our kernel. We may extend this epilogue part to let users to build their own cutomized epilogues.

**[ref]**  
reference implementation of cpu or gpu. This folder is supposed to include a specific header on demand.

## examples
currently we put all ck_tile related example under [/example/ck_tile](/example/ck_tile/) folder. Please check each example's subfolder.
