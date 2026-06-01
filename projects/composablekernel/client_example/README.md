[Back to the main page](../README.md)
# Composable Kernel client examples
##
Client application links to CK library, and therefore CK library needs to be installed before building client applications.


## Build
```bash
mkdir -p client_example/build
cd client_example/build
```

```bash
cmake                                                                 \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
-D CMAKE_PREFIX_PATH="/opt/rocm;${PATH_TO_CK_INSTALL_DIRECTORY}"      \
-D GPU_TARGETS="gfx908;gfx90a"                                        \
..
```
You must set the `GPU_TARGETS` macro to specify the GPU target architecture(s).

### Build client example
```bash
 make -j 
```
