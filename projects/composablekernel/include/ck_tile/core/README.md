# ck_tile/core #

`ck_tile/core` contains every basic functions and structures to create a GPU kernel using `ck_tile`. User should only include `ck_tile/core.hpp` this single header to use all the functionality. Everything is under `ck_tile` namespace. The coding style under this folder should be similar to `std` (`snake_case` for structure/function, Camel for template types...)

```
algorithm/
    coordinate transform and some other reusable algorithm
arch/
    contains some basic device building block like mma, buffer addressing, etc...
container/
    contains basic container data structure, array/sequence/tuple/...
numeric/
    data type, and data type related math
tensor/
    tensor descriptors and tile level API
utility/
    other utility function for both host/device
```
