## common
this folder is designed not to be included directly by use, e.g. if use include `ck_tile/ops/fmha.hpp`, then everything under `common` should also be included.

to achieve this we will duplicate the header include path under `common` to other module under `ops/*` inside remod.py. for internal developer, you can also include `ck_tile/ops/common.hpp` for convenience. (and so does external users...)
