# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pathlib
from pathlib import Path
import subprocess
import os
import copy

NS = "ck_tile"
OPS = "ops"
OPS_COMMON = "common"  # common header will be duplicated into ops/* other module

IGNORED_DIRS = ["utility", "ref"]
HEADER_COMMON = """// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
"""


# aa/bb/cc/file.hpp -> (aa, bb, cc, file.hpp)
def get_module(f, level=0):
    all_parts = f.parts
    return str(all_parts[level])


all_files = []
for p in sorted(Path("./").rglob("*")):
    if p.suffix == ".hpp":
        all_files.append(pathlib.PurePath(p))


class submodule_t:
    def __init__(self):
        self.m = dict()

    def push(self, f):
        if len(f.parents) != 1:  # ignore ./xxx.hpp
            mod = get_module(f)
            # Should only be included by demand
            if mod in IGNORED_DIRS:
                return
            if mod == OPS:
                if mod not in self.m.keys():
                    self.m[mod] = dict()
                mod2 = get_module(f, 1)
                if Path(mod2).suffix != ".hpp":
                    # ignore ops/xxx.hpp
                    if mod2 not in self.m[mod].keys():
                        self.m[mod][mod2] = list()
                    self.m[mod][mod2].append(f)
            else:
                if mod not in self.m.keys():
                    self.m[mod] = list()
                self.m[mod].append(f)

    def gen(self):
        def gen_header(hpath, include_list):
            # print(hpath)
            if os.path.exists(str(hpath)):
                os.remove(str(hpath))
            with hpath.open("w") as f:
                f.write(HEADER_COMMON)
                f.write("#pragma once\n")
                f.write("\n")
                for individual_header in include_list:
                    header_path = NS + "/" + str(individual_header)
                    f.write(f'#include "{header_path}"\n')
                # f.write('\n') # otherwise clang-format will complain

        # print(self.m)
        # restructure common
        for k, v in self.m.items():
            if k == OPS and OPS_COMMON in v.keys():
                common_list = copy.deepcopy(v[OPS_COMMON])
                # v.pop(OPS_COMMON)
                for km in v.keys():
                    if km != OPS_COMMON:
                        v[km].extend(common_list)

        for k, v in self.m.items():
            if k == OPS:
                for km, kv in v.items():
                    gen_header(Path(k) / (f"{km}.hpp"), kv)
            else:
                gen_header(Path(f"{k}.hpp"), v)


submodule = submodule_t()
# formatting
format_procs = []
for x in all_files:
    dos2unix = f"python3 -m dos2unix {str(x)} {str(x)}"
    clang_format = f"clang-format -style=file -i {str(x)}"
    # One process to avoid race conditions.
    cmd = f"{dos2unix} && {clang_format}"
    format_procs.append(
        subprocess.Popen(cmd, shell=True, stdout=open(os.devnull, "wb"))
    )
    submodule.push(x)

# Wait for formatting to complete before generating headers.
for p in format_procs:
    p.wait()

submodule.gen()
