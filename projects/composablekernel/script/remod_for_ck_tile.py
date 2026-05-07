# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os

root_dir = os.getcwd()
ck_tile_include = root_dir + "/include/ck_tile"
ck_tile_example = root_dir + "/example/ck_tile"

# Run for include
os.chdir(ck_tile_include)
_ = os.system("python remod.py")

# Run for example
os.chdir(ck_tile_example)
_ = os.system("python remod.py")
