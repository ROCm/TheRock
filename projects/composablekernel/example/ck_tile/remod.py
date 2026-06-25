# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import pathlib
from pathlib import Path
import subprocess

all_files = []
for p in sorted(Path("./").rglob("*")):
    if p.suffix in [".hpp", ".cpp"]:
        all_files.append(pathlib.PurePath(p))


# formatting
format_procs = []
for x in all_files:
    dos2unix = f"python -m dos2unix {str(x)} {str(x)}"
    clang_format = f"clang-format -style=file -i {str(x)}"
    # One process to avoid race conditions.
    cmd = f"{dos2unix} && {clang_format}"
    format_procs.append(
        subprocess.Popen(cmd, shell=True, stdout=open(os.devnull, "wb"))
    )

# Wait for formatting to complete.
for p in format_procs:
    p.wait()
