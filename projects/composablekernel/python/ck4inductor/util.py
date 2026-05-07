# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import functools
import os


@functools.lru_cache(None)
def library_path():
    return os.path.join(os.path.dirname(__file__), "library")
