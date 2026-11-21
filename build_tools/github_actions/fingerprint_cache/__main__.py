#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enable ``python -m fingerprint_cache``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
