#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|hpp|txt|inc)$'); do
    if [ -x "$file" ]; then
        chmod -x "$file"
        echo "[remove-exec-bit] Removed executable bit from $file" >&2
    fi
done
