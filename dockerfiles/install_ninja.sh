#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

NINJA_VERSION="$1"

ARCH="$(uname -m)"

curl --silent --fail --show-error --location \
    "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip" \
    --output ninja.zip

unzip ninja.zip
cp ninja /usr/local/bin
