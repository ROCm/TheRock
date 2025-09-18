#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

set -euo pipefail

DVC_VERSION="$1"

# https://dvc.org/doc/install/linux
curl --silent --fail --show-error --location \
    "https://dvc.org/download/linux-rpm/dvc-${DVC_VERSION}" \
    --output dvc.rpm

if [ "$EUID" -ne 0 ]; then
    sudo yum install dvc.rpm
else
    yum install dvc.rpm
fi
