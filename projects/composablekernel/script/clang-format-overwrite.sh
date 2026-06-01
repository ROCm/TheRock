# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find . -name deps -prune -o -name build -prune -o -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' -o -iname '*.h.in' -o -iname '*.hpp.in' -o -iname '*.cpp.in' -o -iname '*.cl' -o -iname '*.cuh' -o -iname '*.cu' -o -iname '*.inc' | grep -v 'build/' | grep -v 'include/rapidjson'| xargs -n 1 -P 16 -I{} -t sh -c 'clang-format-18 -i -style=file {}'
git status --porcelain | awk '$1 != "D" && (match($2, "\\.cpp|.hpp|.inc|include/rapidjson/")) {print $2}' | xargs -n 1 -P 16 -I{} -t sh -c 'clang-format-18 -i -style=file {}'
