# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(rocjitsu_embed_package LANGUAGES NONE)

install(FILES "${ROCJITSU_DIST_DIR}/lib/librocjitsu_embed.so" DESTINATION lib)
install(PROGRAMS "${ROCJITSU_DIST_DIR}/libexec/rocjitsu-embed/rocjitsu-memory"
  DESTINATION libexec/rocjitsu-embed
)
