#!/bin/bash
# Patches installed binaries from the external build system.
# Args: install_dir patchelf_binary
set -e

PREFIX="${1:?Expected install prefix argument}"

# Create symbolic links for libdrm.so and libdrm_amdgpu.so.
mv $PREFIX/lib/librocm_sysdeps_drm.so $PREFIX/lib/libdrm.so
mv $PREFIX/lib/librocm_sysdeps_drm_amdgpu.so $PREFIX/lib/libdrm_amdgpu.so
