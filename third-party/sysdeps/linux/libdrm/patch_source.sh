#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
DRM_MESON_BUILD="$SOURCE_DIR/meson.build"
AMDGPU_MESON_BUILD="$SOURCE_DIR/amdgpu/meson.build"
echo "Patching sources..."
sed -i "s/'drm',/'rocm_sysdeps_drm',/g" $DRM_MESON_BUILD
sed -i "s/'drm_amdgpu',/'rocm_sysdeps_drm_amdgpu',/g" $AMDGPU_MESON_BUILD
