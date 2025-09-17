#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
DRM_MESON_BUILD="$SOURCE_DIR/meson.build"
AMDGPU_MESON_BUILD="$SOURCE_DIR/amdgpu/meson.build"
echo "Patching sources..."

# Replace 'drm' in library() block with 'rocm_sysdeps_drm'
sed -i -E "/libdrm[[:space:]]*=[[:space:]]*library\(/,/\)/ s/^([[:space:]]*)'drm'[[:space:]]*,/\1'rocm_sysdeps_drm',/" "$DRM_MESON_BUILD"
# Remove libdrm from pkg.generate block, otherwise it will add '-lrocm_sysdeps_drm to the pkgconfig file
sed -i "/pkg\.generate\s*(/,/)/ s/\blibdrm,\s*//" "$DRM_MESON_BUILD"
# Add libraries tag to pkg.generate block
sed -i "/pkg\.generate\s*(/a\  libraries : ['-L\${libdir}', '-ldrm']," $DRM_MESON_BUILD
# Replace 'drm_amdgpu' in library() block with 'rocm_sysdeps_drm_amdgpu'
sed -i -E "/libdrm_amdgpu[[:space:]]*=[[:space:]]*library\(/,/\)/ s/^([[:space:]]*)'drm_amdgpu'[[:space:]]*,/\1'rocm_sysdeps_drm_amdgpu',/" "$AMDGPU_MESON_BUILD"
# Remove libdrm_amdgpu from pkg.generate block, otherwise it will add '-lrocm_sysdeps_drm_amdgpu to the pkgconfig file
sed -i "/pkg\.generate\s*(/,/)/ s/\blibdrm_amdgpu,\s*//" "$AMDGPU_MESON_BUILD"
# Add libraries tag to pkg.generate block
sed -i "/pkg\.generate\s*(/a\  libraries : ['-L\${libdir}', '-ldrm_amdgpu']," $AMDGPU_MESON_BUILD
