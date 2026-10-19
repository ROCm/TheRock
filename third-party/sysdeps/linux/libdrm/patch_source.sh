#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRM_MESON_BUILD="$SOURCE_DIR/meson.build"
AMDGPU_MESON_BUILD="$SOURCE_DIR/amdgpu/meson.build"
VERSION_LDS="$SCRIPT_DIR/version.lds"
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

# Apply symbol versioning to the library targets instead of to the whole project.
# Passing -Wl,--version-script via LDFLAGS applies it to every link in the
# project, and meson duplicates such arguments in its compiler sanity check
# (1.12.0), which ld rejects with "duplicate version tag". Per-target link_args
# are not duplicated and are the correct scope for a version script anyway.
sed -i "/^libdrm = library($/,/^)$/ s|^\([[:space:]]*\)install : true,|\1link_args : ['-Wl,--version-script=$VERSION_LDS'],\n\1install : true,|" "$DRM_MESON_BUILD"
sed -i "/^libdrm_amdgpu = library($/,/^)$/ s|^\([[:space:]]*\)install : true,|\1link_args : ['-Wl,--version-script=$VERSION_LDS'],\n\1install : true,|" "$AMDGPU_MESON_BUILD"
for meson_build in "$DRM_MESON_BUILD" "$AMDGPU_MESON_BUILD"; do
  if ! grep -q -- "--version-script" "$meson_build"; then
    echo "ERROR: Failed to patch $meson_build with --version-script" >&2
    exit 1
  fi
done

# Add/update device IDs in amdgpu.ids. Upstream libdrm releases periodically
# add their own entries for devices we've already listed here (sometimes with
# a stale or differently-cased name), so each entry is applied by device/
# revision key: any existing line for that key is deleted (a no-op if there
# isn't one), then the entry is appended. This keeps the table correct across
# libdrm version bumps without silently accumulating duplicate or
# conflicting entries for the same device/revision.
AMDGPU_IDS="$SOURCE_DIR/data/amdgpu.ids"
AMDGPU_ID_ENTRIES=(
  "7590,	CF,	AMD Radeon RX 9050"
  "7590,	DF,	AMD Radeon RX 9050 4GB"
  "75A8,	00,	AMD Instinct MI350P"
  "75B8,	00,	AMD Instinct MI350P VF"
  "7551,	C1,	AMD Radeon AI PRO R9700S"
  "7551,	C8,	AMD Radeon AI Pro R9600 Series"
)
for entry in "${AMDGPU_ID_ENTRIES[@]}"; do
  device_id="$(cut -d',' -f1 <<< "$entry" | xargs)"
  revision_id="$(cut -d',' -f2 <<< "$entry" | xargs)"
  sed -i -E "/^${device_id},[[:space:]]*${revision_id},/d" "$AMDGPU_IDS"
  printf '%s\n' "$entry" >> "$AMDGPU_IDS"
done
