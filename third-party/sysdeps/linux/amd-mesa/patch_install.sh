#!/bin/bash
set -e

PREFIX="${1:?Expected install prefix argument}"

# Handle gallium video driver
GALLIUM_LIB="$PREFIX/lib/librocm_sysdeps_gallium_drv_video.so"
if [ -e "$GALLIUM_LIB" ]; then
    # For plain .so files without versioning, just create a symlink directly
    pushd "$PREFIX/lib" > /dev/null
    ln -sf "librocm_sysdeps_gallium_drv_video.so" "libgallium_drv_video.so"
    popd > /dev/null
else
    echo "Warning: Could not find librocm_sysdeps_gallium_drv_video.so"
fi

# Find the actual libva symlinks that meson created (e.g., librocm_sysdeps_va.so.2)
VA_LIB=$(find "$PREFIX/lib" -name "librocm_sysdeps_va.so.2*" -type l | head -1)
if [ -n "$VA_LIB" ]; then
    # Keep the meson-created symlinks, just add the additional symlink we need
    pushd "$PREFIX/lib" > /dev/null
    ln -sf "$(basename "$VA_LIB")" "libva.so"
    popd > /dev/null
else
    echo "Warning: Could not find librocm_sysdeps_va.so.2 symlink"
fi

VA_DRM_LIB=$(find "$PREFIX/lib" -name "librocm_sysdeps_va-drm.so.2*" -type l | head -1)
if [ -n "$VA_DRM_LIB" ]; then
    # Keep the meson-created symlinks, just add the additional symlink we need
    pushd "$PREFIX/lib" > /dev/null
    ln -sf "$(basename "$VA_DRM_LIB")" "libva-drm.so"
    popd > /dev/null
else
    echo "Warning: Could not find librocm_sysdeps_va-drm.so.2 symlink"
fi
