#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
VA_MESON_BUILD="$SOURCE_DIR/src/gallium/targets/va/meson.build"
LIBVA_MESON_BUILD="$SOURCE_DIR/subprojects/libva-2.22.0/va/meson.build"
LIBVA_MAIN_MESON_BUILD="$SOURCE_DIR/subprojects/libva-2.22.0/meson.build"
LIBVA_PKGCONFIG_MESON_BUILD="$SOURCE_DIR/subprojects/libva-2.22.0/pkgconfig/meson.build"
LIBVA_SOURCE="$SOURCE_DIR/subprojects/libva-2.22.0/va/va.c"
echo "Patching sources..."

# Replace 'gallium_drv_video' in shared_library() calls with 'rocm_sysdeps_gallium_drv_video'
# This handles both the with_amd_decode_only branch and the standard branch
sed -i -E "/shared_library\(/,/\)/ s/'gallium_drv_video'/'rocm_sysdeps_gallium_drv_video'/" "$VA_MESON_BUILD"

# Replace 'va' library name with 'rocm_sysdeps_va' in libva meson.build
sed -i -E "/shared_library\(/,/\)/ s/'va',/'rocm_sysdeps_va',/" "$LIBVA_MESON_BUILD"

# Replace 'va-drm' library name with 'rocm_sysdeps_va-drm' in libva meson.build
sed -i -E "/shared_library\(/,/\)/ s/'va-drm',/'rocm_sysdeps_va-drm',/" "$LIBVA_MESON_BUILD"

# Remove libva from pkg.generate block and add explicit name/libraries to override automatic detection
sed -i "/pkg\.generate(libva,/,/version:/ s/pkg\.generate(libva,/pkg.generate(/" "$LIBVA_PKGCONFIG_MESON_BUILD"
sed -i "/description: 'Userspace Video Acceleration (VA) core interface'/a\  name : 'va'," "$LIBVA_PKGCONFIG_MESON_BUILD"
sed -i "/description: 'Userspace Video Acceleration (VA) core interface'/a\  libraries : ['-L\${libdir}', '-lva']," "$LIBVA_PKGCONFIG_MESON_BUILD"

# Remove libva_drm from pkg.generate block and add explicit name/libraries to override automatic detection
sed -i "/pkg\.generate(libva_drm,/,/version:/ s/pkg\.generate(libva_drm,/pkg.generate(/" "$LIBVA_PKGCONFIG_MESON_BUILD"
sed -i "/description: 'Userspace Video Acceleration (VA) DRM interface'/a\  name : 'va-drm'," "$LIBVA_PKGCONFIG_MESON_BUILD"
sed -i "/description: 'Userspace Video Acceleration (VA) DRM interface'/a\  libraries : ['-L\${libdir}', '-lva-drm']," "$LIBVA_PKGCONFIG_MESON_BUILD"

# Modify libva meson.build to set driverdir to libdir
sed -i "/driverdir = join_paths(get_option('prefix'), get_option('libdir'), 'dri')/c\    driverdir = join_paths(get_option('prefix'), get_option('libdir'))" "$LIBVA_MAIN_MESON_BUILD"

# This eliminates the need for LIBVA_DRIVERS_PATH environment variable
sed -i "/^[[:space:]]*if[[:space:]]*(![[:space:]]*search_path)[[:space:]]*$/{
    N
    /^[[:space:]]*if[[:space:]]*(![[:space:]]*search_path)[[:space:]]*\n[[:space:]]*search_path[[:space:]]*=[[:space:]]*VA_DRIVERS_PATH;/{
        c\
    if (!search_path) {\
        char *rocm_path = secure_getenv(\"ROCM_PATH\");\
        if (rocm_path) {\
            if (asprintf(&search_path, \"%s/lib/rocm_sysdeps/lib\", rocm_path) == -1) {\
                search_path = NULL;\
            }\
        } else {\
            search_path = VA_DRIVERS_PATH;\
        }\
    }
    }
}" "$LIBVA_SOURCE"

# Modify pkgconfig generation to make driverdir relative to ${libdir} for relocatable packages
sed -i "/va_vars = vars + \['driverdir=' + driverdir\]/c\va_vars = vars + ['driverdir=\${libdir}']" "$LIBVA_PKGCONFIG_MESON_BUILD"
