#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
VA_MESON_BUILD="$SOURCE_DIR/src/gallium/targets/va/meson.build"

# Detect the libva subproject directory without hardcoding the version number.
LIBVA_SUBPROJECT_DIR="$(echo "$SOURCE_DIR"/subprojects/libva-[0-9]*/)"
if [[ ! -d "$LIBVA_SUBPROJECT_DIR" ]]; then
  echo "ERROR: Could not find libva subproject directory under $SOURCE_DIR/subprojects/" >&2
  exit 1
fi
# Strip trailing slash so paths below are clean.
LIBVA_SUBPROJECT_DIR="${LIBVA_SUBPROJECT_DIR%/}"

LIBVA_MESON_BUILD="$LIBVA_SUBPROJECT_DIR/va/meson.build"
LIBVA_MAIN_MESON_BUILD="$LIBVA_SUBPROJECT_DIR/meson.build"
LIBVA_PKGCONFIG_MESON_BUILD="$LIBVA_SUBPROJECT_DIR/pkgconfig/meson.build"
LIBVA_SOURCE="$LIBVA_SUBPROJECT_DIR/va/va.c"
echo "Patching sources..."

# Replace 'gallium_drv_video' in shared_library() calls with 'rocm_sysdeps_gallium_drv_video'
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
sed -i '/^[[:space:]]*char \*search_path = NULL;/a\    char *temp_path = NULL;' "$LIBVA_SOURCE"
sed -i "/^[[:space:]]*if[[:space:]]*(![[:space:]]*search_path)[[:space:]]*$/{
    N
    /^[[:space:]]*if[[:space:]]*(![[:space:]]*search_path)[[:space:]]*\n[[:space:]]*search_path[[:space:]]*=[[:space:]]*VA_DRIVERS_PATH;/{
        c\
    if (!search_path) {\
        char *rocm_path = secure_getenv(\"ROCM_PATH\");\
        if (rocm_path) {\
            if (asprintf(&temp_path, \"%s/lib/rocm_sysdeps/lib\", rocm_path) == -1) {\
                temp_path = NULL;\
            } else {\
                search_path = temp_path;\
            }\
        } else {\
            Dl_info dl_info;\
            if (dladdr((void *)va_openDriver, &dl_info) && dl_info.dli_fname) {\
                const char *last_slash = strrchr(dl_info.dli_fname, '/');\
                if (last_slash) {\
                    temp_path = strndup(dl_info.dli_fname, last_slash - dl_info.dli_fname);\
                    if (temp_path)\
                        search_path = temp_path;\
                }\
            }\
            if (!search_path)\
                search_path = VA_DRIVERS_PATH;\
        }\
    }
    }
}" "$LIBVA_SOURCE"
sed -i '/^[[:space:]]*search_path = strdup((const char \*)*search_path);$/a\    if (temp_path) { free(temp_path); temp_path = NULL; }' "$LIBVA_SOURCE"

# Modify pkgconfig generation to make driverdir relative to ${libdir} for relocatable packages
sed -i "/va_vars = vars + \['driverdir=' + driverdir\]/c\va_vars = vars + ['driverdir=\${libdir}']" "$LIBVA_PKGCONFIG_MESON_BUILD"
