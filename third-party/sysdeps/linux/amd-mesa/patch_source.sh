#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VA_MESON_BUILD="$SOURCE_DIR/src/gallium/targets/va/meson.build"
VERSION_LDS="$SCRIPT_DIR/version.lds"

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

# Apply symbol versioning to the library targets instead of to the whole project.
# Passing -Wl,--version-script via LDFLAGS applies it to every link in the
# project, and meson duplicates such arguments in its compiler sanity check
# (1.12.0), which ld rejects with "duplicate version tag". Per-target link_args
# are not duplicated and are the correct scope for a version script anyway.
#
# Unlike the other meson sysdeps, two of these targets already carry a version
# script of their own, so the sysdeps script has to be merged with care:
#
#   * The gallium VA megadriver uses mesa's va.sym, which is an *anonymous*
#     version node. ld refuses to combine an anonymous node with the named
#     AMDROCM_SYSDEPS_1.0 node ("anonymous version tag cannot be combined with
#     other version tags"), so va.sym is replaced rather than appended to.
#     This is a no-op in practice: with version.lds on LDFLAGS, mesa's own
#     with_ld_version_script probe fails ("duplicate expression `*'", because
#     build-support/conftest.map and version.lds both match `*`), so va.sym is
#     already silently disabled today. Dropping it here keeps the linked result
#     identical while making the behaviour explicit instead of accidental.
#   * libva's va.syms uses named nodes and no `*` wildcard, so it coexists with
#     AMDROCM_SYSDEPS_1.0 and is appended to rather than replaced.
sed -i "\|^  va_link_args += \['-Wl,--version-script', join_paths(meson.current_build_dir(), 'va.sym')\]$|d" "$VA_MESON_BUILD"
sed -i "s|^va_link_args = \[\]$|va_link_args = ['-Wl,--version-script=$VERSION_LDS']|" "$VA_MESON_BUILD"
sed -i "s|^  link_args : libva_link_args,$|  link_args : [libva_link_args, '-Wl,--version-script=$VERSION_LDS'],|" "$LIBVA_MESON_BUILD"
sed -i "/^  libva_drm = shared_library($/,/^  libva_drm_dep = declare_dependency($/ s|^\([[:space:]]*\)install : true,|\1link_args : ['-Wl,--version-script=$VERSION_LDS'],\n\1install : true,|" "$LIBVA_MESON_BUILD"

# The VA megadriver must end up with exactly one version script, and both libva
# targets must have picked one up. A silent sed miss here would ship unversioned
# libraries, so fail loudly instead.
if [[ "$(grep -c -- "--version-script" "$VA_MESON_BUILD")" != "1" ]]; then
  echo "ERROR: Expected exactly one --version-script in $VA_MESON_BUILD" >&2
  exit 1
fi
if [[ "$(grep -c -- "--version-script=$VERSION_LDS" "$LIBVA_MESON_BUILD")" != "2" ]]; then
  echo "ERROR: Failed to patch both libva targets in $LIBVA_MESON_BUILD with --version-script" >&2
  exit 1
fi

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
