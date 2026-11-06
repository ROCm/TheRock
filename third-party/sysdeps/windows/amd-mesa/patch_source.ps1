# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory=$true)]
  [string]$SourceDir
)

$ErrorActionPreference = "Stop"

# Detect the libva subproject directory without hardcoding the version number.
$LibvaSubprojectDir = Get-ChildItem -Path "$SourceDir/subprojects" -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '^libva-[0-9]' } | Select-Object -First 1
if (-not $LibvaSubprojectDir) {
  Write-Error "Could not find libva subproject directory under $SourceDir/subprojects/ - was libva extracted by meson setup?"
  exit 1
}

$LibvaMesonBuild = "$($LibvaSubprojectDir.FullName)/va/meson.build"

Write-Host "Patching sources..."

if (-not (Test-Path $LibvaMesonBuild)) {
  Write-Error "Could not find $LibvaMesonBuild - was libva extracted by meson setup?"
  exit 1
}

# Normalize CRLF -> LF so string matching is independent of the line endings
# of both this script (git may check it out CRLF via autocrlf) and the libva
# tarball (extracted LF). All patterns below are normalized the same way.
function Convert-Lf([string]$s) { return $s.Replace("`r`n", "`n") }

# Write UTF-8 without a BOM. Set-Content -Encoding utf8 emits a BOM under
# Windows PowerShell 5.1 (which CMakeLists invokes via `powershell`), and a
# leading BOM can break Meson parsing; the default encoding there is UTF-16LE,
# which is worse. The .NET writer is BOM-free and behaves identically on 5.1
# and 7+.
function Set-Utf8NoBom([string]$path, [string]$text) {
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($path, $text, $utf8NoBom)
}

$content = Convert-Lf (Get-Content $LibvaMesonBuild -Raw)

# Replace the libva_win32_dep declare_dependency block to:
# 1. Add fs.copyfile() loop that copies win32/va_win32.h into the flat
#    va/ build directory so consumers can #include <va/va_win32.h>.
# 2. Pass the generated copy targets as sources: so consumers get a
#    build-order edge and are not compiled before the copy completes.
# Remove libva_headers_subproject += libva_win32_headers so that the
# generic copyfile loop at the bottom does not also copy win32/va_win32.h,
# which would create a duplicate target when our explicit copyfile loop runs.
$removeSubprojectLine = '  libva_headers_subproject += libva_win32_headers'
if ($content.Contains($removeSubprojectLine)) {
  $content = $content.Replace($removeSubprojectLine + "`r`n", '')
  $content = $content.Replace($removeSubprojectLine + "`n", '')
}

# Replace the libva_win32_dep declare_dependency block to:
# 1. Add fs.copyfile() loop that copies win32/va_win32.h into the flat
#    va/ build directory so consumers can #include <va/va_win32.h>.
# 2. Pass the generated copy targets as sources: so consumers get a
#    build-order edge and are not compiled before the copy completes.
$old = @'
  libva_win32_dep = declare_dependency(
    link_with : libva_win32,
    include_directories : configinc,
    dependencies : deps)
'@

$new = @'
  libva_win32_copied_headers = []
  foreach header : libva_win32_headers
    if meson.version().version_compare('>= 0.64')
      libva_win32_copied_headers += fs.copyfile(header)
    else
      libva_win32_copied_headers += configure_file(
        output : fs.name(header), input : header, copy : true)
    endif
  endforeach

  libva_win32_dep = declare_dependency(
    link_with : libva_win32,
    sources : libva_win32_copied_headers,
    include_directories : configinc,
    dependencies : deps)
'@

$old = Convert-Lf $old
$new = Convert-Lf $new
if ($content -notmatch [regex]::Escape($old.Trim())) {
  Write-Error "Could not find expected libva_win32_dep block in $LibvaMesonBuild - patch may already be applied or file changed upstream."
  exit 1
}

$content = $content.Replace($old, $new)

# Rename the shared_library targets so meson emits rocm_sysdeps_-prefixed
# outputs (rocm_sysdeps_va.{dll,lib}, rocm_sysdeps_va_win32.{dll,lib}). This
# matches the Linux sysdeps naming and keeps all internal references consistent
# (e.g. va_win32's import of va.dll is relinked by meson automatically). The
# runtime driver name (vaon12_drv_video.dll) is unaffected.
$vaOld = @'
libva = shared_library(
  'va',
'@
$vaNew = @'
libva = shared_library(
  'rocm_sysdeps_va',
'@
$vaOld = Convert-Lf $vaOld
$vaNew = Convert-Lf $vaNew
if ($content -notmatch [regex]::Escape($vaOld)) {
  Write-Error "Could not find libva shared_library('va') declaration in $LibvaMesonBuild - patch may already be applied or file changed upstream."
  exit 1
}
$content = $content.Replace($vaOld, $vaNew)

$vaWin32Old = @'
  libva_win32 = shared_library(
    'va_win32',
'@
$vaWin32New = @'
  libva_win32 = shared_library(
    'rocm_sysdeps_va_win32',
'@
$vaWin32Old = Convert-Lf $vaWin32Old
$vaWin32New = Convert-Lf $vaWin32New
if ($content -notmatch [regex]::Escape($vaWin32Old)) {
  Write-Error "Could not find libva_win32 shared_library('va_win32') declaration in $LibvaMesonBuild - patch may already be applied or file changed upstream."
  exit 1
}
$content = $content.Replace($vaWin32Old, $vaWin32New)

Set-Utf8NoBom $LibvaMesonBuild $content

Write-Host "Patched $LibvaMesonBuild"

# Patch libva's va.c so the VA-API driver (vaon12_drv_video.dll) is found next
# to libva without a LIBVA_DRIVERS_PATH environment variable. This mirrors the
# Linux sysdeps patch in patch_source.sh. The driver search path is built in
# priority order:
#   1. The directory containing rocm_sysdeps_va.dll itself (TheRock ships libva
#      and the mesa VA driver side by side, so this is expected to succeed).
#   2. %ROCM_PATH%\lib\rocm_sysdeps\bin
#   3. %LIBVA_DRIVERS_PATH%, or the compiled-in VA_DRIVERS_PATH if unset.
# va_openDriver() walks the ';'-separated list and stops at the first driver
# that loads, so a stale ROCM_PATH or LIBVA_DRIVERS_PATH can no longer shadow
# the bundled driver.
#
# Windows-specific differences from the Linux patch:
#   * MSVC has no dladdr()/asprintf()/strndup(), so the module directory comes
#     from GetModuleHandleExA()+GetModuleFileNameA() and strings are built with
#     snprintf()+malloc(). The module path buffer grows on truncation so long
#     installation paths (beyond MAX_PATH) are still resolved.
#   * libva's compat_win32.h maps secure_getenv() to a helper that returns a
#     pointer into a single static buffer, so the LIBVA_DRIVERS_PATH value is
#     copied before ROCM_PATH is read, otherwise it would be overwritten.
#   * Runtime .dll files are installed to bin\ (lib\ holds only .lib import
#     libraries), so the ROCM_PATH entry ends in bin rather than lib.
$LibvaSource = "$($LibvaSubprojectDir.FullName)/va/va.c"
if (-not (Test-Path $LibvaSource)) {
  Write-Error "Could not find $LibvaSource - was libva extracted by meson setup?"
  exit 1
}

$va = Convert-Lf (Get-Content $LibvaSource -Raw)

# 1. Declare a scratch buffer for the combined search path.
$declOld = "    char *search_path = NULL;"
$declNew = "    char *search_path = NULL;`n    char *temp_path = NULL;"
if (-not $va.Contains($declOld)) {
  Write-Error "Could not find 'char *search_path = NULL;' in $LibvaSource - patch may already be applied or file changed upstream."
  exit 1
}
$va = $va.Replace($declOld, $declNew)

# 2. Build "<libva dir>;<ROCM_PATH>\lib\rocm_sysdeps\bin;<LIBVA_DRIVERS_PATH or
#    VA_DRIVERS_PATH>", omitting any entry that is unavailable.
$searchOld = Convert-Lf @'
    if (!search_path)
        search_path = VA_DRIVERS_PATH;
'@
$searchNew = Convert-Lf @'
    {
        char *env_dir = search_path ? strdup(search_path) : NULL;
        const char *last_dir = env_dir ? env_dir : VA_DRIVERS_PATH;
        char *self_dir = NULL;
        char *rocm_dir = NULL;
        char *rocm_path;
        HMODULE self_module = NULL;
        char *module_path = NULL;
        DWORD size = MAX_PATH;
        DWORD len = 0;
        int n;

        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)&va_openDriver, &self_module)) {
            /* GetModuleFileNameA() returns the buffer size when the path was
             * truncated, so grow the buffer until it fits (long paths). */
            for (;;) {
                char *buf = (char *)realloc(module_path, size);
                if (!buf) {
                    len = 0;
                    break;
                }
                module_path = buf;
                len = GetModuleFileNameA(self_module, module_path, size);
                if (len < size)
                    break;
                len = 0;
                if (size >= 32768)
                    break;
                size = size * 2 > 32768 ? 32768 : size * 2;
            }
            if (len > 0) {
                char *last_sep = strrchr(module_path, '\\');
                char *last_fwd = strrchr(module_path, '/');
                if (!last_sep || (last_fwd && last_fwd > last_sep))
                    last_sep = last_fwd;
                if (last_sep) {
                    *last_sep = '\0';
                    self_dir = strdup(module_path);
                }
            }
            free(module_path);
        }

        rocm_path = secure_getenv("ROCM_PATH");
        if (rocm_path) {
            n = snprintf(NULL, 0, "%s\\lib\\rocm_sysdeps\\bin", rocm_path);
            if (n >= 0 && (rocm_dir = (char *)malloc(n + 1)))
                snprintf(rocm_dir, n + 1, "%s\\lib\\rocm_sysdeps\\bin", rocm_path);
        }

        n = snprintf(NULL, 0, "%s%s%s%s%s",
                     self_dir ? self_dir : "", self_dir ? ENV_VAR_SEPARATOR : "",
                     rocm_dir ? rocm_dir : "", rocm_dir ? ENV_VAR_SEPARATOR : "",
                     last_dir);
        if (n >= 0 && (temp_path = (char *)malloc(n + 1))) {
            snprintf(temp_path, n + 1, "%s%s%s%s%s",
                     self_dir ? self_dir : "", self_dir ? ENV_VAR_SEPARATOR : "",
                     rocm_dir ? rocm_dir : "", rocm_dir ? ENV_VAR_SEPARATOR : "",
                     last_dir);
            search_path = temp_path;
        } else {
            search_path = VA_DRIVERS_PATH;
        }

        free(self_dir);
        free(rocm_dir);
        free(env_dir);
    }
'@
if (-not $va.Contains($searchOld)) {
  Write-Error "Could not find LIBVA_DRIVERS_PATH fallback block in $LibvaSource - patch may already be applied or file changed upstream."
  exit 1
}
$va = $va.Replace($searchOld, $searchNew)

# 3. Free the scratch buffer once search_path has been strdup()'d.
$freeOld = "    search_path = strdup((const char *)search_path);"
$freeNew = "    search_path = strdup((const char *)search_path);`n    if (temp_path) { free(temp_path); temp_path = NULL; }"
if (-not $va.Contains($freeOld)) {
  Write-Error "Could not find 'search_path = strdup(...)' in $LibvaSource - patch may already be applied or file changed upstream."
  exit 1
}
$va = $va.Replace($freeOld, $freeNew)

Set-Utf8NoBom $LibvaSource $va

Write-Host "Patched $LibvaSource"
