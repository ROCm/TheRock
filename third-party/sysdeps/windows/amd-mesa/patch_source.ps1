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

# Patch libva's va.c so the VA-API driver (vaon12_drv_video.dll) is located
# relative to ROCM_PATH, removing the need for a LIBVA_DRIVERS_PATH environment
# variable. This mirrors the Linux sysdeps patch (patch_source.sh lines 47-67),
# but builds the path with a portable snprintf()+malloc() sequence because MSVC
# has no asprintf(). secure_getenv() is supplied on Windows by libva's own
# compat_win32.h shim. The search path ends in bin/ (not lib/) because Windows
# installs runtime .dll files to bin/ while lib/ holds only .lib import libs.
$LibvaSource = "$($LibvaSubprojectDir.FullName)/va/va.c"
if (-not (Test-Path $LibvaSource)) {
  Write-Error "Could not find $LibvaSource - was libva extracted by meson setup?"
  exit 1
}

$va = Convert-Lf (Get-Content $LibvaSource -Raw)

# 1. Declare a scratch buffer for the ROCM_PATH-derived search path.
$declOld = "    char *search_path = NULL;"
$declNew = "    char *search_path = NULL;`n    char *temp_path = NULL;"
if (-not $va.Contains($declOld)) {
  Write-Error "Could not find 'char *search_path = NULL;' in $LibvaSource - patch may already be applied or file changed upstream."
  exit 1
}
$va = $va.Replace($declOld, $declNew)

# 2. Prefer <ROCM_PATH>/lib/rocm_sysdeps/lib, falling back to VA_DRIVERS_PATH.
$searchOld = Convert-Lf @'
    if (!search_path)
        search_path = VA_DRIVERS_PATH;
'@
$searchNew = Convert-Lf @'
    if (!search_path) {
        char *rocm_path = secure_getenv("ROCM_PATH");
        if (rocm_path) {
            const char *suffix = "/lib/rocm_sysdeps/bin";
            int n = snprintf(NULL, 0, "%s%s", rocm_path, suffix);
            if (n < 0) {
                temp_path = NULL;
            } else {
                temp_path = (char *)malloc(n + 1);
                if (temp_path) {
                    snprintf(temp_path, n + 1, "%s%s", rocm_path, suffix);
                    search_path = temp_path;
                }
            }
        } else {
            search_path = VA_DRIVERS_PATH;
        }
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
