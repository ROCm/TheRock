# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory=$true)]
  [string]$SourceDir
)

$ErrorActionPreference = "Stop"

$LibvaMesonBuild = "$SourceDir/subprojects/libva-2.22.0/va/meson.build"

Write-Host "Patching sources..."

if (-not (Test-Path $LibvaMesonBuild)) {
  Write-Error "Could not find $LibvaMesonBuild - was libva extracted by meson setup?"
  exit 1
}

$content = Get-Content $LibvaMesonBuild -Raw

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

if ($content -notmatch [regex]::Escape($old.Trim())) {
  Write-Error "Could not find expected libva_win32_dep block in $LibvaMesonBuild - patch may already be applied or file changed upstream."
  exit 1
}

$content = $content.Replace($old, $new)
Set-Content $LibvaMesonBuild $content -NoNewline

Write-Host "Patched $LibvaMesonBuild"
