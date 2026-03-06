#!/usr/bin/env pwsh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

<#
.SYNOPSIS
    Validates the Windows development environment for building TheRock/ROCm from source.

.DESCRIPTION
    Checks system configuration, required tools, and settings needed for a
    successful TheRock source build on Windows. Outputs a checklist-style
    report with PASS/WARN/FAIL for each item.

    See docs/development/windows_support.md for setup instructions.

.EXAMPLE
    .\build_tools\validate_windows_install.ps1
#>
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"

$script:PassCount = 0
$script:WarnCount = 0
$script:FailCount = 0

function Write-Pass {
    param([string]$Message, [string]$Detail = "")
    Write-Host "  [PASS] " -ForegroundColor Green -NoNewline
    Write-Host $Message
    if ($Detail) { Write-Host "         $Detail" -ForegroundColor DarkGray }
    $script:PassCount++
}

function Write-Warn {
    param([string]$Message, [string]$Detail = "")
    Write-Host "  [WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
    if ($Detail) { Write-Host "         $Detail" -ForegroundColor DarkGray }
    $script:WarnCount++
}

function Write-Fail {
    param([string]$Message, [string]$Detail = "")
    Write-Host "  [FAIL] " -ForegroundColor Red -NoNewline
    Write-Host $Message
    if ($Detail) { Write-Host "         $Detail" -ForegroundColor DarkGray }
    $script:FailCount++
}

function Write-Info {
    param([string]$Message)
    Write-Host "  [info] " -ForegroundColor Cyan -NoNewline
    Write-Host $Message -ForegroundColor DarkGray
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host $Title -ForegroundColor White
    Write-Host ("-" * $Title.Length) -ForegroundColor DarkGray
}

$RepoRoot = Split-Path $PSScriptRoot -Parent

# ============================================================================
Write-Host ""
Write-Host "TheRock Windows Environment Validator" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor DarkCyan
Write-Host "Repo: $RepoRoot"
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

# ============================================================================
Write-Section "1. System Resources"

# RAM
try {
    $ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
    if ($ramGB -ge 32) {
        Write-Pass "RAM: ${ramGB} GB"
    } elseif ($ramGB -ge 16) {
        Write-Warn "RAM: ${ramGB} GB (16 GB minimum met; 32 GB+ recommended for full builds)"
    } else {
        Write-Fail "RAM: ${ramGB} GB (16 GB minimum required)"
    }
} catch {
    Write-Warn "Could not determine total RAM"
}

# Free disk space on the drive hosting the repo
try {
    $driveLetter = (Split-Path $RepoRoot -Qualifier).TrimEnd(':')
    $drive = Get-PSDrive -Name $driveLetter
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    if ($freeGB -ge 200) {
        Write-Pass "Free disk on ${driveLetter}: ${freeGB} GB"
    } elseif ($freeGB -ge 80) {
        Write-Warn "Free disk on ${driveLetter}: ${freeGB} GB (200 GB recommended for a full build)"
    } else {
        Write-Fail "Free disk on ${driveLetter}: ${freeGB} GB (200 GB recommended; full builds use 100-200 GB)"
    }
} catch {
    Write-Warn "Could not determine free disk space on repo drive"
}

# ============================================================================
Write-Section "2. Windows System Configuration"

# Long paths (registry)
try {
    $longPathsVal = (Get-ItemProperty `
        -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
        -Name "LongPathsEnabled").LongPathsEnabled
    if ($longPathsVal -eq 1) {
        Write-Pass "Long path support enabled (registry LongPathsEnabled=1)"
    } else {
        Write-Fail "Long path support NOT enabled" `
            -Detail "Fix (as admin): reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f"
    }
} catch {
    Write-Fail "Could not read LongPathsEnabled (try running as administrator)" `
        -Detail "See: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation"
}

# Long path functional test - actually create/read a file beyond MAX_PATH (260 chars).
# The registry key is necessary but not sufficient; apps must also opt in, and
# some older tools silently truncate or fail on long paths.
# Use a 250-char directory name so the full path exceeds 260 even on short %TEMP% values.
$longDirName = "therock_longpath_test_" + ("a" * 228)   # 250 chars total
$longDir  = Join-Path $env:TEMP $longDirName
$longFile = Join-Path $longDir "test.txt"
try {
    New-Item -ItemType Directory -Path $longDir -Force -ErrorAction Stop | Out-Null
    Set-Content -Path $longFile -Value "ok" -ErrorAction Stop
    $readBack = Get-Content -Path $longFile -ErrorAction Stop
    if ($readBack -eq "ok") {
        Write-Pass "Long path I/O works (tested at $($longFile.Length) chars)"
    } else {
        Write-Fail "Long path file read returned unexpected content"
    }
} catch {
    Write-Fail "Long path I/O failed (path length: $($longFile.Length) chars)" `
        -Detail "$($_.Exception.Message)"
} finally {
    Remove-Item $longDir -Recurse -Force -ErrorAction SilentlyContinue
}

# Developer Mode (allows symlinks without elevation)
try {
    $devModeVal = (Get-ItemProperty `
        -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" `
        -Name "AllowDevelopmentWithoutDevLicense").AllowDevelopmentWithoutDevLicense
    if ($devModeVal -eq 1) {
        Write-Pass "Developer Mode enabled (symlinks without elevation)"
    } else {
        Write-Warn "Developer Mode not enabled" `
            -Detail "Enable via: Settings > System > For developers > Developer Mode"
    }
} catch {
    Write-Warn "Could not read Developer Mode registry key" `
        -Detail "Enable via: Settings > System > For developers > Developer Mode"
}

# Symlink creation - actually try it
$tmpTarget = Join-Path $env:TEMP "therock_val_tgt_$([System.Guid]::NewGuid().ToString('N'))"
$tmpLink   = Join-Path $env:TEMP "therock_val_lnk_$([System.Guid]::NewGuid().ToString('N'))"
$symlinkOk = $false
try {
    New-Item -ItemType File -Path $tmpTarget -Force | Out-Null
    $null = New-Item -ItemType SymbolicLink -Path $tmpLink -Target $tmpTarget -ErrorAction Stop
    $symlinkOk = $true
    Write-Pass "Symbolic link creation works"
} catch {
    # If Developer Mode is already enabled, the likely cause is a stale session token
    # (the privilege is only inherited by new logon sessions) or a Group Policy override.
    $hint = if ($devModeVal -eq 1) {
        "Developer Mode is ON but this session predates it, or a Group Policy is overriding it. " +
        "Try: start a new terminal session, or check 'secpol.msc > Local Policies > User Rights Assignment > Create symbolic links'"
    } else {
        "Enable Developer Mode (Settings > System > For developers) or grant 'Create symbolic links' in secpol.msc"
    }
    Write-Fail "Cannot create symbolic links: $($_.Exception.Message)" -Detail $hint
} finally {
    Remove-Item $tmpLink   -Force -ErrorAction SilentlyContinue
    Remove-Item $tmpTarget -Force -ErrorAction SilentlyContinue
}

# Warn about conflicting HIP SDK / ROCm installs
$conflicts = @()
if ($env:HIP_PATH)  { $conflicts += "HIP_PATH=$($env:HIP_PATH)" }
if ($env:ROCM_PATH) { $conflicts += "ROCM_PATH=$($env:ROCM_PATH)" }
if ($conflicts.Count -gt 0) {
    Write-Warn "Existing HIP/ROCm SDK variables detected - may conflict with the build" `
        -Detail ($conflicts -join "  |  ")
    Write-Info "See GitHub issue #651; consider uninstalling the HIP SDK before building"
} else {
    Write-Pass "No conflicting HIP_PATH / ROCM_PATH environment variables"
}

# Active code page (important on non-English systems)
try {
    $chcpOutput = (& chcp 2>&1) -replace "[^0-9]", ""
    # 65001 = UTF-8, 437 = English US (OEM) - both are fine for English systems.
    # Non-Latin code pages (e.g. 932 Shift-JIS, 936 GBK) can corrupt source paths.
    $latinCodePages = @("65001", "437", "850", "858", "1252")
    if ($chcpOutput -eq "65001") {
        Write-Pass "Active code page is UTF-8 (65001)"
    } elseif ($chcpOutput -in $latinCodePages) {
        Write-Pass "Active code page is $chcpOutput (Latin/English - OK)"
    } else {
        Write-Warn "Active code page is $chcpOutput (non-Latin)" `
            -Detail "Run 'chcp 65001' before building to avoid path encoding issues"
    }
} catch {
    Write-Warn "Could not determine active code page"
}

# ============================================================================
Write-Section "3. MSVC Compiler"

$clCmd = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clCmd) {
    # cl.exe prints its version banner to stderr
    $clBanner = (& cl.exe 2>&1 | Select-Object -First 1).ToString().Trim()
    Write-Pass "cl.exe in PATH: $($clCmd.Source)"
    Write-Info $clBanner
} else {
    Write-Warn "cl.exe not in PATH - checking for VS installation..."

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = (& $vswhere -latest `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null)
        if ($vsPath) {
            Write-Warn "Visual Studio with MSVC found but developer environment not activated" `
                -Detail "Run: `"$vsPath\VC\Auxiliary\Build\vcvars64.bat`""
        } else {
            $vsAnyPath = (& $vswhere -latest -property installationPath 2>$null)
            if ($vsAnyPath) {
                Write-Fail "Visual Studio found but VC Tools (x86/x64) component is missing" `
                    -Detail "Add component: Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
            } else {
                Write-Fail "No Visual Studio installation found" `
                    -Detail "Install: winget install --id Microsoft.VisualStudio.2022.BuildTools"
            }
        }
    } else {
        Write-Fail "cl.exe not found and vswhere.exe is unavailable" `
            -Detail "Install Visual Studio 2022 Build Tools with MSVC component"
    }
}

# Check developer environment activation and architecture
if ($env:VCINSTALLDIR) {
    $arch = $env:VSCMD_ARG_TGT_ARCH
    if ($arch -eq "x64") {
        Write-Pass "MSVC developer environment activated (x64)"
    } elseif ($arch) {
        Write-Fail "MSVC developer environment activated for wrong arch: $arch (x64 required)" `
            -Detail "Use 'x64 Native Tools Command Prompt for VS 2022' or vcvars64.bat"
    } else {
        Write-Pass "MSVC developer environment activated (VCINSTALLDIR set)"
    }
} else {
    Write-Warn "MSVC developer environment not activated (VCINSTALLDIR not set)" `
        -Detail "Use 'x64 Native Tools Command Prompt for VS 2022' or run vcvars64.bat"
}

# ATL (required by the build)
if ($env:VCINSTALLDIR) {
    $atlPath = Join-Path $env:VCINSTALLDIR "Tools\MSVC\*\atlmfc\include\atlbase.h"
    if (Test-Path $atlPath) {
        Write-Pass "C++ ATL headers found"
    } else {
        Write-Fail "C++ ATL headers not found" `
            -Detail "Add component: Microsoft.VisualStudio.Component.VC.ATL"
    }
} else {
    Write-Info "Skipping ATL check (MSVC environment not activated)"
}

# rc.exe (Windows Resource Compiler - lives in SDK bin, not MSVC bin)
$rcCmd = Get-Command rc.exe -ErrorAction SilentlyContinue
if ($rcCmd) {
    Write-Pass "rc.exe in PATH: $($rcCmd.Source)"
} else {
    Write-Fail "rc.exe not found in PATH (Windows SDK bin directory missing)" `
        -Detail "The VS dev shell may not have set up the SDK paths. Check that the Windows SDK is installed and its bin\x64 dir is on PATH."
}

# Test compile: exercises headers from each SDK include directory and links
# kernel32.lib.  This is the most reliable way to detect a broken VS dev
# shell where cl.exe is on PATH but SDK include/lib dirs are not configured.
#   windows.h  → Include\<ver>\um
#   stdio.h    → Include\<ver>\ucrt
#   rpcndr.h   → Include\<ver>\shared
#   winstring.h→ Include\<ver>\winrt
#   Link kernel32.lib → Lib\<ver>\um\x64 + Lib\<ver>\ucrt\x64
if ($clCmd) {
    $testDir = Join-Path $env:TEMP "therock_compile_test_$([System.Guid]::NewGuid().ToString('N'))"
    $testSrc = Join-Path $testDir "test.c"
    $testExe = Join-Path $testDir "test.exe"
    try {
        New-Item -ItemType Directory -Path $testDir -Force -ErrorAction Stop | Out-Null
        Set-Content -Path $testSrc -Value @"
#include <windows.h>   /* um */
#include <stdio.h>     /* ucrt */
#include <rpcndr.h>    /* shared */
#include <winstring.h> /* winrt */
int main(void) {
    HANDLE h = GetProcessHeap();
    printf("ok %p\n", (void*)h);
    return 0;
}
"@ -ErrorAction Stop
        $compileOut = (& cl.exe /nologo /Fo:"$testDir\" /Fe:"$testExe" "$testSrc" /link kernel32.lib 2>&1) -join "`n"
        if ((Test-Path $testExe)) {
            Write-Pass "Test compile + link succeeded (um, ucrt, shared, winrt headers; kernel32.lib)"
        } else {
            Write-Fail "Test compile failed - SDK environment is incomplete" `
                -Detail $compileOut.Substring(0, [Math]::Min(300, $compileOut.Length))
        }
    } catch {
        Write-Fail "Test compile could not run: $($_.Exception.Message)"
    } finally {
        Remove-Item $testDir -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Info "Skipping test-compile check (cl.exe not available)"
}

# ============================================================================
Write-Section "4. Core Build Tools"

# CMake
$cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCmd) {
    $cmakeVerStr = ((& cmake --version 2>&1 | Select-Object -First 1) -replace "cmake version ", "").Trim()
    Write-Pass "CMake $cmakeVerStr"
} else {
    Write-Fail "CMake not found in PATH" -Detail "Install: winget install cmake"
}

# Ninja
$ninjaCmd = Get-Command ninja -ErrorAction SilentlyContinue
if ($ninjaCmd) {
    $ninjaVer = ((& ninja --version 2>&1) -join "").Trim()
    Write-Pass "Ninja $ninjaVer"
} else {
    Write-Fail "Ninja not found in PATH" -Detail "Install: winget install ninja-build.ninja"
}

# Git
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if ($gitCmd) {
    $gitVer = ((& git --version 2>&1) -replace "git version ", "").Trim()
    Write-Pass "Git $gitVer"
} else {
    Write-Fail "Git not found in PATH" `
        -Detail "Install: winget install Git.Git --custom `"/o:PathOption=CmdTools`""
}

# bash (from Git for Windows Unix tools - needed by build scripts)
$bashCmd = Get-Command bash -ErrorAction SilentlyContinue
if ($bashCmd) {
    Write-Pass "bash.exe found (Unix tools available): $($bashCmd.Source)"
} else {
    Write-Fail "bash.exe not found in PATH" `
        -Detail "Reinstall Git with 'Use Git and optional Unix tools from the Windows Command Prompt'"
}

# patch (needed for fetch_sources.py)
$patchCmd = Get-Command patch -ErrorAction SilentlyContinue
if ($patchCmd) {
    Write-Pass "patch found: $($patchCmd.Source)"
} else {
    Write-Fail "patch not found in PATH" `
        -Detail "Available via Strawberry Perl or Git for Windows (with Unix tools)"
}

# DVC (for CLR interop files)
$dvcCmd = Get-Command dvc -ErrorAction SilentlyContinue
if ($dvcCmd) {
    $dvcVer = ((& dvc --version 2>&1 | Select-Object -First 1) -join "").Trim()
    Write-Pass "DVC $dvcVer"
} else {
    Write-Fail "DVC not found in PATH" `
        -Detail "Required for CLR interop files. Install: winget install --id Iterative.DVC"
}

# ============================================================================
Write-Section "5. Python"

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    $pyVerStr = ((& python --version 2>&1) -replace "Python ", "").Trim()
    try {
        $pyVer = [Version]$pyVerStr
        if ($pyVer -ge [Version]"3.11") {
            Write-Pass "Python $pyVerStr (>= 3.11)"
        } elseif ($pyVer -ge [Version]"3.8") {
            Write-Warn "Python $pyVerStr (3.11+ recommended)"
        } else {
            Write-Fail "Python $pyVerStr (3.11+ recommended)"
        }
    } catch {
        Write-Pass "Python found: $pyVerStr"
    }

    # Python install path must not contain spaces
    $pyPath = $pythonCmd.Source
    if ($pyPath -match " ") {
        Write-Warn "Python installed in a path with spaces: $pyPath" `
            -Detail "Prefer a path without spaces (e.g. C:\Users\<user>\AppData\Local\Programs\Python\...)"
    } else {
        Write-Pass "Python path has no spaces: $pyPath"
    }
} else {
    Write-Fail "Python not found in PATH" `
        -Detail "Install Python 3.11+ from https://www.python.org/downloads/ (avoid paths with spaces)"
}

# Virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Pass "Virtual environment active: $($env:VIRTUAL_ENV)"

    # Check requirements are installed
    $reqFile = Join-Path $RepoRoot "requirements.txt"
    if (Test-Path $reqFile) {
        # Use pip's dry-run to check requirements - this correctly evaluates
        # environment markers like "; platform_system != 'Windows'" and
        # "; python_version <= '3.10'" so conditional deps are not falsely flagged.
        $dryRun = (& python -m pip install -r $reqFile --dry-run 2>&1)
        $wouldInstall = $dryRun | Where-Object { $_ -match "^Would install" }
        $pipError     = $dryRun | Where-Object { $_ -match "^ERROR" }
        if ($pipError) {
            Write-Warn "pip dry-run reported errors checking requirements.txt" `
                -Detail ($pipError -join "; ")
        } elseif ($wouldInstall) {
            # "Collecting pkg>=X,<Y (from -r ...)" lines tell us what needs changing
            $collecting = $dryRun | Where-Object { $_ -match "^Collecting " }
            Write-Warn "Some requirements.txt packages need installing or upgrading"
            Write-Info "Run: pip install -r requirements.txt"
            foreach ($line in $collecting) {
                # e.g. "Collecting boto3<1.42.0,>=1.41.0 (from -r requirements.txt (line 16))"
                if ($line -match "^Collecting ([^\s(]+)") {
                    $spec = $matches[1]  # e.g. "boto3<1.42.0,>=1.41.0"
                    $name = $spec -replace "[><=!,].*$", ""
                    $installed = (& python -m pip show $name 2>$null) |
                        Where-Object { $_ -match "^Version:" } |
                        ForEach-Object { $_ -replace "Version:\s*", "" }
                    $installedStr = if ($installed) { "installed: $installed" } else { "not installed" }
                    Write-Info "  $spec  ($installedStr)"
                }
            }
        } else {
            Write-Pass "All requirements.txt packages satisfied (markers evaluated by pip)"
        }
    }
} else {
    $venvPath = Join-Path $RepoRoot ".venv"
    if (Test-Path $venvPath) {
        Write-Warn "Virtual environment exists (.venv) but is NOT activated" `
            -Detail "Run: .venv\Scripts\Activate.bat"
    } else {
        Write-Warn "No virtual environment active" `
            -Detail "Run: python -m venv .venv && .venv\Scripts\Activate.bat && pip install -r requirements.txt"
    }
}

# ============================================================================
Write-Section "6. Fortran and Perl (Strawberry Perl)"

$perlCmd = Get-Command perl -ErrorAction SilentlyContinue
if ($perlCmd) {
    $perlSrc = $perlCmd.Source
    if ($perlSrc -imatch "Strawberry") {
        Write-Pass "Strawberry Perl: $perlSrc"
    } else {
        Write-Pass "Perl found: $perlSrc"
        Write-Info "Strawberry Perl is recommended (also provides gfortran and patch)"
    }
} else {
    Write-Fail "Perl not found in PATH" `
        -Detail "Install Strawberry Perl: winget install strawberryperl"
}

$gfortranCmd = Get-Command gfortran -ErrorAction SilentlyContinue
if ($gfortranCmd) {
    $gfVer = ((& gfortran --version 2>&1 | Select-Object -First 1) -replace "GNU Fortran \(.*?\) ", "").Trim()
    Write-Pass "gfortran $gfVer : $($gfortranCmd.Source)"
} else {
    Write-Fail "gfortran not found in PATH" `
        -Detail "Install Strawberry Perl (includes gfortran): winget install strawberryperl"
}

# ============================================================================
Write-Section "7. Optional / Recommended Tools"

# ccache or sccache (for faster incremental rebuilds)
$ccacheCmd  = Get-Command ccache  -ErrorAction SilentlyContinue
$sccacheCmd = Get-Command sccache -ErrorAction SilentlyContinue
if ($ccacheCmd) {
    $ccVer = ((& ccache --version 2>&1 | Select-Object -First 1) -replace "ccache version ", "").Trim()
    Write-Pass "ccache $ccVer (speeds up rebuilds)"
    Write-Info "Note: use -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded with ccache to avoid /Zi conflicts"
} elseif ($sccacheCmd) {
    $scVer = ((& sccache --version 2>&1) -replace "sccache ", "").Trim()
    Write-Pass "sccache $scVer (speeds up rebuilds)"
} else {
    Write-Warn "Neither ccache nor sccache found" `
        -Detail "Recommended for faster rebuilds: winget install ccache"
}

# pkg-config
$pkgConfigCmd = Get-Command pkg-config -ErrorAction SilentlyContinue
if ($pkgConfigCmd) {
    Write-Pass "pkg-config found: $($pkgConfigCmd.Source)"
} else {
    Write-Warn "pkg-config not found" `
        -Detail "Install: winget install bloodrock.pkg-config-lite"
}

# ============================================================================
Write-Section "8. Git Configuration"

if ($gitCmd) {
    # core.symlinks
    $gitSymlinks = ((& git config --global core.symlinks 2>&1) -join "").Trim()
    if ($gitSymlinks -eq "true") {
        Write-Pass "git config --global core.symlinks = true"
    } else {
        Write-Fail "git config --global core.symlinks = '$gitSymlinks' (should be true)" `
            -Detail "Run: git config --global core.symlinks true"
    }

    # core.longpaths
    $gitLongpaths = ((& git config --global core.longpaths 2>&1) -join "").Trim()
    if ($gitLongpaths -eq "true") {
        Write-Pass "git config --global core.longpaths = true"
    } else {
        Write-Fail "git config --global core.longpaths = '$gitLongpaths' (should be true)" `
            -Detail "Run: git config --global core.longpaths true"
    }

    # core.autocrlf - warn if true
    $gitAutoCrlf = ((& git config --global core.autocrlf 2>&1) -join "").Trim()
    if ($gitAutoCrlf -eq "true") {
        Write-Warn "git config --global core.autocrlf = true (may corrupt build scripts)" `
            -Detail "Recommended: git config --global core.autocrlf input"
    } elseif ($gitAutoCrlf -eq "input") {
        Write-Pass "git config --global core.autocrlf = input"
    } else {
        Write-Info "git config --global core.autocrlf = '$gitAutoCrlf'"
    }
} else {
    Write-Info "Skipping git config checks (git not found)"
}

# ============================================================================
Write-Section "9. Repository State"

# Submodule population check (sample a few key ones)
$keySubmodulePaths = @(
    "rocm-libraries",
    "rocm-systems",
    "compiler/llvm-project"
)
$uninitCount = 0
foreach ($subPath in $keySubmodulePaths) {
    $fullPath = Join-Path $RepoRoot $subPath
    $hasContent = (Test-Path $fullPath) -and
                  ((Get-ChildItem $fullPath -ErrorAction SilentlyContinue).Count -gt 0)
    if (-not $hasContent) { $uninitCount++ }
}
if ($uninitCount -eq 0) {
    Write-Pass "Key submodules appear populated (rocm-libraries, rocm-systems, llvm-project)"
} elseif ($uninitCount -lt $keySubmodulePaths.Count) {
    Write-Warn "$uninitCount/$($keySubmodulePaths.Count) checked submodules appear empty" `
        -Detail "Run: python build_tools/fetch_sources.py"
} else {
    Write-Warn "Submodules do not appear to be initialized" `
        -Detail "Run: python build_tools/fetch_sources.py"
}

# Python requirements file present
$reqFile = Join-Path $RepoRoot "requirements.txt"
if (Test-Path $reqFile) {
    Write-Pass "requirements.txt present"
} else {
    Write-Warn "requirements.txt not found at $reqFile"
}

# ============================================================================
Write-Host ""
Write-Host "======================================" -ForegroundColor DarkCyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor DarkCyan
Write-Host ("  {0,-6} PASS" -f $script:PassCount) -ForegroundColor Green
Write-Host ("  {0,-6} WARN" -f $script:WarnCount) -ForegroundColor Yellow
Write-Host ("  {0,-6} FAIL" -f $script:FailCount) -ForegroundColor Red
Write-Host ""

if ($script:FailCount -gt 0) {
    Write-Host "Action required: address FAIL items before attempting a build." -ForegroundColor Red
    Write-Host "See docs/development/windows_support.md for setup instructions."
} elseif ($script:WarnCount -gt 0) {
    Write-Host "Environment is mostly ready - review WARN items for potential issues." -ForegroundColor Yellow
} else {
    Write-Host "All checks passed - environment looks ready to build TheRock." -ForegroundColor Green
}
Write-Host ""

exit $script:FailCount
