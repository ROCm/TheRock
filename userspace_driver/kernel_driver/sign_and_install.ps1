#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Sign and install the AMDGPU MCDM kernel driver.

.DESCRIPTION
    Creates a test certificate, generates a catalog file (inf2cat),
    signs both .sys and .cat, and installs via pnputil.
    Must be run from an elevated PowerShell prompt.

.EXAMPLE
    .\sign_and_install.ps1
    .\sign_and_install.ps1 -SkipSign    # If already signed
    .\sign_and_install.ps1 -Uninstall   # Remove the driver
#>

param(
    [switch]$SkipSign,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

# --- Paths ---
$DriverDir   = Join-Path $PSScriptRoot "x64\Release"
$DriverSys   = Join-Path $DriverDir "amdgpu_mcdm.sys"
$DriverInf   = Join-Path $DriverDir "amdgpu_mcdm.inf"
$DriverCat   = Join-Path $DriverDir "amdgpu_mcdm.cat"
$SrcInf      = Join-Path $PSScriptRoot "amdgpu_mcdm.inf"
$CertFile    = Join-Path $PSScriptRoot "AmdGpuTest.cer"
$CertName    = "AMDGPU Test"
$CertStore   = "PrivateCertStore"

# --- WDK tools ---
$WdkBin64  = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
$WdkBinX86 = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86"
$SignTool  = Join-Path $WdkBin64 "signtool.exe"
$MakeCert  = Join-Path $WdkBin64 "makecert.exe"
$Inf2Cat   = Join-Path $WdkBinX86 "Inf2Cat.exe"

function Write-Step($n, $total, $msg) {
    Write-Host "`n[$n/$total] $msg" -ForegroundColor Cyan
}

# --- Uninstall path ---
if ($Uninstall) {
    Write-Host "=== Uninstalling AMDGPU MCDM driver ===" -ForegroundColor Yellow

    $drivers = pnputil /enum-drivers 2>&1 | Out-String
    $pattern = "(?m)^Published Name\s*:\s*(oem\d+\.inf).*?Original Name\s*:\s*amdgpu_mcdm\.inf"
    if ($drivers -match $pattern) {
        $oemInf = $Matches[1]
        Write-Host "  Removing $oemInf..."
        pnputil /delete-driver $oemInf /uninstall /force
    } else {
        Write-Host "  Driver not found in driver store."
    }
    return
}

# --- Preflight checks ---
Write-Host "============================================================" -ForegroundColor Green
Write-Host " AMD GPU MCDM Driver - Sign and Install" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

if (-not (Test-Path $DriverSys)) {
    Write-Error "Driver not found at $DriverSys`nBuild first: msbuild amdgpu_mcdm.vcxproj /p:Configuration=Release /p:Platform=x64"
}

foreach ($tool in @($SignTool, $MakeCert, $Inf2Cat)) {
    if (-not (Test-Path $tool)) {
        Write-Error "WDK tool not found: $tool`nInstall WDK 10.0.26100.0"
    }
}

$totalSteps = if ($SkipSign) { 3 } else { 7 }

# --- Step 1: Test signing ---
Write-Step 1 $totalSteps "Checking test signing..."

$bcd = bcdedit /enum "{current}" 2>&1 | Out-String
if ($bcd -match "testsigning\s+Yes") {
    Write-Host "  Test signing is enabled." -ForegroundColor Green
} else {
    Write-Host "  Test signing is NOT enabled. Enabling..." -ForegroundColor Yellow
    bcdedit -set testsigning on
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to enable test signing."
    }
    Write-Host ""
    Write-Host "  *** REBOOT REQUIRED ***" -ForegroundColor Red
    Write-Host "  Test signing was just enabled. Reboot, then run this script again." -ForegroundColor Red
    return
}

if (-not $SkipSign) {
    # --- Step 2: Create test certificate ---
    Write-Step 2 $totalSteps "Creating test certificate..."

    if (Test-Path $CertFile) {
        Write-Host "  Certificate already exists: $CertFile"
        Write-Host "  Delete it to regenerate."
    } else {
        & $MakeCert -r -pe -ss $CertStore -n "CN=$CertName" $CertFile
        if ($LASTEXITCODE -ne 0) {
            Write-Error "makecert failed."
        }
        Write-Host "  Created: $CertFile"

        certutil -addstore Root $CertFile
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  WARNING: Could not add cert to Trusted Root store." -ForegroundColor Yellow
        } else {
            Write-Host "  Added to Trusted Root CA store." -ForegroundColor Green
        }

        certutil -addstore TrustedPublisher $CertFile
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  WARNING: Could not add cert to Trusted Publisher store." -ForegroundColor Yellow
        } else {
            Write-Host "  Added to Trusted Publisher store." -ForegroundColor Green
        }
    }

    # --- Step 3: Copy INF to build output ---
    Write-Step 3 $totalSteps "Preparing driver package..."

    Copy-Item $SrcInf $DriverInf -Force
    Write-Host "  INF copied to $DriverInf"

    # --- Step 4: Generate catalog file ---
    Write-Step 4 $totalSteps "Generating catalog file (inf2cat)..."

    # Remove old .cat if present
    if (Test-Path $DriverCat) {
        Remove-Item $DriverCat -Force
    }

    & $Inf2Cat /driver:$DriverDir /os:10_x64,10_NI_x64,10_VB_x64 /uselocaltime
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  inf2cat failed. Trying with just 10_x64..." -ForegroundColor Yellow
        & $Inf2Cat /driver:$DriverDir /os:10_x64 /uselocaltime
        if ($LASTEXITCODE -ne 0) {
            Write-Error "inf2cat failed. Check INF syntax."
        }
    }

    if (-not (Test-Path $DriverCat)) {
        Write-Error "Catalog file was not created at $DriverCat"
    }
    Write-Host "  Catalog created: $DriverCat" -ForegroundColor Green

    # --- Step 5: Sign .sys ---
    Write-Step 5 $totalSteps "Signing driver (.sys)..."

    & $SignTool sign /s $CertStore /n $CertName /fd sha256 /v $DriverSys
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to sign .sys"
    }
    Write-Host "  amdgpu_mcdm.sys signed." -ForegroundColor Green

    # --- Step 6: Sign .cat ---
    Write-Step 6 $totalSteps "Signing catalog (.cat)..."

    & $SignTool sign /s $CertStore /n $CertName /fd sha256 /v $DriverCat
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to sign .cat"
    }
    Write-Host "  amdgpu_mcdm.cat signed." -ForegroundColor Green
}

# --- Copy INF if skipping sign (may not have been copied) ---
if ($SkipSign) {
    $stepCopy = 2
    Write-Step $stepCopy $totalSteps "Preparing driver package..."
    if (-not (Test-Path $DriverInf)) {
        Copy-Item $SrcInf $DriverInf -Force
        Write-Host "  INF copied to $DriverInf"
    } else {
        Write-Host "  INF already in place."
    }
}

# --- Install ---
$stepInstall = if ($SkipSign) { 3 } else { 7 }
Write-Step $stepInstall $totalSteps "Installing driver..."

$result = pnputil /add-driver $DriverInf /install 2>&1 | Out-String
Write-Host $result

if ($LASTEXITCODE -ne 0) {
    Write-Host "  pnputil /install failed. Trying add-only..." -ForegroundColor Yellow
    pnputil /add-driver $DriverInf
    Write-Host ""
    Write-Host "  Driver added to store but may not be bound to device." -ForegroundColor Yellow
    Write-Host "  Open Device Manager (devmgmt.msc), find:" -ForegroundColor Yellow
    Write-Host "    'Video Controller (VGA Compatible)' under 'Other devices'" -ForegroundColor Yellow
    Write-Host "  Right-click -> Update driver -> Browse -> Let me pick ->" -ForegroundColor Yellow
    Write-Host "    'AMD Radeon RX 9070 XT (Compute)'" -ForegroundColor Yellow
}

# --- Done ---
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Done!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Verify in Device Manager (devmgmt.msc):"
Write-Host "  Look for 'AMD Radeon RX 9070 XT (Compute)'"
Write-Host "  under 'Compute Accelerator' class."
Write-Host ""
Write-Host "Then test from Python:"
Write-Host "  cd D:\R\userspace_driver\python"
Write-Host "  python test_hw_hello.py"
