# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

<#
.SYNOPSIS
    Quick start script for TheRock GTest samples with logging demo
    
.DESCRIPTION
    This script automates the setup and execution of GTest samples with
    TheRock's logging framework. It handles environment setup, building,
    and test execution.
    
.PARAMETER BuildOnly
    Only build tests without running them
    
.PARAMETER Clean
    Clean build directory before building
    
.PARAMETER DryRun
    Show what would be executed without actually running
    
.PARAMETER NoBuild
    Skip build step and only run tests
    
.EXAMPLE
    .\quick_start.ps1
    Build and run all tests
    
.EXAMPLE
    .\quick_start.ps1 -BuildOnly
    Only build tests
    
.EXAMPLE
    .\quick_start.ps1 -Clean
    Clean and rebuild
#>

param(
    [switch]$BuildOnly,
    [switch]$Clean,
    [switch]$DryRun,
    [switch]$NoBuild
)

# Colors for output
function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

Write-Step "TheRock GTest Samples - Quick Start"

# Change to project root
Set-Location $ProjectRoot
Write-Info "Project root: $ProjectRoot"

# Check Python
Write-Step "Checking Prerequisites"

$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    Write-Err "Python not found! Please install Python 3.9 or higher."
    exit 1
}

$PythonVersion = & python --version 2>&1
Write-Info "Python: $PythonVersion"

# Check CMake
$CMakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $CMakeCmd) {
    Write-Err "CMake not found! Please install CMake 3.14 or higher."
    exit 1
}

$CMakeVersion = & cmake --version | Select-Object -First 1
Write-Info "CMake: $CMakeVersion"

# Check for PyYAML
Write-Info "Checking Python dependencies..."
$PyYAMLCheck = & python -c "import yaml; print('PyYAML installed')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warn "PyYAML not found. Installing..."
    & python -m pip install PyYAML
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to install PyYAML"
        exit 1
    }
    Write-Info "PyYAML installed successfully"
} else {
    Write-Info "PyYAML: OK"
}

# Clean if requested
if ($Clean) {
    Write-Step "Cleaning Build Directory"
    if (Test-Path "build") {
        Write-Info "Removing build directory..."
        Remove-Item -Recurse -Force build
    }
    if (Test-Path "logs") {
        Write-Info "Removing logs directory..."
        Remove-Item -Recurse -Force logs
    }
    if (Test-Path "test_results") {
        Write-Info "Removing test_results directory..."
        Remove-Item -Recurse -Force test_results
    }
    Write-Info "Clean complete"
}

# Build command arguments
$RunArgs = @("build_tools\run_logging_demo.py")
$RunArgs += "--config", "build_tools\logging_demo.yaml"

if ($BuildOnly) {
    $RunArgs += "--build-only"
}

if ($DryRun) {
    $RunArgs += "--dry-run"
}

# Run the test runner
Write-Step "Running Test Suite"

if ($DryRun) {
    Write-Warn "DRY RUN MODE - No actual execution"
}

Write-Info "Command: python $($RunArgs -join ' ')"
Write-Host ""

& python $RunArgs

$ExitCode = $LASTEXITCODE

# Summary
Write-Host ""
Write-Step "Execution Complete"

if ($ExitCode -eq 0) {
    Write-Info "✅ SUCCESS: All operations completed successfully!"
    
    if (-not $BuildOnly -and -not $DryRun -and -not $NoBuild) {
        Write-Host ""
        Write-Info "Test Results:"
        Write-Info "  - Logs: logs\logging_demo.log"
        if (Test-Path "test_results\summary.json") {
            Write-Info "  - Summary: test_results\summary.json"
        }
        if (Test-Path "build\test_results") {
            $XmlFiles = Get-ChildItem -Path "build\test_results" -Filter "*.xml" -ErrorAction SilentlyContinue
            if ($XmlFiles) {
                Write-Info "  - XML Reports: build\test_results\*.xml ($($XmlFiles.Count) files)"
            }
        }
    }
} else {
    Write-Err "❌ FAILED: Some operations failed (exit code: $ExitCode)"
    Write-Host ""
    Write-Warn "Troubleshooting:"
    Write-Host "  1. Check logs\logging_demo.log for detailed error messages"
    Write-Host "  2. Try cleaning and rebuilding: .\quick_start.ps1 -Clean"
    Write-Host "  3. Run in dry-run mode to check configuration: .\quick_start.ps1 -DryRun"
    Write-Host "  4. Check CMake configuration in build directory"
}

Write-Host ""
Write-Info "For more options, see: tests\gtest_samples\README.md"

exit $ExitCode


