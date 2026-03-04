@echo off
REM ============================================================
REM sign_and_install.bat — Sign and install the AMDGPU MCDM driver
REM
REM Must be run from an ELEVATED (admin) command prompt.
REM ============================================================

setlocal

REM --- Configuration ---
set DRIVER_DIR=%~dp0x64\Release
set DRIVER_SYS=%DRIVER_DIR%\amdgpu_mcdm.sys
set DRIVER_INF=%DRIVER_DIR%\amdgpu_mcdm.inf
set CERT_NAME=AMDGPU Test
set CERT_STORE=PrivateCertStore
set CERT_FILE=%~dp0AmdGpuTest.cer

REM WDK tools
set WDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64
set SIGNTOOL="%WDK_BIN%\signtool.exe"
set MAKECERT="%WDK_BIN%\makecert.exe"
set INF2CAT="C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86\Inf2Cat.exe"

REM --- Check admin ---
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator.
    echo Right-click Command Prompt -^> "Run as administrator"
    exit /b 1
)

REM --- Check driver exists ---
if not exist "%DRIVER_SYS%" (
    echo ERROR: Driver not found at %DRIVER_SYS%
    echo Build the driver first: msbuild amdgpu_mcdm.vcxproj /p:Configuration=Release /p:Platform=x64
    exit /b 1
)

echo ============================================================
echo AMD GPU MCDM Driver — Sign and Install
echo ============================================================
echo.

REM --- Step 1: Check/enable test signing ---
echo [1/5] Checking test signing...
bcdedit /enum {current} | findstr /i "testsigning.*Yes" >nul 2>&1
if %errorlevel% neq 0 (
    echo   Test signing is NOT enabled. Enabling now...
    bcdedit -set testsigning on
    if %errorlevel% neq 0 (
        echo   ERROR: Failed to enable test signing.
        exit /b 1
    )
    echo   Test signing enabled. A REBOOT IS REQUIRED before the driver will load.
    echo   After rebooting, run this script again.
    echo.
    set NEED_REBOOT=1
) else (
    echo   Test signing already enabled.
)

REM --- Step 2: Create test certificate ---
echo.
echo [2/5] Creating test certificate...
if exist "%CERT_FILE%" (
    echo   Certificate file already exists: %CERT_FILE%
    echo   Skipping creation. Delete it to regenerate.
) else (
    %MAKECERT% -r -pe -ss %CERT_STORE% -n "CN=%CERT_NAME%" "%CERT_FILE%"
    if %errorlevel% neq 0 (
        echo   ERROR: makecert failed.
        exit /b 1
    )
    echo   Created: %CERT_FILE%

    REM Install cert to Trusted Root
    certutil -addstore Root "%CERT_FILE%"
    if %errorlevel% neq 0 (
        echo   WARNING: Could not add cert to Trusted Root store.
        echo   Driver may not load without trusted root cert.
    ) else (
        echo   Added to Trusted Root CA store.
    )
)

REM --- Step 3: Sign the driver ---
echo.
echo [3/5] Signing driver...
%SIGNTOOL% sign /s %CERT_STORE% /n "%CERT_NAME%" /fd sha256 /v "%DRIVER_SYS%"
if %errorlevel% neq 0 (
    echo   ERROR: signtool failed. Check certificate store.
    exit /b 1
)
echo   Driver signed successfully.

REM --- Step 4: Copy INF alongside SYS ---
echo.
echo [4/5] Preparing driver package...
copy /y "%~dp0amdgpu_mcdm.inf" "%DRIVER_DIR%\amdgpu_mcdm.inf" >nul

REM --- Step 5: Install via pnputil ---
echo.
echo [5/5] Installing driver...
if defined NEED_REBOOT (
    echo   SKIPPING install — reboot required first for test signing.
    echo   After reboot, run: pnputil /add-driver "%DRIVER_INF%" /install
    goto :done
)

pnputil /add-driver "%DRIVER_INF%" /install
if %errorlevel% neq 0 (
    echo.
    echo   pnputil failed. Trying alternative approach...
    echo   Adding driver to store first...
    pnputil /add-driver "%DRIVER_INF%"
    echo.
    echo   Now try Device Manager to manually assign the driver to:
    echo     PCI\VEN_1002^&DEV_7551
)

:done
echo.
echo ============================================================
echo Done. Check Device Manager for "AMD Radeon RX 9070 XT (Compute)"
echo under "Compute Accelerator" class.
echo.
echo To verify: devmgmt.msc
echo To check status: pnputil /enum-drivers
echo ============================================================

endlocal
