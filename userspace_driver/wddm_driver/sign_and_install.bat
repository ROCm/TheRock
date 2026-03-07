@echo off
REM ============================================================
REM sign_and_install.bat - Sign and install the AMDGPU WDDM driver
REM
REM Must be run from an ELEVATED (admin) command prompt.
REM ============================================================

setlocal

REM --- Configuration ---
set DRIVER_DIR=%~dp0x64\Release
set DRIVER_SYS=%DRIVER_DIR%\amdgpu_wddm.sys
set DRIVER_INF=%DRIVER_DIR%\amdgpu_wddm.inf
set CERT_NAME=AMDGPU Test
set CERT_STORE=PrivateCertStore

REM WDK tools
set WDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64
set SIGNTOOL="%WDK_BIN%\signtool.exe"
set MAKECAT="%WDK_BIN%\makecat.exe"
set DEVCON="C:\Program Files (x86)\Windows Kits\10\Tools\10.0.26100.0\x64\devcon.exe"

REM --- Check admin ---
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator.
    exit /b 1
)

REM --- Check driver exists ---
if not exist "%DRIVER_SYS%" (
    echo ERROR: Driver not found at %DRIVER_SYS%
    echo Build first: msbuild amdgpu_wddm.vcxproj /p:Configuration=Release /p:Platform=x64
    exit /b 1
)

echo ============================================================
echo AMD GPU WDDM Display Driver - Sign and Install
echo ============================================================
echo.

REM --- Step 1: Prepare package directory ---
echo [1/5] Preparing driver package...
set PKG_DIR=%~dp0package
if not exist "%PKG_DIR%" mkdir "%PKG_DIR%"
copy /y "%DRIVER_SYS%" "%PKG_DIR%\amdgpu_wddm.sys" >nul
copy /y "%~dp0amdgpu_wddm.inf" "%PKG_DIR%\amdgpu_wddm.inf" >nul

REM --- Step 2: Create catalog file ---
echo [2/5] Creating catalog...
echo [CatalogHeader] > "%PKG_DIR%\amdgpu_wddm.cdf"
echo Name=amdgpu_wddm.cat >> "%PKG_DIR%\amdgpu_wddm.cdf"
echo ResultDir=%PKG_DIR% >> "%PKG_DIR%\amdgpu_wddm.cdf"
echo PublicVersion=0x00000001 >> "%PKG_DIR%\amdgpu_wddm.cdf"
echo CatalogVersion=2 >> "%PKG_DIR%\amdgpu_wddm.cdf"
echo HashAlgorithms=SHA256 >> "%PKG_DIR%\amdgpu_wddm.cdf"
echo PageHashes=false >> "%PKG_DIR%\amdgpu_wddm.cdf"
echo. >> "%PKG_DIR%\amdgpu_wddm.cdf"
echo [CatalogFiles] >> "%PKG_DIR%\amdgpu_wddm.cdf"
echo amdgpu_wddm.inf=%PKG_DIR%\amdgpu_wddm.inf >> "%PKG_DIR%\amdgpu_wddm.cdf"
echo amdgpu_wddm.sys=%PKG_DIR%\amdgpu_wddm.sys >> "%PKG_DIR%\amdgpu_wddm.cdf"

%MAKECAT% "%PKG_DIR%\amdgpu_wddm.cdf"
if %errorlevel% neq 0 (
    echo WARNING: makecat failed. Continuing...
)

REM --- Step 3: Sign the .sys and .cat ---
echo [3/5] Signing driver...
%SIGNTOOL% sign /s %CERT_STORE% /n "%CERT_NAME%" /fd sha256 /v "%PKG_DIR%\amdgpu_wddm.sys"
%SIGNTOOL% sign /s %CERT_STORE% /n "%CERT_NAME%" /fd sha256 /v "%PKG_DIR%\amdgpu_wddm.cat"

REM --- Step 4: Install ---
echo.
echo [4/5] Installing WDDM display driver...
echo This will replace the current display driver for AMD RX 9070 XT.
echo.

REM Use devcon updateni to force install without user prompt
%DEVCON% updateni "%PKG_DIR%\amdgpu_wddm.inf" "PCI\VEN_1002&DEV_7551"
if %errorlevel% neq 0 (
    echo.
    echo devcon updateni failed. Trying pnputil...
    pnputil /add-driver "%PKG_DIR%\amdgpu_wddm.inf" /install
)

REM --- Step 5: Verify ---
echo.
echo [5/5] Verifying...
echo Check Device Manager for "AMD Radeon RX 9070 XT (WDDM Display)"
echo under "Display adapters".
echo.
echo Registry check:
reg query HKLM\SOFTWARE\AmdGpuWddm 2>nul
echo.
echo ============================================================
echo Done.
echo ============================================================

endlocal
