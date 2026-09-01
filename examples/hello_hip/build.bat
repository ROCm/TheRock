@echo off
setlocal

REM === Hello HIP Remote - Build Script ===
REM
REM Compiles the GPU kernel and host program entirely on Windows.
REM The kernel ISA is embedded directly in the executable.
REM
REM Requires: amdclang++ (from ROCm SDK), cl.exe (MSVC), python on PATH.

set HIP_INCLUDE=..\..\core\hip-remote-client\include
set HIP_LIB=..\..\build-hip-remote\amdhip64.lib
set GPU_ARCH=gfx942

REM --- Step 1: Compile GPU kernel to code object ---
echo [1/4] Compiling GPU kernel
where amdclang++ >nul 2>&1
if %errorlevel% neq 0 (
    echo       amdclang++ not found. Install: pip install rocm
    if not exist vector_add.co (
        echo       No existing vector_add.co found either. Cannot continue.
        exit /b 1
    )
    echo       Using existing vector_add.co
) else (
    amdclang++ --offload-arch=%GPU_ARCH% --cuda-device-only -o vector_add.co -x hip vector_add_kernel.hip
    if %errorlevel% neq 0 exit /b 1
)
echo       OK

REM --- Step 2: Embed code object as C header ---
echo [2/4] Embedding kernel ISA into header
python embed_co.py vector_add.co vector_add_co.h
if %errorlevel% neq 0 exit /b 1

REM --- Step 3: Compile host program ---
echo [3/4] Compiling host program
cl /nologo /I %HIP_INCLUDE% hello_hip.c /Fe:hello_hip.exe /link %HIP_LIB%
if %errorlevel% neq 0 exit /b 1
echo       OK

REM --- Step 4: Copy runtime DLL ---
echo [4/4] Copying amdhip64.dll
copy /Y ..\..\build-hip-remote\amdhip64.dll . >nul
echo       OK

echo.
echo Build complete. Run:
echo   set TF_WORKER_HOST=^<your-gpu-server-ip^>
echo   hello_hip.exe
