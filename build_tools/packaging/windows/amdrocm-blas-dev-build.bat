@echo off
echo Building amdrocm-blas-dev package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-blas-dev.msi "%~dp0\amdrocm-blas-dev.wxs"
if %ERRORLEVEL% equ 0 (
    echo amdrocm-blas-dev package built successfully
) else (
    echo Failed to build amdrocm-blas-dev package
    exit /b 1
)
if not "%~1" == "/no_pause" (
    pause
)
