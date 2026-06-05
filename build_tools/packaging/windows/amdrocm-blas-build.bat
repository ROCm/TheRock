@echo off
echo Building amdrocm-blas package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-blas.msi "%~dp0\amdrocm-blas.wxs"
if errorlevel 1 (
    echo Failed to build amdrocm-blas package
    exit /b 1
)
echo amdrocm-blas package built successfully
if not "%~1" == "/no_pause" (
    pause
)
