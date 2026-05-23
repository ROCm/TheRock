@echo off
echo Building amdrocm-blas package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-blas.msi "%~dp0\amdrocm-blas.wxs"
if "%ERRORLEVEL%" == "0" (
    echo amdrocm-blas package built successfully
) else (
    echo Failed to build amdrocm-blas package
)
if not "%~1" == "/no_pause" (
    pause
)
