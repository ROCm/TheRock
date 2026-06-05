@echo off
echo Building amdrocm-runtime-dev package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-runtime-dev.msi "%~dp0\amdrocm-runtime-dev.wxs"
if errorlevel 1 (
    echo Failed to build amdrocm-runtime-dev package
    exit /b 1
)
echo amdrocm-runtime-dev package built successfully
if not "%~1" == "/no_pause" (
    pause
)
