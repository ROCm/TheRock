@echo off
echo Building amdrocm-runtime package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-runtime.msi "%~dp0\amdrocm-runtime.wxs"
if errorlevel 1 (
    echo Failed to build amdrocm-runtime package
    exit /b 1
)
echo amdrocm-runtime package built successfully
if not "%~1" == "/no_pause" (
    pause
)
