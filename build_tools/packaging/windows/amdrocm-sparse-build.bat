@echo off
echo Building amdrocm-sparse package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-sparse.msi "%~dp0\amdrocm-sparse.wxs"
if errorlevel 1 (
    echo Failed to build amdrocm-sparse package
    exit /b 1
)
echo amdrocm-sparse package built successfully
if not "%~1" == "/no_pause" (
    pause
)
