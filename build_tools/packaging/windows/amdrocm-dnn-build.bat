@echo off
echo Building amdrocm-dnn package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-dnn.msi "%~dp0\amdrocm-dnn.wxs"
if errorlevel 1 (
    echo Failed to build amdrocm-dnn package
    exit /b 1
)
echo amdrocm-dnn package built successfully
if not "%~1" == "/no_pause" (
    pause
)
