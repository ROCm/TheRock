@echo off
echo Building amdrocm-dnn-dev package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-dnn-dev.msi "%~dp0\amdrocm-dnn-dev.wxs"
if %ERRORLEVEL% equ 0 (
    echo amdrocm-dnn-dev package built successfully
) else (
    echo Failed to build amdrocm-dnn-dev package
    exit /b 1
)
if not "%~1" == "/no_pause" (
    pause
)
