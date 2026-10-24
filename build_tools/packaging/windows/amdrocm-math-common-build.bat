@echo off
echo Building amdrocm-math-common package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-math-common.msi "%~dp0\amdrocm-math-common.wxs"
if %ERRORLEVEL% equ 0 (
    echo amdrocm-math-common package built successfully
) else (
    echo Failed to build amdrocm-math-common package
    exit /b 1
)
if not "%~1" == "/no_pause" (
    pause
)
