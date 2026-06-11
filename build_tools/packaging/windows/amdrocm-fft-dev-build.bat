@echo off
echo Building amdrocm-fft-dev package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-fft-dev.msi "%~dp0\amdrocm-fft-dev.wxs"
if %ERRORLEVEL% equ 0 (
    echo amdrocm-fft-dev package built successfully
) else (
    echo Failed to build amdrocm-fft-dev package
    exit /b 1
)
if not "%~1" == "/no_pause" (
    pause
)
