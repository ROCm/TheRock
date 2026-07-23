@echo off
echo Building amdrocm-fft package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-fft.msi "%~dp0\amdrocm-fft.wxs"
if %ERRORLEVEL% equ 0 (
    echo amdrocm-fft package built successfully
) else (
    echo Failed to build amdrocm-fft package
    exit /b 1
)
if not "%~1" == "/no_pause" (
    pause
)
