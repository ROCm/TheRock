@echo off
echo Building amdrocm-fft package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-fft.msi "%~dp0\amdrocm-fft.wxs"
if "%ERRORLEVEL%" == "0" (
    echo amdrocm-fft package built successfully
) else (
    echo Failed to build amdrocm-fft package
)
if not "%~1" == "/no_pause" (
    pause
)
