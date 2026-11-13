@echo off
echo Building amdrocm-ck package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-ck.msi "%~dp0\amdrocm-ck.wxs"
if %ERRORLEVEL% equ 0 (
    echo amdrocm-ck package built successfully
) else (
    echo Failed to build amdrocm-ck package
    exit /b 1
)
if not "%~1" == "/no_pause" (
    pause
)
