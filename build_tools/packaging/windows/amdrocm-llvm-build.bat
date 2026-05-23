@echo off
echo Building amdrocm-llvm package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-llvm.msi "%~dp0\amdrocm-llvm.wxs"
if "%ERRORLEVEL%" == "0" (
    echo amdrocm-llvm package built successfully
) else (
    echo Failed to build amdrocm-llvm package
)
if not "%~1" == "/no_pause" (
    pause
)
