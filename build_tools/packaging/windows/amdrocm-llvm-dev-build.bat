@echo off
echo Building amdrocm-llvm-dev package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-llvm-dev.msi "%~dp0\amdrocm-llvm-dev.wxs"
if errorlevel 1 (
    echo Failed to build amdrocm-llvm-dev package
    exit /b 1
)
echo amdrocm-llvm-dev package built successfully
if not "%~1" == "/no_pause" (
    pause
)
