@echo off
echo Building amdrocm-llvm package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-llvm.msi "%~dp0\amdrocm-llvm.wxs"
if errorlevel 1 (
    echo Failed to build amdrocm-llvm package
    exit /b 1
)
echo amdrocm-llvm package built successfully
if not "%~1" == "/no_pause" (
    pause
)
