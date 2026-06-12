@echo off
echo Building amdrocm-rand package...
wix build -arch x64 -ext WixToolset.Dependency.wixext -ext WixToolset.UI.wixext -out amdrocm-rand.msi "%~dp0\amdrocm-rand.wxs"
if errorlevel 1 (
    echo Failed to build amdrocm-rand package
    exit /b 1
)
echo amdrocm-rand package built successfully
if not "%~1" == "/no_pause" (
    pause
)
