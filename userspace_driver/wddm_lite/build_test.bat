@echo off
REM Build wddm_lite_test.exe using MSVC
REM Run from VS Developer Command Prompt or after calling vcvarsall.bat

set VSTOOLS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if exist %VSTOOLS% (
    call %VSTOOLS% amd64
) else (
    echo ERROR: VS Build Tools not found
    exit /b 1
)

set WDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
set WDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0

echo Building wddm_lite_test.exe...

cl.exe /nologo /EHsc /W3 /O2 ^
    /I"%WDK_INC%\shared" ^
    /I"%WDK_INC%\um" ^
    wddm_lite.cpp gpu_init.cpp wddm_lite_test.cpp ^
    /Fe:wddm_lite_test.exe ^
    /link /LIBPATH:"%WDK_LIB%\um\x64" ^
    gdi32.lib user32.lib

if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED
    exit /b 1
)

echo BUILD SUCCEEDED
dir wddm_lite_test.exe
