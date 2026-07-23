@echo off
setlocal enabledelayedexpansion
echo Building all available MSI packages...
if not "%~1" == "" (
    pushd "%~1" || exit /b 1
    echo Changed directory to %~1
)
for /f %%I IN ('dir /b "%~dp0*-build.bat"') do (
  call "%~dp0%%I" /no_pause
  if !ERRORLEVEL! neq 0 (
    rem popd
    rem echo Restored directory
    rem exit /b 1
  )
)
if not "%~1" == "" (
    popd
    echo Restored directory
)
echo Finished
if not "%~2" == "/no_pause" (
    pause
)
@echo on
exit /b 0
