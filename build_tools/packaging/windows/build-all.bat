@echo off
echo Building all available MSI packages...

if not "%~1"=="" (
  pushd "%~1" || exit /b 1
)

for /f %%I in ('dir /b "%~dp0*-build.bat"') do (
  call "%~dp0%%I" /no_pause
  if errorlevel 1 exit /b 1
)

echo Finished

if /i not "%~2"=="/no_pause" (
  pause
)

if not "%~1"=="" (
  popd
)

@echo on
