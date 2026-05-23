@echo off
echo Building all available MSI packages...
if not "%~1" == "" (
rem Changing directory
    %~d1
    cd "%~dp1"
)
for /f %%I IN ('dir /b "%~dp0*-build.bat"') do (
  call "%~dp0%%I" /no_pause
)
echo Finished
pause
@echo on
