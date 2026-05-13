@echo off
rem Local helper: invoke any command inside an MSVC x64 dev-env shell.
rem Usage:   build_tools\msvc_shell.cmd <command and args...>
setlocal enableextensions
rem Make winget-installed tools visible (ccache, ninja, etc.) and the .venv
set "PATH=%~dp0..\.venv\Scripts;%LOCALAPPDATA%\Microsoft\WinGet\Links;C:\Program Files\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\Installer;C:\Strawberry\perl\bin;C:\Strawberry\c\bin;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo Failed to activate MSVC x64 environment 1>&2
  exit /b 1
)
%*
