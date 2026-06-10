@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM docs/visuals/render.bat
REM 一键把所有 mermaid .mmd 源文件渲染成 png
REM 依赖：npm install -g @mermaid-js/mermaid-cli
REM =====================================================

cd /d "%~dp0"

where mmdc >nul 2>&1
if errorlevel 1 (
    echo [ERROR] mmdc not found on PATH.
    echo.
    echo Install via:
    echo   npm install -g @mermaid-js/mermaid-cli
    echo.
    echo Or render manually online: https://mermaid.live/
    exit /b 1
)

echo [INFO] mmdc found, starting batch render...
echo.

set count=0
set fail=0

for /R mermaid %%F in (*.mmd) do (
    set "src=%%F"
    set "dst=!src:\mermaid\=\png\!"
    set "dst=!dst:.mmd=.png!"

    REM 确保输出目录存在
    for %%D in ("!dst!") do set "outdir=%%~dpD"
    if not exist "!outdir!" mkdir "!outdir!"

    echo [RENDER] %%~nxF
    mmdc -i "%%F" -o "!dst!" -t dark -b transparent -w 2000 --quiet
    if errorlevel 1 (
        echo   [FAIL] %%~nxF
        set /a fail+=1
    ) else (
        set /a count+=1
    )
)

echo.
echo =====================================================
echo [DONE] rendered !count! files, !fail! failed
echo =====================================================

if !fail! gtr 0 exit /b 2
exit /b 0
