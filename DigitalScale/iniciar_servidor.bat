@echo off
setlocal enabledelayedexpansion
title DigitalScale AI - Servidor

cd /d "%~dp0"

echo ===================================================================
echo               DIGITALSCALE AI - SERVIDOR LOCAL
echo           Estimacion Biometrica de IMC por Vision Artificial
echo ===================================================================
echo.

set "VENV_ACTIVATED=0"
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Entorno virtual detectado en .venv
    call ".venv\Scripts\activate.bat"
    set "VENV_ACTIVATED=1"
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Entorno virtual detectado en venv
    call "venv\Scripts\activate.bat"
    set "VENV_ACTIVATED=1"
) else if exist "backend\.venv\Scripts\activate.bat" (
    echo [INFO] Entorno virtual detectado en backend\.venv
    call "backend\.venv\Scripts\activate.bat"
    set "VENV_ACTIVATED=1"
) else if exist "backend\venv\Scripts\activate.bat" (
    echo [INFO] Entorno virtual detectado en backend\venv
    call "backend\venv\Scripts\activate.bat"
    set "VENV_ACTIVATED=1"
)

set "PYTHON_EXE="

if "!VENV_ACTIVATED!"=="1" (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    where py >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py"
    )
)

if not defined PYTHON_EXE (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    )
)

if not defined PYTHON_EXE (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
        if exist "%%D\python.exe" (
            set "PYTHON_EXE=%%D\python.exe"
        )
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] No se pudo encontrar el ejecutable de Python en el sistema.
    echo Por favor asegurese de tener Python 3.10 o superior instalado.
    echo.
    pause
    exit /b 1
)

echo [INFO] Python detectado: !PYTHON_EXE!
echo [INFO] Iniciando servidor web...
echo        (El navegador se abrira automaticamente en cuanto el servidor este listo)
echo        (Presione Ctrl + C para detener el servidor)
echo.

"!PYTHON_EXE!" server.py

if %errorlevel% neq 0 (
    echo.
    echo [AVISO] El servidor se ha detenido con codigo de salida %errorlevel%.
    pause
)
