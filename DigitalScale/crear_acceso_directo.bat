@echo off
title Crear Acceso Directo - DigitalScale AI
cd /d "%~dp0"

echo ===================================================================
echo       DIGITALSCALE AI - CREADOR DE ACCESO DIRECTO (WINDOWS)
echo ===================================================================
echo.

set "SCRIPT_DIR=%~dp0"
set "TARGET_BAT=%SCRIPT_DIR%iniciar_servidor.bat"

powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $desktop = [Environment]::GetFolderPath('Desktop'); $shortcut = $ws.CreateShortcut((Join-Path $desktop 'DigitalScale AI.lnk')); $shortcut.TargetPath = '%TARGET_BAT%'; $shortcut.WorkingDirectory = '%SCRIPT_DIR%'; $shortcut.IconLocation = 'shell32.dll,13'; $shortcut.Description = 'DigitalScale AI - Estimacion Biometrica de IMC'; $shortcut.Save()"

if %errorlevel% equ 0 (
    echo [OK] Acceso directo 'DigitalScale AI' creado con exito en tu Escritorio!
    echo.
    echo Ahora puedes hacer doble clic en el acceso directo del Escritorio
    echo para iniciar DigitalScale AI directamente.
) else (
    echo [ERROR] No se pudo crear el acceso directo automaticamente.
    echo Puedes hacer clic derecho en 'iniciar_servidor.bat' ^> Enviar a ^> Escritorio.
)

echo.
pause
