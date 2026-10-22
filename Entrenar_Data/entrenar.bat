@echo off
title DigitalScale AI - Reentrenamiento de Precision RTX 3050
color 0A

echo ======================================================================
echo           DIGITALSCALE AI - PIPELINE DE REENTRENAMIENTO (RTX 3050)
echo ======================================================================
echo.

cd /d "%~dp0"

echo [PASO 1/3] Preprocesamiento anatomico con MediaPipe (Recorte de Torso)...
python preprocess_dataset.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo el preprocesamiento anatomico.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [PASO 2/3] Entrenando SE-DenseNet-121 (35 epocas en GPU RTX 3050)...
python train_rtx3050.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo el entrenamiento.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [PASO 3/3] Exportando pesos calibrados al servidor de DigitalScale...
python export_to_digitalscale.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo la exportacion de pesos.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo   PROCESO COMPLETADO: El modelo de alta precision ya esta en DigitalScale!
echo ======================================================================
pause
