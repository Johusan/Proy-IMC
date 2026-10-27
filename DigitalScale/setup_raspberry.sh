#!/usr/bin/env bash
# ==============================================================================
# DIGITALSCALE AI - INSTALADOR AUTOMATIZADO PARA RASPBERRY PI 5 (AARCH64)
# Compatible con Debian Linux 12/13 / Raspberry Pi OS 64-bit
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================================================="
echo "    INSTALADOR DE DIGITALSCALE AI PARA RASPBERRY PI 5 (AARCH64)"
echo "==================================================================="
echo ""

# 1. Comprobar arquitectura aarch64
ARCH=$(uname -m)
echo "[1/5] Arquitectura detectada: $ARCH"
if [ "$ARCH" != "aarch64" ]; then
    echo "[AVISO] La arquitectura detectada no es aarch64 (ARM64). Podrían requerirse paquetes específicos."
fi

# 2. Instalar dependencias del sistema operativo (Debian/Ubuntu/RPi OS)
echo ""
echo "[2/5] Instalando dependencias del sistema vía apt..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
        git
else
    echo "[AVISO] apt-get no encontrado. Asegúrese de tener instalados libgl1, libglib2.0-0 y libgomp1."
fi

# 3. Crear entorno virtual gestionado (PEP 668 compliant)
echo ""
echo "[3/5] Creando entorno virtual en $SCRIPT_DIR/venv..."
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    python3 -m venv "$SCRIPT_DIR/venv"
    echo "[OK] Entorno virtual creado exitosamente."
else
    echo "[INFO] Entorno virtual ya existente en venv/."
fi

# shellcheck disable=SC1090
source "$SCRIPT_DIR/venv/bin/activate"

# 4. Actualizar pip e instalar requerimientos
echo ""
echo "[4/5] Instalando dependencias de Python (PyTorch CPU, MediaPipe, FastAPI)..."
# Limpiar caché previa de pip para liberar espacio en disco
pip cache purge 2>/dev/null || true

pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar PyTorch y torchvision versión CPU-only 
echo ""
echo "[INFO] Instalando PyTorch CPU y Torchvision para ARM64 (sin paquetes NVIDIA)..."
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias del servidor
REQUIREMENTS_FILE="$SCRIPT_DIR/backend/requirements-server.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo ""
    echo "[INFO] Instalando dependencias del servidor FastAPI..."
    pip install --no-cache-dir -r "$REQUIREMENTS_FILE"
else
    echo "[ERROR] No se encontró $REQUIREMENTS_FILE"
    exit 1
fi

# 5. Verificación de modelos y pesos
echo ""
echo "[5/5] Verificando modelos y pesos..."

WEIGHTS_DIR="$SCRIPT_DIR/backend/weights"
mkdir -p "$WEIGHTS_DIR"

if [ -f "$WEIGHTS_DIR/best_model.ckpt" ]; then
    echo "[OK] Pesos del modelo encontrados en $WEIGHTS_DIR/best_model.ckpt"
else
    echo "-------------------------------------------------------------------"
    echo "[IMPORTANTE] Falta transferir 'best_model.ckpt' al Raspberry Pi."
    echo "Ruta de destino: $WEIGHTS_DIR/best_model.ckpt"
    echo "Comando para transferir desde tu PC:"
    echo "  scp DigitalScale/backend/weights/best_model.ckpt <usuario>@<ip_raspberry>:$WEIGHTS_DIR/"
    echo "-------------------------------------------------------------------"
fi

# Probar importaciones críticas de PyTorch y MediaPipe
echo ""
echo "Verificando módulos en entorno Python ARM64..."
python3 -c "
import torch, mediapipe, cv2, fastapi
print('  * PyTorch versión:', torch.__version__)
print('  * MediaPipe versión:', mediapipe.__version__)
print('  * OpenCV versión:', cv2.__version__)
print('  * FastAPI versión:', fastapi.__version__)
print('  * Hilos de CPU PyTorch:', torch.get_num_threads())
"

chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

echo ""
echo "==================================================================="
echo "  INSTALACIÓN COMPLETADA EXITOSAMENTE EN RASPBERRY PI 5!"
echo "  1. Para iniciar el servidor ejecuta:"
echo "     ./iniciar_servidor.sh"
echo "     (Los certificados HTTPS para la cámara WebRTC se autogeneran"
echo "      y validan automáticamente con la IP local de tu Raspberry Pi)"
echo "==================================================================="
