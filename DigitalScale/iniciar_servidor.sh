#!/usr/bin/env bash
# ==============================================================================
# DIGITALSCALE AI - SCRIPT DE INICIO PARA LINUX / RASPBERRY PI 5 (AARCH64)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================================================="
echo "              DIGITALSCALE AI - SERVIDOR LOCAL"
echo "          Estimación Biométrica de IMC (Raspberry Pi 5 / Linux)"
echo "==================================================================="
echo ""

# Detección y activación del entorno virtual
VENV_PATH=""
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    VENV_PATH="$SCRIPT_DIR/.venv"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    VENV_PATH="$SCRIPT_DIR/venv"
elif [ -f "$SCRIPT_DIR/backend/.venv/bin/activate" ]; then
    VENV_PATH="$SCRIPT_DIR/backend/.venv"
elif [ -f "$SCRIPT_DIR/backend/venv/bin/activate" ]; then
    VENV_PATH="$SCRIPT_DIR/backend/venv"
fi

if [ -n "$VENV_PATH" ]; then
    echo "[INFO] Activando entorno virtual en: $VENV_PATH"
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    PYTHON_CMD="python3"
else
    echo "[AVISO] No se encontró entorno virtual (venv/.venv)."
    echo "        Se recomienda ejecutar './setup_raspberry.sh' para crearlo."
    PYTHON_CMD="python3"
fi

# Verificar si Python 3 está disponible
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "[ERROR] No se encontró '$PYTHON_CMD'. Instale Python 3 antes de continuar."
    exit 1
fi

# Comprobar que los pesos del modelo existan
WEIGHTS_FILE="$SCRIPT_DIR/backend/weights/best_model.ckpt"
ALT_WEIGHTS_FILE="$SCRIPT_DIR/backend/trained_models/model_5/best_model.ckpt"

if [ ! -f "$WEIGHTS_FILE" ] && [ ! -f "$ALT_WEIGHTS_FILE" ]; then
    echo ""
    echo "[ERROR CRÍTICO] No se encontró el archivo de pesos del modelo:"
    echo "  Falta: $WEIGHTS_FILE"
    echo ""
    echo "  Recuerda transferir 'best_model.ckpt' desde tu PC de entrenamiento a esa ruta."
    echo "  (Puedes usar scp, FileZilla o una memoria USB)."
    echo ""
    exit 1
fi

echo "[INFO] Iniciando servidor FastAPI en DigitalScale..."
echo "       Presione Ctrl + C para detener el servidor."
echo ""

exec "$PYTHON_CMD" server.py
