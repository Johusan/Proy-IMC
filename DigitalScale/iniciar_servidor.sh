#!/usr/bin/env bash
# ==============================================================================
# DIGITALSCALE AI - SCRIPT DE INICIO PARA LINUX / RASPBERRY PI 5 (AARCH64)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# Configuración opcional de puerto mediante argumento (ej: ./iniciar_servidor.sh 8080)
if [ -n "$1" ]; then
    export DIGITALSCALE_PORT="$1"
    echo "[CONFIG] Puerto personalizado recibido por argumento: $DIGITALSCALE_PORT"
fi

TARGET_PORT="${DIGITALSCALE_PORT:-8000}"

# Comprobar si el puerto ya está en uso antes de arrancar
PORT_IN_USE=false
if command -v ss &> /dev/null; then
    if ss -tulpn 2>/dev/null | grep -q ":$TARGET_PORT\b"; then
        PORT_IN_USE=true
    fi
elif command -v lsof &> /dev/null; then
    if lsof -i :"$TARGET_PORT" &> /dev/null; then
        PORT_IN_USE=true
    fi
fi

if [ "$PORT_IN_USE" = true ]; then
    echo ""

    echo "[AVISO] El puerto $TARGET_PORT ya está siendo utilizado por otro proceso:"
    if command -v ss &> /dev/null; then
        ss -tulpn 2>/dev/null | grep ":$TARGET_PORT\b" || true
    elif command -v lsof &> /dev/null; then
        lsof -i :"$TARGET_PORT" || true
    fi
    echo ""
    echo "   Opciones de solución:"
    echo "   1) Usar otro puerto libre pasando el número como argumento:"
    echo "      ./iniciar_servidor.sh 8080"
    echo "      ./iniciar_servidor.sh 8001"
    echo "   2) O liberar el puerto $TARGET_PORT matando el proceso anterior:"
    echo "      sudo fuser -k $TARGET_PORT/tcp"
    echo ""
    read -r -p "¿Desea continuar de todos modos en el puerto $TARGET_PORT? (s/N): " RESP
    if [[ ! "$RESP" =~ ^[sSyY]$ ]]; then
        echo "[CANCELADO] Por favor elija un puerto libre o libere el $TARGET_PORT."
        exit 1
    fi
fi

# Verificar estado de certificados HTTPS
if [ -f "$SCRIPT_DIR/certs/cert.pem" ] && [ -f "$SCRIPT_DIR/certs/key.pem" ]; then
    echo "[SEGURIDAD] Certificados SSL detectados: Modo HTTPS activo."
else
    echo "[AVISO] Certificados SSL no detectados (Modo HTTP activo)."
    echo "        Para habilitar la cámara desde otros dispositivos en red, ejecute: ./generar_certificados.sh"
fi

echo "[INFO] Iniciando servidor FastAPI en DigitalScale (Puerto: $TARGET_PORT)..."
echo "       Presione Ctrl + C para detener el servidor."
echo ""

exec "$PYTHON_CMD" server.py
