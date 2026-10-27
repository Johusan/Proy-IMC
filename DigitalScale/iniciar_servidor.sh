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

# ------------------------------------------------------------------------------
# 5. AUTO-SANACIÓN Y VALIDACIÓN DE CERTIFICADOS HTTPS PARA WEBRTC
# ------------------------------------------------------------------------------
CERT_FILE="$SCRIPT_DIR/certs/cert.pem"
KEY_FILE="$SCRIPT_DIR/certs/key.pem"
REGEN_SSL=0
SSL_REASON=""

# Detectar IP local LAN actual de la Raspberry Pi
CURRENT_LAN_IP=""
if command -v hostname &> /dev/null; then
    for ip in $(hostname -I 2>/dev/null); do
        if [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && [ "$ip" != "127.0.0.1" ]; then
            CURRENT_LAN_IP="$ip"
            break
        fi
    done
fi
if [ -z "$CURRENT_LAN_IP" ] && command -v ip &> /dev/null; then
    CURRENT_LAN_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7}')
fi

if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ] || [ ! -s "$CERT_FILE" ] || [ ! -s "$KEY_FILE" ]; then
    REGEN_SSL=1
    SSL_REASON="Certificados HTTPS no encontrados o vacíos"
elif command -v openssl &> /dev/null; then
    # Verificar vigencia (al menos 24 horas restantes)
    if ! openssl x509 -in "$CERT_FILE" -noout -checkend 86400 &>/dev/null; then
        REGEN_SSL=1
        SSL_REASON="Certificado HTTPS vencido o próximo a expirar"
    elif [ -n "$CURRENT_LAN_IP" ]; then
        # Verificar si la IP local actual está en el SAN (Subject Alternative Name)
        SAN_OUTPUT=$(openssl x509 -in "$CERT_FILE" -noout -ext subjectAltName 2>/dev/null)
        if ! echo "$SAN_OUTPUT" | grep -q "$CURRENT_LAN_IP"; then
            REGEN_SSL=1
            SSL_REASON="Cambio de dirección IP detectado (IP actual: $CURRENT_LAN_IP no está en el certificado)"
        fi
    fi
fi

if [ "$REGEN_SSL" -eq 1 ]; then
    echo ""
    echo "[AUTO-SSL] $SSL_REASON."
    echo "[AUTO-SSL] Regenerando certificados HTTPS automáticamente para asegurar WebRTC..."
    if [ -f "$SCRIPT_DIR/generar_certificados.sh" ]; then
        chmod +x "$SCRIPT_DIR/generar_certificados.sh" 2>/dev/null || true
        "$SCRIPT_DIR/generar_certificados.sh" "$CURRENT_LAN_IP"
    elif [ -f "$SCRIPT_DIR/generar_certificados.py" ]; then
        "$PYTHON_CMD" "$SCRIPT_DIR/generar_certificados.py"
    fi
    echo ""
fi

if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "[SEGURIDAD] Modo HTTPS seguro activo (Certificado validado para la IP actual)."
else
    echo "[AVISO] Servidor en Modo HTTP no seguro (la cámara puede estar restringida en clientes remotos)."
fi

echo "[INFO] Iniciando servidor FastAPI en DigitalScale (Puerto: $TARGET_PORT)..."
echo "       Presione Ctrl + C para detener el servidor."
echo ""

exec "$PYTHON_CMD" server.py

