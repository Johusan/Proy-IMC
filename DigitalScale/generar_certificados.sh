#!/usr/bin/env bash
# ==============================================================================
# DIGITALSCALE AI - GENERADOR DE CERTIFICADOS SSL (HTTPS) PARA RASPBERRY PI
# Genera certificados autofirmados con SAN dinámico para habilitar la cámara
# WebRTC en redes locales desde otros dispositivos (laptops, celulares, tablets).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CERTS_DIR="$SCRIPT_DIR/certs"
mkdir -p "$CERTS_DIR"

echo "        DIGITALSCALE AI - GENERADOR DE CERTIFICADOS HTTPS""

# 1. Verificar disponibilidad de OpenSSL
if ! command -v openssl &> /dev/null; then
    echo "[ERROR] 'openssl' no está instalado en este sistema."
    echo "        Instálelo con: sudo apt-get update && sudo apt-get install -y openssl"
    exit 1
fi

# 2. Detección dinámica de IPs de red local
DETECTED_IPS=()

# Si el usuario pasa una IP específica como argumento, priorizarla
if [ -n "$1" ]; then
    DETECTED_IPS+=("$1")
    echo "[CONFIG] IP personalizada recibida por argumento: $1"
fi

# Detección automática mediante hostname -I
if command -v hostname &> /dev/null; then
    for ip in $(hostname -I 2>/dev/null); do
        if [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && [ "$ip" != "127.0.0.1" ]; then
            DETECTED_IPS+=("$ip")
        fi
    done
fi

# Respaldo con ip route si no se detectó ninguna IP
if [ ${#DETECTED_IPS[@]} -eq 0 ] && command -v ip &> /dev/null; then
    FALLBACK_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7}')
    if [ -n "$FALLBACK_IP" ]; then
        DETECTED_IPS+=("$FALLBACK_IP")
    fi
fi

# 3. Construir Subject Alternative Names (SAN)
HOST_NAME=$(hostname 2>/dev/null || echo "raspberrypi")
SAN_LIST="IP:127.0.0.1,DNS:localhost,DNS:${HOST_NAME}.local,DNS:${HOST_NAME},DNS:raspberrypi.local"

# Añadir cada IP única detectada al SAN
declare -A SEEN_IPS
for ip in "${DETECTED_IPS[@]}"; do
    if [ -z "${SEEN_IPS[$ip]}" ]; then
        SEEN_IPS[$ip]=1
        SAN_LIST="${SAN_LIST},IP:${ip}"
    fi
done

echo "[INFO] Nombres y direcciones autorizadas en el certificado:"
echo "       $SAN_LIST"
echo ""

# 4. Generar certificado autofirmado con OpenSSL
KEY_FILE="$CERTS_DIR/key.pem"
CERT_FILE="$CERTS_DIR/cert.pem"

echo "[INFO] Generando clave privada RSA y certificado X.509 (825 días de validez)..."
openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -days 825 \
    -subj "/C=ES/ST=Local/L=Local/O=DigitalScaleAI/CN=${HOST_NAME}" \
    -addext "subjectAltName=${SAN_LIST}" \
    2>/dev/null

chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

echo ""
echo "==================================================================="
echo " ¡CERTIFICADOS HTTPS GENERADOS CON ÉXITO!"
echo "==================================================================="
echo "Archivos creados en:"
echo "  -> Clave privada:      $KEY_FILE"
echo "  -> Certificado público: $CERT_FILE"
echo ""
echo "Instrucciones de uso:"
echo "1. Inicie el servidor con: ./iniciar_servidor.sh"
echo "2. DigitalScale detectará automáticamente los certificados y activará HTTPS."
echo "3. Abra en el navegador de su equipo cliente:"
for ip in "${!SEEN_IPS[@]}"; do
    echo "    https://${ip}:8000"
done
echo "   https://localhost:8000"
echo ""
echo " NOTA DEL NAVEGADOR:"
echo "Al ser un certificado autofirmado, la primera vez el navegador mostrará:"
echo "  'La conexión no es privada' o 'Advertencia de seguridad'."
echo "Haga clic en 'Configuración Avanzada' -> 'Continuar a ${DETECTED_IPS[0]:-la IP} (no seguro)'."
echo "Tras ese clic, el navegador concederá acceso pleno y SEGURO a la CÁMARA WEBRTC."
echo "==================================================================="
