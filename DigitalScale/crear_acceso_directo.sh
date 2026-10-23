#!/usr/bin/env bash
# ==============================================================================
# DIGITALSCALE AI - CREADOR DE ACCESO DIRECTO EN EL ESCRITORIO
# Compatible con Raspberry Pi OS / Debian Linux (LXDE, Wayfire, Labwc)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detectar carpeta de escritorio (Desktop o Escritorio)
DESKTOP_DIR="$HOME/Desktop"
if [ -d "$HOME/Escritorio" ]; then
    DESKTOP_DIR="$HOME/Escritorio"
elif [ ! -d "$DESKTOP_DIR" ]; then
    mkdir -p "$DESKTOP_DIR"
fi

SHORTCUT_PATH="$DESKTOP_DIR/digitalscale.desktop"
MENU_DIR="$HOME/.local/share/applications"
mkdir -p "$MENU_DIR"

echo "Creando acceso directo en: $SHORTCUT_PATH..."

cat << EOF > "$SHORTCUT_PATH"
[Desktop Entry]
Type=Application
Name=DigitalScale AI
Comment=Estimación Biométrica de IMC con Visión Artificial
Exec=$SCRIPT_DIR/iniciar_servidor.sh
Icon=camera-web
Terminal=true
Categories=Utility;Science;Education;
StartupNotify=true
Path=$SCRIPT_DIR
EOF

# Asignar permisos de ejecución
chmod +x "$SHORTCUT_PATH"

# Copiar también al menú de aplicaciones del sistema
cp "$SHORTCUT_PATH" "$MENU_DIR/digitalscale.desktop"

# Para Raspberry Pi OS con PCManFM / Wayfire (marcar como de confianza si gio está disponible)
if command -v gio &> /dev/null; then
    gio set "$SHORTCUT_PATH" metadata::trusted true 2>/dev/null || true
fi

echo ""
echo "==================================================================="
echo "  [OK] Acceso directo 'DigitalScale AI' creado con éxito!"
echo "  Ubicación: $SHORTCUT_PATH"
echo "  Ahora puedes hacer doble clic en el icono del escritorio para"
echo "  iniciar el sistema automáticamente."
echo "==================================================================="
