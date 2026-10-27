"""
DIGITALSCALE AI - GENERADOR Y VALIDADOR DE CERTIFICADOS SSL (AUTO-HEALING)
Permite generar cert.pem y key.pem tanto en Linux (Raspberry Pi) como en Windows.
Valida automáticamente si la IP cambió o el certificado expiró, regenerándolo de inmediato.
"""

import os
import sys
import socket
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CERTS_DIR = os.path.join(SCRIPT_DIR, "certs")
KEY_FILE = os.path.join(CERTS_DIR, "key.pem")
CERT_FILE = os.path.join(CERTS_DIR, "cert.pem")


def get_local_ips():
    """Detecta todas las IPs locales asociadas al equipo (evitando solo 127.0.0.1)."""
    ips = set(["127.0.0.1"])
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        detected = s.getsockname()[0]
        if detected:
            ips.add(detected)
        s.close()
    except Exception:
        pass
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    return [ip for ip in ips if ip]


def is_cert_valid():
    """
    Verifica si el certificado existente:
    1. Existe y no está vacío.
    2. Sigue vigente (al menos 24 horas restantes).
    3. Contiene en su SAN (Subject Alternative Name) la IP LAN actual del equipo.
    """
    if not os.path.exists(KEY_FILE) or not os.path.exists(CERT_FILE):
        return False, "Certificados inexistentes"

    if os.path.getsize(KEY_FILE) == 0 or os.path.getsize(CERT_FILE) == 0:
        return False, "Archivos de certificado vacíos"

    # 1. Comprobar vigencia con OpenSSL
    try:
        check_exp = subprocess.run(
            ["openssl", "x509", "-in", CERT_FILE, "-noout", "-checkend", "86400"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if check_exp.returncode != 0:
            return False, "Certificado expirado o próximo a vencer"
    except Exception:
        # Si openssl no está en PATH o falla la llamada, no podemos validar vigencia
        pass

    # 2. Comprobar si las IPs locales actuales están registradas en el SAN
    try:
        san_proc = subprocess.run(
            ["openssl", "x509", "-in", CERT_FILE, "-noout", "-ext", "subjectAltName"],
            capture_output=True,
            text=True
        )
        if san_proc.returncode == 0:
            san_text = san_proc.stdout
            current_ips = get_local_ips()
            for ip in current_ips:
                if ip != "127.0.0.1" and ip not in san_text:
                    return False, f"Cambio de IP detectado (IP {ip} no figura en el certificado)"
    except Exception:
        pass

    return True, "Certificado vigente y alineado con la IP actual"


def generate_with_openssl():
    """Genera certificados autofirmados con SAN dinámico para todas las IPs locales."""
    os.makedirs(CERTS_DIR, exist_ok=True)
    local_ips = get_local_ips()
    hostname = socket.gethostname()

    san_parts = [
        "IP:127.0.0.1",
        "DNS:localhost",
        f"DNS:{hostname}",
        f"DNS:{hostname}.local",
        "DNS:raspberrypi.local"
    ]
    for ip in local_ips:
        if ip != "127.0.0.1":
            san_parts.append(f"IP:{ip}")
    san_str = ",".join(san_parts)

    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", KEY_FILE,
        "-out", CERT_FILE,
        "-days", "825",
        "-subj", f"/C=ES/ST=Local/L=Local/O=DigitalScaleAI/CN={hostname}",
        "-addext", f"subjectAltName={san_str}"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            os.chmod(KEY_FILE, 0o600)
            os.chmod(CERT_FILE, 0o644)
        except Exception:
            pass
        return True, local_ips
    except Exception as e:
        return False, str(e)


def ensure_valid_certificates(silent=False):
    """
    Garantiza que existan certificados válidos y actualizados para la IP actual.
    Si no existen o la IP cambió, los regenera automáticamente.
    """
    valid, reason = is_cert_valid()
    if valid:
        return True

    if not silent:
        print(f"[AUTO-SSL] {reason}. Regenerando certificados HTTPS automáticamente...", flush=True)

    ok, res = generate_with_openssl()
    if ok:
        if not silent:
            print("[AUTO-SSL] Certificados generados y actualizados con éxito.", flush=True)
        return True
    else:
        if not silent:
            print(f"[AVISO] No se pudo generar certificado SSL: {res}", flush=True)
        return False


def main():
    print("=" * 65)
    print("       DIGITALSCALE AI - GENERADOR DE CERTIFICADOS HTTPS")
    print("=================================================================")

    valid, reason = is_cert_valid()
    print(f"[ESTADO] {reason}")

    if not valid:
        ok, res = generate_with_openssl()
        if ok:
            print(f"\n[OK] Certificados generados exitosamente en:\n     {CERTS_DIR}")
            print("\nDirecciones autorizadas para la cámara WebRTC:")
            for ip in res:
                print(f"   -> https://{ip}:8000")
        else:
            print(f"\n[ERROR] Falló la generación de certificados: {res}")
    else:
        print(f"\n[INFO] Los certificados existentes son válidos y cubren las IPs locales.")
        print(f"       Ubicación: {CERTS_DIR}")


if __name__ == "__main__":
    main()
