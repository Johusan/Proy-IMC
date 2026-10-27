"""
DIGITALSCALE AI - GENERADOR DE CERTIFICADOS SSL (PYTHON CROSS-PLATFORM)
Permite generar cert.pem y key.pem tanto en Linux (Raspberry Pi) como en Windows.
Usa OpenSSL si esta en el PATH o la libreria cryptography si esta disponible.
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
    ips = set(["127.0.0.1"])
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
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
    return list(ips)


def generate_with_openssl():
    os.makedirs(CERTS_DIR, exist_ok=True)
    local_ips = get_local_ips()
    hostname = socket.gethostname()

    san_parts = ["IP:127.0.0.1", "DNS:localhost", f"DNS:{hostname}", f"DNS:{hostname}.local", "DNS:raspberrypi.local"]
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
        return True, local_ips
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 65)
    print("       DIGITALSCALE AI - GENERADOR DE CERTIFICADOS HTTPS")
    print("=" * 65)

    ok, res = generate_with_openssl()
    if ok:
        print(f"\n[OK] Certificados generados exitosamente en:\n     {CERTS_DIR}")
        print("\nDirecciones autorizadas para la camara WebRTC:")
        for ip in res:
            print(f"   -> https://{ip}:8000")
        print("\nInicie el servidor normalmente con: python server.py (o ./iniciar_servidor.sh)")
    else:
        print(f"\n[AVISO] No se pudo ejecutar OpenSSL directamente: {res}")
        print("Si esta en Raspberry Pi, ejecute: ./generar_certificados.sh")


if __name__ == "__main__":
    main()
