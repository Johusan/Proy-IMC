"""
Lanzador de conveniencia en la raiz de DigitalScale.
Permite ejecutar 'python server.py' directamente desde la carpeta DigitalScale.
Redirige la ejecucion al servidor modular en backend/server/main.py.
"""

import sys
import os
import logging

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Evitar conflicto si el archivo actual se llama server.py y se importa desde la raiz
if "server" in sys.modules and not hasattr(sys.modules["server"], "__path__"):
    del sys.modules["server"]

from server.config import HOST, PORT, RELOAD

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("DigitalScaleLauncher")

    import socket

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()

    logger.info("Iniciando DigitalScale AI desde la raiz del proyecto...")
    logger.info(f"Servidor Local:       http://localhost:{PORT}")
    if HOST in ("0.0.0.0", ""):
        logger.info(f"Acceso en Red Local:  http://{local_ip}:{PORT}")
    logger.info(f"Documentacion (API):  http://localhost:{PORT}/docs")

    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=RELOAD)
