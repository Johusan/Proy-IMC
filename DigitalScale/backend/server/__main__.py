"""
Modulo ejecutable principal para el paquete 'server'.
Permite ejecutar 'python server' o 'python -m server' desde la carpeta backend.
"""

import os
import sys
import logging

# Asegurar que el directorio backend este en sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from server.config import HOST, PORT, RELOAD

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("DigitalScaleModule")

    logger.info("Iniciando DigitalScale AI como modulo...")
    logger.info(f"Servidor Web Local:  http://{HOST}:{PORT}")
    logger.info(f"Documentacion (API): http://{HOST}:{PORT}/docs")

    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=RELOAD)
