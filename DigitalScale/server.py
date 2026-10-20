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

    logger.info("Iniciando DigitalScale AI desde la raiz del proyecto...")
    logger.info(f"Servidor Web Local:  http://{HOST}:{PORT}")
    logger.info(f"Documentacion (API): http://{HOST}:{PORT}/docs")

    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=RELOAD)
