"""
Punto de entrada principal de la aplicacion FastAPI para DigitalScale AI.
Configura el ciclo de vida, la inyeccion de dependencias de los servicios,
el middleware de seguridad CORS y el enrutamiento general.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configurar logging profesional en espanol
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DigitalScaleServer")

# Asegurar que el directorio backend este en sys.path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from .config import FRONTEND_DIR, HOST, PORT, RELOAD, ALLOWED_ORIGINS, PROTOCOL, IS_SSL, SSL_CERTFILE, SSL_KEYFILE
from .services import BMIPredictor, PostureAnalyzer
from .routes import health_router, predict_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicacion.
    Inicializa los modelos una sola vez al arrancar el servidor
    y libera los recursos de memoria al cerrarse.
    """
    logger.info("Iniciando motor de inteligencia artificial de DigitalScale...")

    # Inicializar servicio de inferencia de IMC (SE-DenseNet-121)
    bmi_service = BMIPredictor()
    bmi_service.initialize()
    app.state.bmi_service = bmi_service
    hw = bmi_service.hardware_info
    logger.info(f"[OK] Modelo SE-DenseNet-121 cargado en: {hw['device']} ({hw['gpu_name']})")

    # Inicializar servicio de analisis de postura (MediaPipe Pose + PCA/K-Means)
    posture_service = PostureAnalyzer()
    posture_service.initialize()
    app.state.posture_service = posture_service
    logger.info(f"[OK] Analizador de postura MediaPipe cargado (Modo: {posture_service.api_type}).")

    # Abrir navegador automaticamente cuando el servidor este listo
    def open_browser_when_ready():
        import time
        import webbrowser
        time.sleep(0.6)
        web_url = f"{PROTOCOL}://localhost:{PORT}"
        logger.info(f"Abriendo interfaz web en {web_url} ...")
        webbrowser.open(web_url)

    import threading
    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    yield

    logger.info("Apagando servidor y liberando recursos de IA...")
    posture_service.close()

# Creacion de la aplicacion FastAPI
app = FastAPI(
    title="DigitalScale AI - API de Estimacion Antropometrica",
    description="Servicio web para estimacion de Indice de Masa Corporal (IMC) y peso corporal mediante vision artificial con SE-DenseNet-121 y MediaPipe.",
    version="2.0.0",
    lifespan=lifespan
)

# Configuracion de politicas de acceso CORS
_allow_all = ALLOWED_ORIGINS == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de routers de endpoints
app.include_router(health_router)
app.include_router(predict_router)

# Montaje de archivos estaticos del frontend (modo solo lectura)
if os.path.exists(FRONTEND_DIR):
    css_path = os.path.join(FRONTEND_DIR, "css")
    js_path = os.path.join(FRONTEND_DIR, "js")
    assets_path = os.path.join(FRONTEND_DIR, "assets")
    pages_path = os.path.join(FRONTEND_DIR, "pages")

    if os.path.exists(css_path):
        app.mount("/css", StaticFiles(directory=css_path), name="css")
    if os.path.exists(js_path):
        app.mount("/js", StaticFiles(directory=js_path), name="js")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
    if os.path.exists(pages_path):
        app.mount("/pages", StaticFiles(directory=pages_path), name="pages")

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    def serve_frontend_root():
        """Sirve la pagina web principal del escaner biometrico."""
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"mensaje": "DigitalScale API en linea. index.html no encontrado."}

    @app.get("/results", include_in_schema=False)
    @app.get("/results.html", include_in_schema=False)
    def serve_frontend_results():
        """Sirve la pantalla del dashboard clinico de resultados."""
        results_file = os.path.join(FRONTEND_DIR, "pages", "results.html")
        if not os.path.exists(results_file):
            # Fallback en caso de estar en la raiz de frontend
            results_file = os.path.join(FRONTEND_DIR, "results.html")
        if os.path.exists(results_file):
            return FileResponse(results_file)
        return {"mensaje": "results.html no encontrado."}

if __name__ == "__main__":
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

    logger.info("Iniciando servidor backend modular de DigitalScale AI...")
    if IS_SSL:
        logger.info(f"[SEGURIDAD] Modo HTTPS activo (Certificado: {SSL_CERTFILE})")
        logger.info(f"Servidor Seguro Local:  https://localhost:{PORT}")
        if HOST in ("0.0.0.0", ""):
            logger.info(f"Acceso Remoto Seguro:   https://{local_ip}:{PORT}")
        logger.info(f"Documentacion (API):    https://localhost:{PORT}/docs")
        uvicorn.run("server.main:app", host=HOST, port=PORT, reload=RELOAD, ssl_keyfile=SSL_KEYFILE, ssl_certfile=SSL_CERTFILE)
    else:
        logger.info(f"Servidor Local:       http://localhost:{PORT}")
        if HOST in ("0.0.0.0", ""):
            logger.info(f"Acceso en Red Local:  http://{local_ip}:{PORT}")
        logger.info(f"Documentacion (API):  http://localhost:{PORT}/docs")
        uvicorn.run("server.main:app", host=HOST, port=PORT, reload=RELOAD)
