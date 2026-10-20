"""
Configuracion central del servidor DigitalScale AI.
Contiene rutas a modelos, constantes de normalizacion de imagenes, umbrales de postura
y parametros de red configurables por variables de entorno.
"""

import os

# Carga opcional de variables de entorno desde archivo .env
try:
    from dotenv import load_dotenv
    # Buscar .env en la raiz del proyecto o en backend
    env_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    env_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.exists(env_root):
        load_dotenv(env_root)
    elif os.path.exists(env_backend):
        load_dotenv(env_backend)
    else:
        load_dotenv()
except ImportError:
    pass

# Rutas base del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "best_model.ckpt")
ALT_WEIGHTS_PATH = os.path.join(BASE_DIR, "trained_models", "model_5", "best_model.ckpt")

KEYPOINT_MODELS_DIR = os.path.join(BASE_DIR, "trained_models", "keypoint_models")
SCALER_PATH = os.path.join(KEYPOINT_MODELS_DIR, "scaler.pkl")
PCA_PATH = os.path.join(KEYPOINT_MODELS_DIR, "pca.pkl")
KMEANS_PATH = os.path.join(KEYPOINT_MODELS_DIR, "kmeans.pkl")

# Ruta a la carpeta estatica del frontend (modo solo lectura)
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

# Parametros de red y ejecucion del servidor FastAPI
HOST = os.getenv("DIGITALSCALE_HOST", "0.0.0.0")
PORT = int(os.getenv("DIGITALSCALE_PORT", "8000"))
RELOAD = os.getenv("DIGITALSCALE_RELOAD", "false").lower() in ("true", "1", "yes")

# Origenes permitidos para seguridad CORS
_origins_env = os.getenv("DIGITALSCALE_ALLOWED_ORIGINS", "")
if _origins_env:
    ALLOWED_ORIGINS = [orig.strip() for orig in _origins_env.split(",") if orig.strip()]
else:
    ALLOWED_ORIGINS = ["*"]

# Parametros de preprocesamiento para la red neuronal SE-DenseNet-121
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# Mapeo de 33 Puntos Anatomicos de MediaPipe Pose a los 17 Keypoints estandar de COCO
MP_TO_COCO = {
    0: 0,   # Nariz
    2: 1,   # Ojo Izquierdo
    5: 2,   # Ojo Derecho
    7: 3,   # Oreja Izquierda
    8: 4,   # Oreja Derecha
    11: 5,  # Hombro Izquierdo
    12: 6,  # Hombro Derecho
    13: 7,  # Codo Izquierdo
    14: 8,  # Codo Derecho
    15: 9,  # Muneca Izquierda
    16: 10, # Muneca Derecha
    23: 11, # Cadera Izquierda
    24: 12, # Cadera Derecha
    25: 13, # Rodilla Izquierda
    26: 14, # Rodilla Derecha
    27: 15, # Tobillo Izquierdo
    28: 16  # Tobillo Derecho
}

# Umbrales para validacion de encuadre y postura
MIN_PERSON_RATIO = 0.15     # Si el area es menor al 15%, esta muy lejos
MAX_PERSON_RATIO = 0.65     # Si el area es mayor al 65%, esta muy cerca
MAX_SHOULDER_SPAN_RATIO = 0.45 # Ancho maximo de hombros respecto al ancho de la camara
TORSO_MARGIN_FACTOR = 1.4   # Factor de holgura para el recorte del torso
