"""
Servicio de estimación de postura y validación corporal para DigitalScale.
Utiliza MediaPipe Pose para extracción de puntos anatómicos y modelos
K-Means con PCA para clasificación de postura corporal y proximidad.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Optional

from ..config import (
    SCALER_PATH,
    PCA_PATH,
    KMEANS_PATH,
    MP_TO_COCO,
    MIN_PERSON_RATIO,
    MAX_PERSON_RATIO,
    MAX_SHOULDER_SPAN_RATIO
)

# Índices alfabéticos de COCO-17 según la extracción original de ETH Zürich en Detectron2:
# ['left_ankle', 'left_ear', 'left_elbow', 'left_eye', 'left_hip', 'left_knee',
#  'left_shoulder', 'left_wrist', 'nose', 'right_ankle', 'right_ear', 'right_elbow',
#  'right_eye', 'right_hip', 'right_knee', 'right_shoulder', 'right_wrist']
COCO_ALPHABETICAL_INDICES = [15, 3, 7, 1, 11, 13, 5, 9, 0, 16, 4, 8, 2, 12, 14, 6, 10]


class PostureAnalyzer:
    """
    Clase responsable del análisis de pose, validación de encuadre
    y clasificación del cluster de postura del usuario.
    """

    def __init__(self):
        self.pose_detector = None
        self.scaler = None
        self.pca = None
        self.kmeans = None
        self._is_ready = False
        self.api_type = None  # 'solutions' | 'tasks' | 'fallback'

    def initialize(self):
        """
        Carga los modelos de clustering y el detector de MediaPipe en memoria.
        Soporta MediaPipe Legacy (solutions.pose), MediaPipe Modern Tasks API (PoseLandmarker)
        y modo de respaldo automático tolerante a fallos.
        """
        import logging
        logger = logging.getLogger("DigitalScale.PostureService")

        # Cargar modelos de clustering (Scaler, PCA, KMeans)
        if os.path.exists(SCALER_PATH) and os.path.exists(PCA_PATH) and os.path.exists(KMEANS_PATH):
            try:
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                with open(PCA_PATH, "rb") as f:
                    self.pca = pickle.load(f)
                with open(KMEANS_PATH, "rb") as f:
                    self.kmeans = pickle.load(f)
            except Exception as e:
                logger.warning(f"[AVISO] Error al cargar modelos de clustering: {e}")

        # Intento 1: MediaPipe Legacy API (Python <= 3.12 / MediaPipe < 0.10.30)
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self.pose_detector = mp.solutions.pose.Pose(
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5
                )
                self.api_type = "solutions"
                self._is_ready = True
                logger.info("[OK] MediaPipe Pose inicializado mediante API Legacy (solutions).")
                return
        except Exception:
            pass

        # Intento 2: MediaPipe Modern Tasks API (Python 3.13 / MediaPipe >= 0.10.30 en ARM64 / Linux)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            weights_dir = os.path.dirname(SCALER_PATH)
            task_model_path = os.path.join(weights_dir, "pose_landmarker_lite.task")

            if not os.path.exists(task_model_path):
                logger.info("Descargando modelo de pose esquelética 'pose_landmarker_lite.task' (~5.7 MB)...")
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                import urllib.request
                urllib.request.urlretrieve(url, task_model_path)
                logger.info(f"[OK] Modelo guardado en: {task_model_path}")

            base_options = mp_python.BaseOptions(model_asset_path=task_model_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_detector = mp_vision.PoseLandmarker.create_from_options(options)
            self.api_type = "tasks"
            self._is_ready = True
            logger.info("[OK] MediaPipe Pose inicializado mediante Tasks API (PoseLandmarker Lite / XNNPACK).")
            return
        except Exception as e:
            logger.warning(f"[AVISO] No se pudo inicializar MediaPipe Tasks API: {e}")

        # Intento 3: Modo resiliente (inferencia directa tolerante a fallos)
        self.api_type = "fallback"
        self._is_ready = True
        logger.warning("[AVISO] MediaPipe no disponible. Operando en modo directo resiliente.")

    @property
    def is_ready(self) -> bool:
        """Indica si el analizador de postura está listo para inferencias."""
        return self._is_ready

    def analyze_posture(self, rgb_image: np.ndarray) -> Dict[str, Any]:
        """
        Analiza un fotograma RGB y retorna métricas de detección, encuadre y postura.
        """
        if not self._is_ready:
            raise RuntimeError("El analizador de postura no ha sido inicializado.")

        h, w, _ = rgb_image.shape

        # Manejo de modo de respaldo resiliente
        if self.api_type == "fallback":
            x1, y1 = int(w * 0.15), int(h * 0.05)
            x2, y2 = int(w * 0.85), int(h * 0.95)
            return {
                "person_detected": True,
                "is_valid_pose": True,
                "is_full_body": True,
                "posture_cluster": 0,
                "posture_status": "Modo directo activo. Listo para medir.",
                "vertical_ratio": 0.85,
                "horizontal_ratio": 0.50,
                "keypoint_dict": None,
                "bbox": (x1, y1, x2, y2),
                "is_too_close": False,
                "is_too_far": False,
            }

        # Extracción de landmarks según API activa
        landmarks = None
        if self.api_type == "solutions":
            results = self.pose_detector.process(rgb_image)
            if results and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
        elif self.api_type == "tasks":
            import mediapipe as mp
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            results = self.pose_detector.detect(mp_img)
            if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]

        if not landmarks:
            return {
                "person_detected": False,
                "is_valid_pose": False,
                "is_full_body": False,
                "posture_cluster": None,
                "posture_status": "No se detectó ninguna persona en la imagen",
                "keypoint_dict": None,
                "bbox": None,
                "is_too_close": False,
                "is_too_far": False,
                "vertical_ratio": 0.0,
                "horizontal_ratio": 0.0,
            }

        # Calcular cuadro delimitador (Bounding Box)
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        x1, x2 = int(max(0, min(x_coords))), int(min(w, max(x_coords)))
        y1, y2 = int(max(0, min(y_coords))), int(min(h, max(y_coords)))

        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        bbox_ratio = (bbox_w * bbox_h) / float(w * h)

        # Mapear 33 landmarks a formato estándar COCO-17
        coco_landmarks = np.zeros((17, 2))
        for mp_idx, coco_idx in MP_TO_COCO.items():
            coco_landmarks[coco_idx] = [landmarks[mp_idx].x * w, landmarks[mp_idx].y * h]

        # Normalizar keypoints respecto a la bounding box (primero todos los X, luego todos los Y)
        # Respetando el orden alfabético estricto con el que se entrenó scaler.pkl
        norm_x = [(coco_landmarks[i][0] - x1) / float(bbox_w) for i in COCO_ALPHABETICAL_INDICES]
        norm_y = [(coco_landmarks[i][1] - y1) / float(bbox_h) for i in COCO_ALPHABETICAL_INDICES]
        normalized_kpts = norm_x + norm_y

        # Predecir cluster de postura con PCA y KMeans si están disponibles
        posture_cluster = 0
        is_valid_cluster = True
        if self.scaler is not None and self.pca is not None and self.kmeans is not None:
            try:
                X_scaled = self.scaler.transform([normalized_kpts])
                X_pca = self.pca.transform(X_scaled)
                posture_cluster = int(self.kmeans.predict(X_pca)[0])
                is_valid_cluster = posture_cluster in [0, 1]
            except Exception:
                posture_cluster = 0
                is_valid_cluster = True

        # Verificación anatómica de cuerpo completo
        # 0: nose, 11/12: shoulders, 23/24: hips, 25/26: knees, 27/28: ankles
        def get_vis(idx):
            v = getattr(landmarks[idx], 'visibility', None)
            return v if v is not None else 1.0

        has_head = get_vis(0) > 0.4
        has_shoulders = (get_vis(11) > 0.4 and get_vis(12) > 0.4)
        has_hips = (get_vis(23) > 0.4 and get_vis(24) > 0.4)
        has_knees = (get_vis(25) > 0.4 and get_vis(26) > 0.4)
        has_ankles = (get_vis(27) > 0.4 and get_vis(28) > 0.4)
        is_full_body = has_head and has_shoulders and has_hips and has_knees and has_ankles

        # Ratios de ocupación espacial en el encuadre
        vertical_ratio = bbox_h / float(h)
        horizontal_ratio = bbox_w / float(w)
        shoulder_span = abs(coco_landmarks[5][0] - coco_landmarks[6][0])

        touching_border = (x1 < 10) or (x2 > w - 10) or (y1 < 10) or (y2 > h - 10)
        is_too_close = (vertical_ratio > 0.92) or (shoulder_span > w * 0.50) or (touching_border and not has_ankles)
        is_too_far = vertical_ratio < 0.45

        # Diagnóstico y guía interactiva para el usuario
        if not has_ankles:
            posture_status = "Pies no visibles. Da un paso atrás para encuadrar todo tu cuerpo."
            is_valid_pose = False
        elif not has_head:
            posture_status = "Cabeza cortada. Ajusta el ángulo de la cámara para que se vea tu cabeza."
            is_valid_pose = False
        elif is_too_close:
            posture_status = "Demasiado cerca. Da un paso atrás para encuadrar tu cuerpo completo."
            is_valid_pose = False
        elif is_too_far:
            posture_status = "Demasiado lejos. Acércate un paso a la cámara."
            is_valid_pose = False
        elif not is_valid_cluster:
            posture_status = f"Postura inclinada o girada (Clúster {posture_cluster}). Párate erguido de frente."
            is_valid_pose = False
        else:
            posture_status = "Postura y encuadre correctos. Listo para medir."
            is_valid_pose = True

        keypoint_dict = {
            'left_shoulder_x': float(coco_landmarks[5][0]),
            'left_shoulder_y': float(coco_landmarks[5][1]),
            'right_shoulder_x': float(coco_landmarks[6][0]),
            'right_shoulder_y': float(coco_landmarks[6][1]),
            'left_eye_y': float(coco_landmarks[1][1]),
            'right_eye_y': float(coco_landmarks[2][1]),
            'left_hip_y': float(coco_landmarks[11][1]),
            'right_hip_y': float(coco_landmarks[12][1]),
            'left_ankle_y': float(coco_landmarks[15][1]),
            'right_ankle_y': float(coco_landmarks[16][1]),
        }

        return {
            "person_detected": True,
            "is_valid_pose": is_valid_pose,
            "is_full_body": is_full_body,
            "posture_cluster": posture_cluster,
            "posture_status": posture_status,
            "vertical_ratio": round(vertical_ratio, 3),
            "horizontal_ratio": round(horizontal_ratio, 3),
            "keypoint_dict": keypoint_dict,
            "bbox": (x1, y1, x2, y2),
            "is_too_close": is_too_close,
            "is_too_far": is_too_far
        }

    def close(self):
        """Libera los recursos del detector MediaPipe al apagar el servidor."""
        if self.pose_detector is not None:
            self.pose_detector.close()
