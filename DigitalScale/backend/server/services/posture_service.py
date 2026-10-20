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

    def initialize(self):
        """
        Carga los modelos de clustering y el detector de MediaPipe en memoria.
        """
        # Cargar MediaPipe Pose con soporte multi-versión
        import mediapipe as mp
        try:
            mp_pose = mp.solutions.pose
        except AttributeError:
            try:
                import mediapipe.python.solutions.pose as mp_pose
            except ImportError:
                import mediapipe.solutions.pose as mp_pose

        self.pose_detector = mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

        # Cargar modelos de clustering (Scaler, PCA, KMeans)
        if not (os.path.exists(SCALER_PATH) and os.path.exists(PCA_PATH) and os.path.exists(KMEANS_PATH)):
            raise FileNotFoundError(
                f"No se encontraron los modelos de postura en: {os.path.dirname(SCALER_PATH)}"
            )

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(PCA_PATH, "rb") as f:
            self.pca = pickle.load(f)
        with open(KMEANS_PATH, "rb") as f:
            self.kmeans = pickle.load(f)

        self._is_ready = True

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
        results = self.pose_detector.process(rgb_image)

        if not results or not results.pose_landmarks:
            return {
                "person_detected": False,
                "is_valid_pose": False,
                "posture_cluster": None,
                "posture_status": "No se detectó ninguna persona en la imagen",
                "keypoint_dict": None,
                "bbox": None
            }

        landmarks = results.pose_landmarks.landmark

        # Calcular cuadro delimitador (Bounding Box)
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        x1, x2 = int(max(0, min(x_coords))), int(min(w, max(x_coords)))
        y1, y2 = int(max(0, min(y_coords))), int(min(h, max(y_coords)))

        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        bbox_ratio = (bbox_w * bbox_h) / (w * h)

        # Mapear 33 landmarks a formato estándar COCO-17
        coco_landmarks = np.zeros((17, 2))
        for mp_idx, coco_idx in MP_TO_COCO.items():
            coco_landmarks[coco_idx] = [landmarks[mp_idx].x * w, landmarks[mp_idx].y * h]

        # Validar proximidad y corte con los bordes de la cámara
        shoulder_span = abs(coco_landmarks[5][0] - coco_landmarks[6][0])
        touching_border = (x1 < 10) or (x2 > w - 10) or (y1 < 10)
        is_too_close = (bbox_ratio > MAX_PERSON_RATIO) or (shoulder_span > w * MAX_SHOULDER_SPAN_RATIO) or touching_border
        is_too_far = bbox_ratio < MIN_PERSON_RATIO

        # Normalizar keypoints respecto a la bounding box
        normalized_kpts = []
        for px, py in coco_landmarks:
            norm_x = (px - x1) / bbox_w
            norm_y = (py - y1) / bbox_h
            normalized_kpts.extend([norm_x, norm_y])

        # Predecir cluster de postura
        X_scaled = self.scaler.transform([normalized_kpts])
        X_pca = self.pca.transform(X_scaled)
        posture_cluster = int(self.kmeans.predict(X_pca)[0])

        # Determinar validez de la postura (clusters 0 a 3 con encuadre correcto)
        is_valid_cluster = posture_cluster in [0, 1, 2, 3]
        is_valid_pose = is_valid_cluster and not is_too_close and not is_too_far

        if is_too_close:
            posture_status = "Demasiado cerca de la cámara. Retroceda para enfocar el torso completo."
        elif is_too_far:
            posture_status = "Demasiado lejos de la cámara. Acérquese un poco."
        elif not is_valid_cluster:
            posture_status = f"Postura inclinada o no erguida (Cluster {posture_cluster}). Póngase de pie recto."
        else:
            posture_status = "Postura correcta y bien encuadrada"

        keypoint_dict = {
            'left_shoulder_x': float(coco_landmarks[5][0]),
            'left_shoulder_y': float(coco_landmarks[5][1]),
            'right_shoulder_x': float(coco_landmarks[6][0]),
            'right_shoulder_y': float(coco_landmarks[6][1]),
            'left_eye_y': float(coco_landmarks[1][1]),
            'right_eye_y': float(coco_landmarks[2][1]),
        }

        return {
            "person_detected": True,
            "is_valid_pose": is_valid_pose,
            "posture_cluster": posture_cluster,
            "posture_status": posture_status,
            "keypoint_dict": keypoint_dict,
            "bbox": (x1, y1, x2, y2),
            "is_too_close": is_too_close,
            "is_too_far": is_too_far
        }

    def close(self):
        """Libera los recursos del detector MediaPipe al apagar el servidor."""
        if self.pose_detector is not None:
            self.pose_detector.close()
