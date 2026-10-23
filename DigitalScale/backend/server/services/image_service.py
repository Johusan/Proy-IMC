"""
Servicio de procesamiento de imágenes para DigitalScale.
Encapsula la decodificación, el recorte inteligente del torso sin barras negras
y la conversión a formatos visualizables en clientes web.
"""

import io
import base64
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

from ..config import TORSO_MARGIN_FACTOR

class ImageProcessor:
    """
    Clase responsable de la manipulación, validación y recorte anatómico de imágenes.
    """

    @staticmethod
    def decode_file_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decodifica un arreglo de bytes en una imagen NumPy en espacio de color RGB.
        Retorna None si la imagen está corrupta o no es válida.
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr_image is None or bgr_image.size == 0:
                return None
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    @staticmethod
    def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
        """
        Decodifica una cadena Base64 (con o sin encabezado data:image/...) a imagen NumPy RGB.
        """
        try:
            raw_b64 = base64_str
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(raw_b64)
            return ImageProcessor.decode_file_bytes(image_bytes)
        except Exception:
            return None

    @staticmethod
    def get_clean_torso_crop(
        pil_img: Image.Image, 
        keypoints: Dict[str, float], 
        img_w: int, 
        img_h: int, 
        margin_factor: float = TORSO_MARGIN_FACTOR
    ) -> Image.Image:
        """
        Calcula y extrae un recorte centrado en el torso delimitado estrictamente
        dentro de las dimensiones de la imagen.
        
        Si el recuadro sobrepasa los márgenes de la cámara, desplaza la ventana hacia adentro
        para capturar píxeles reales, evitando que PIL rellene con bandas negras artificiales.
        """
        # Calcular ancho de hombros
        shoulder_w = abs(keypoints['right_shoulder_x'] - keypoints['left_shoulder_x'])
        if shoulder_w < 10:
            shoulder_w = img_w * 0.25 # Valor de respaldo si están muy juntos

        margin = shoulder_w * (margin_factor - 1.0)
        
        min_sx = min(keypoints['left_shoulder_x'], keypoints['right_shoulder_x'])
        max_sx = max(keypoints['left_shoulder_x'], keypoints['right_shoulder_x'])
        min_ey = min(keypoints['left_eye_y'], keypoints['right_eye_y'])
        max_sy = max(keypoints['left_shoulder_y'], keypoints['right_shoulder_y'])

        # Límites crudos del torso
        raw_x1 = min_sx - margin
        raw_x2 = max_sx + margin
        raw_y1 = min_ey - (margin * 0.5)
        raw_y2 = max_sy + (shoulder_w * 1.6)

        box_w = raw_x2 - raw_x1
        box_h = raw_y2 - raw_y1
        side = max(box_w, box_h)
        
        cx = (raw_x1 + raw_x2) / 2.0
        cy = (raw_y1 + raw_y2) / 2.0

        x1 = cx - side / 2.0
        x2 = cx + side / 2.0
        y1 = cy - side / 2.0
        y2 = cy + side / 2.0

        # Desplazamiento preventivo hacia el interior del fotograma
        if x1 < 0:
            x2 += abs(x1)
            x1 = 0.0
        if x2 > img_w:
            x1 -= (x2 - img_w)
            x2 = float(img_w)
        if y1 < 0:
            y2 += abs(y1)
            y1 = 0.0
        if y2 > img_h:
            y1 -= (y2 - img_h)
            y2 = float(img_h)

        # Acotamiento estricto a los bordes físicos de la cámara
        final_x1 = max(0, min(img_w - 2, int(round(x1))))
        final_x2 = max(final_x1 + 10, min(img_w, int(round(x2))))
        final_y1 = max(0, min(img_h - 2, int(round(y1))))
        final_y2 = max(final_y1 + 10, min(img_h, int(round(y2))))

        return pil_img.crop((final_x1, final_y1, final_x2, final_y2))

    @staticmethod
    def get_clean_fullbody_crop(
        pil_img: Image.Image,
        bbox: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        margin_factor: float = 1.08
    ) -> Image.Image:
        """
        Calcula y extrae un recorte centrado en el cuerpo completo de la persona,
        aplicando un margen anatómico proporcional y acotándose a los límites del sensor.
        """
        x1, y1, x2, y2 = bbox
        bw = max(10, x2 - x1)
        bh = max(10, y2 - y1)

        margin_x = int(bw * (margin_factor - 1.0) * 0.5)
        margin_y = int(bh * (margin_factor - 1.0) * 0.5)

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(img_w, x2 + margin_x)
        crop_y2 = min(img_h, y2 + margin_y)

        return pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    @staticmethod
    def image_to_base64(pil_img: Image.Image, quality: int = 85) -> str:
        """
        Convierte una imagen PIL a formato Data URI Base64 (data:image/jpeg;base64,...)
        para su visualización inmediata en el frontend.
        """
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
