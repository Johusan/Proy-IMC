"""
Servicios con la lógica de negocio desacoplada de la API web.
"""

from .image_service import ImageProcessor
from .posture_service import PostureAnalyzer
from .bmi_service import BMIPredictor

__all__ = ["ImageProcessor", "PostureAnalyzer", "BMIPredictor"]
