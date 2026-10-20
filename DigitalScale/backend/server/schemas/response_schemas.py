"""
Esquemas de formato de salida (Responses) para la API de DigitalScale.
"""

from typing import Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """
    Respuesta del estado de salud del servidor y recursos de hardware.
    """
    status: str = Field(..., description="Estado general del servidor (ej. online)")
    device: str = Field(..., description="Dispositivo de cómputo en uso (cuda o cpu)")
    gpu_name: str = Field(..., description="Nombre de la tarjeta gráfica detectada")
    cuda_available: bool = Field(..., description="Indica si PyTorch tiene aceleración por GPU")
    model_loaded: bool = Field(..., description="Indica si la red neuronal SE-DenseNet está en memoria")
    mediapipe_ready: bool = Field(..., description="Indica si el detector MediaPipe está listo")

class PredictionResponse(BaseModel):
    """
    Respuesta detallada de la estimación de IMC, postura y métricas antropométricas.
    """
    success: bool = Field(..., description="Indica si la predicción se completó con éxito")
    bmi: Optional[float] = Field(None, description="Índice de Masa Corporal estimado en kg/m²")
    height_m: Optional[float] = Field(None, description="Estatura utilizada en el cálculo (en metros)")
    estimated_weight_kg: Optional[float] = Field(None, description="Peso corporal estimado en kg")
    category: Optional[str] = Field(None, description="Clasificación según la OMS (ej. Normal, Sobrepeso)")
    category_code: Optional[str] = Field(None, description="Código de categoría (ej. normal, overweight)")
    category_color: Optional[str] = Field(None, description="Color hexadecimal para la interfaz de usuario")
    posture_cluster: Optional[int] = Field(None, description="Identificador del cluster de postura (0 a 3)")
    is_valid_pose: bool = Field(..., description="Indica si la postura fue adecuada para la medición")
    posture_status: str = Field(..., description="Descripción del estado de postura y encuadre")
    cropped_torso_base64: Optional[str] = Field(None, description="Miniatura del torso recortado en formato Base64")
    message: Optional[str] = Field(None, description="Mensaje explicativo del resultado o advertencias")
