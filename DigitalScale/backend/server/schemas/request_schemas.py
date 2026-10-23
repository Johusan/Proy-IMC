"""
Esquemas de validación de entrada (Requests) para la API de DigitalScale.
"""

from typing import Optional
from pydantic import BaseModel, Field

class Base64PredictRequest(BaseModel):
    """
    Esquema para peticiones con imagen en formato Base64 enviadas desde clientes web.
    """
    image_base64: Optional[str] = Field(
        None, 
        description="Cadena en formato Base64 con la imagen capturada por la cámara (ej. data:image/jpeg;base64,...)"
    )
    images_base64: Optional[list[str]] = Field(
        None,
        description="Lista de cadenas Base64 (ráfaga de 2 a 5 fotogramas consecutivos) para estabilización y reducción del error real"
    )
    height_m: Optional[float] = Field(
        None, 
        ge=0.5, 
        le=2.5, 
        description="Estatura del usuario en metros (ej. 1.75). Opcional, necesario para calcular el peso en kg."
    )
