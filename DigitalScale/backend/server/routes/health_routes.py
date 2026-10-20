"""
Ruta de diagnóstico y monitoreo de salud del servidor (Health Check).
"""

from fastapi import APIRouter, Request
from ..schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["Monitoreo"])

@router.get("/health", response_model=HealthResponse)
def get_health_status(request: Request):
    """
    Verifica el estado de disponibilidad del servidor, el dispositivo de cómputo (CUDA/CPU)
    y confirma si los modelos neuronales y detectores están activos en memoria.
    """
    bmi_service = getattr(request.app.state, "bmi_service", None)
    posture_service = getattr(request.app.state, "posture_service", None)

    if bmi_service is None or posture_service is None:
        return HealthResponse(
            status="starting",
            device="CPU",
            gpu_name="Cargando modelos...",
            cuda_available=False,
            model_loaded=False,
            mediapipe_ready=False
        )

    hw_info = bmi_service.hardware_info

    return HealthResponse(
        status="online",
        device=hw_info["device"],
        gpu_name=hw_info["gpu_name"],
        cuda_available=hw_info["cuda_available"],
        model_loaded=bmi_service.is_ready,
        mediapipe_ready=posture_service.is_ready
    )
