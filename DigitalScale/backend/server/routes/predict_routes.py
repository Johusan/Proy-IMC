"""
Rutas de predicción antropométrica e inferencia de IMC.
Orquesta los servicios de procesamiento de imágenes, análisis de postura y predicción por IA.
"""

from typing import Optional
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from PIL import Image

from ..schemas import Base64PredictRequest, PredictionResponse
from ..services import ImageProcessor

router = APIRouter(prefix="/api", tags=["Predicción de IMC"])

def _execute_prediction_pipeline(
    request: Request,
    rgb_image,
    height_m: Optional[float] = None
) -> PredictionResponse:
    """
    Función interna que coordina los servicios de postura, recorte e inferencia.
    """
    bmi_service = request.app.state.bmi_service
    posture_service = request.app.state.posture_service

    h, w, _ = rgb_image.shape

    # 1. Analizar postura y encuadre con MediaPipe + K-Means
    posture_result = posture_service.analyze_posture(rgb_image)

    if not posture_result["person_detected"]:
        return PredictionResponse(
            success=False,
            is_valid_pose=False,
            posture_status="No se detectó ninguna persona en la imagen",
            message="Asegúrese de estar de pie, de frente a la cámara y con buena iluminación."
        )

    # 2. Recorte inteligente del torso sin bandas negras
    keypoints = posture_result["keypoint_dict"]
    pil_img = Image.fromarray(rgb_image)

    try:
        cropped_torso = ImageProcessor.get_clean_torso_crop(pil_img, keypoints, w, h)
    except Exception:
        # Recorte de respaldo acotado si fallara el cálculo geométrico
        bbox = posture_result["bbox"]
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])
        cropped_torso = pil_img.crop((x1, y1, x2, y2))

    # 3. Inferencia con SE-DenseNet-121
    bmi = bmi_service.predict_bmi(cropped_torso)

    # 4. Clasificación clínica de la OMS
    cat_name, cat_code, cat_color = bmi_service.categorize_who(bmi)

    # 5. Cálculo de peso estimado si se suministró la estatura
    estimated_weight = bmi_service.calculate_weight(bmi, height_m)

    # 6. Miniatura en Base64 para visualización en el frontend
    torso_base64 = ImageProcessor.image_to_base64(cropped_torso)

    return PredictionResponse(
        success=True,
        bmi=bmi,
        height_m=height_m,
        estimated_weight_kg=estimated_weight,
        category=cat_name,
        category_code=cat_code,
        category_color=cat_color,
        posture_cluster=posture_result["posture_cluster"],
        is_valid_pose=posture_result["is_valid_pose"],
        posture_status=posture_result["posture_status"],
        cropped_torso_base64=torso_base64,
        message="Medición completada exitosamente."
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict_from_form_file(
    request: Request,
    file: UploadFile = File(..., description="Archivo de imagen (JPEG/PNG)"),
    height_m: Optional[float] = Form(None, description="Estatura en metros (ej. 1.75)")
):
    """
    Recibe una imagen como archivo adjunto Multipart (formulario web o API externa).
    """
    contents = await file.read()
    rgb_image = ImageProcessor.decode_file_bytes(contents)

    if rgb_image is None:
        raise HTTPException(
            status_code=400,
            detail="El archivo proporcionado no es una imagen válida o está dañado."
        )

    return _execute_prediction_pipeline(request, rgb_image, height_m)

@router.post("/predict-base64", response_model=PredictionResponse)
def predict_from_base64(
    request: Request,
    payload: Base64PredictRequest
):
    """
    Recibe un fotograma capturado directamente por la cámara del navegador en formato Base64.
    """
    rgb_image = ImageProcessor.decode_base64_image(payload.image_base64)

    if rgb_image is None:
        raise HTTPException(
            status_code=400,
            detail="No se pudo decodificar la imagen Base64 proporcionada."
        )

    return _execute_prediction_pipeline(request, rgb_image, payload.height_m)
