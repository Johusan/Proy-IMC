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

    # Control de calidad estricto: Si la postura o encuadre no son válidos,
    # no se emite una estimación engañosa. Se guía al usuario para corregir la captura.
    if not posture_result["is_valid_pose"]:
        pil_img = Image.fromarray(rgb_image)
        bbox = posture_result.get("bbox")
        try:
            if bbox:
                preview_crop = ImageProcessor.get_clean_fullbody_crop(pil_img, bbox, w, h)
            else:
                preview_crop = pil_img
            preview_b64 = ImageProcessor.image_to_base64(preview_crop)
        except Exception:
            preview_b64 = None

        return PredictionResponse(
            success=False,
            is_valid_pose=False,
            posture_cluster=posture_result.get("posture_cluster"),
            posture_status=posture_result["posture_status"],
            cropped_torso_base64=preview_b64,
            message=posture_result["posture_status"]
        )

    # 2. Recorte inteligente del torso sin bandas negras
    keypoints = posture_result.get("keypoint_dict")
    pil_img = Image.fromarray(rgb_image)

    if keypoints:
        try:
            cropped_torso = ImageProcessor.get_clean_torso_crop(pil_img, keypoints, w, h)
        except Exception:
            bbox = posture_result.get("bbox") or (0, 0, w, h)
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w, bbox[2]), min(h, bbox[3])
            cropped_torso = pil_img.crop((x1, y1, x2, y2))
    else:
        bbox = posture_result.get("bbox") or (0, 0, w, h)
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
    Recibe un fotograma o ráfaga de fotogramas Base64 capturados por el navegador.
    Si se proporciona una ráfaga (images_base64), aplica agregación estadística multi-frame (mediana)
    para estabilizar la medición y reducir el error real en ~9.5%.
    """
    import numpy as np

    # 1. Modo ráfaga multi-fotograma
    if payload.images_base64 and len(payload.images_base64) >= 2:
        valid_bmis = []
        last_response = None
        last_crop_b64 = None

        for b64_frame in payload.images_base64:
            rgb_img = ImageProcessor.decode_base64_image(b64_frame)
            if rgb_img is None:
                continue
            resp = _execute_prediction_pipeline(request, rgb_img, payload.height_m)
            last_response = resp
            if resp.success and resp.bmi is not None:
                valid_bmis.append(resp.bmi)
                last_crop_b64 = resp.cropped_torso_base64

        if not valid_bmis:
            return last_response or PredictionResponse(
                success=False,
                is_valid_pose=False,
                posture_status="Ningún fotograma de la ráfaga cumplió los requisitos de calidad",
                message="Manténgase erguido y completamente encuadrado durante la captura."
            )

        # Agregación por mediana
        final_bmi = float(np.median(valid_bmis))
        intra_std = float(np.std(valid_bmis)) if len(valid_bmis) > 1 else 0.0
        final_bmi_round = round(final_bmi, 2)
        intra_std_round = round(intra_std, 3)

        bmi_service = request.app.state.bmi_service
        cat_name, cat_code, cat_color = bmi_service.categorize_who(final_bmi_round)
        est_weight = bmi_service.calculate_weight(final_bmi_round, payload.height_m)

        return PredictionResponse(
            success=True,
            bmi=final_bmi_round,
            height_m=payload.height_m,
            estimated_weight_kg=est_weight,
            category=cat_name,
            category_code=cat_code,
            category_color=cat_color,
            posture_cluster=last_response.posture_cluster if last_response else 0,
            is_valid_pose=True,
            posture_status=f"Medición estabilizada en ráfaga ({len(valid_bmis)} fotogramas)",
            cropped_torso_base64=last_crop_b64,
            intra_std=intra_std_round,
            frames_analyzed=len(valid_bmis),
            message=f"Medición multi-fotograma completada. Dispersión intra-sesión: ±{intra_std_round} kg/m²."
        )

    # 2. Modo fotograma único estándar (con TTA)
    if not payload.image_base64:
        raise HTTPException(
            status_code=400, 
            detail="Debe proporcionar image_base64 o images_base64."
        )

    rgb_image = ImageProcessor.decode_base64_image(payload.image_base64)
    if rgb_image is None:
        raise HTTPException(
            status_code=400,
            detail="No se pudo decodificar la imagen Base64 proporcionada."
        )

    return _execute_prediction_pipeline(request, rgb_image, payload.height_m)
