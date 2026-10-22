"""
Preprocesador anatómico de Dataset para DigitalScale AI.
Alinea el dataset 2DImage2BMI (FVL2020) con el pipeline de inferencia en producción:
1. Extrae los puntos anatómicos clave usando MediaPipe Pose.
2. Aplica el recorte inteligente de torso idéntico al de ImageProcessor en producción.
3. Guarda los recortes en 'Image Dataset/Torso_Crops' y genera 'torso_unified_dataset.csv'.
"""

import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
from PIL import Image

# Rutas del entorno
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DIGITALSCALE_BACKEND = os.path.join(PROJECT_ROOT, "DigitalScale", "backend")

if DIGITALSCALE_BACKEND not in sys.path:
    sys.path.append(DIGITALSCALE_BACKEND)

try:
    import mediapipe as mp
    try:
        mp_pose = mp.solutions.pose
    except AttributeError:
        import mediapipe.python.solutions.pose as mp_pose
except ImportError:
    print("Error: mediapipe no está instalado en este entorno.")
    sys.exit(1)


def get_clean_torso_crop(pil_img: Image.Image, keypoints: dict, img_w: int, img_h: int, margin_factor: float = 1.35) -> Image.Image:
    """
    Replica exacta de ImageProcessor.get_clean_torso_crop del servidor DigitalScale.
    Garantiza consistencia absoluta entre entrenamiento e inferencia.
    """
    shoulder_w = abs(keypoints['right_shoulder_x'] - keypoints['left_shoulder_x'])
    if shoulder_w < 10:
        shoulder_w = img_w * 0.25

    margin = shoulder_w * (margin_factor - 1.0)
    
    min_sx = min(keypoints['left_shoulder_x'], keypoints['right_shoulder_x'])
    max_sx = max(keypoints['left_shoulder_x'], keypoints['right_shoulder_x'])
    min_ey = min(keypoints['left_eye_y'], keypoints['right_eye_y'])
    max_sy = max(keypoints['left_shoulder_y'], keypoints['right_shoulder_y'])

    # Límites del torso
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

    # Desplazamiento hacia el interior
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

    final_x1 = max(0, min(img_w - 2, int(round(x1))))
    final_x2 = max(final_x1 + 10, min(img_w, int(round(x2))))
    final_y1 = max(0, min(img_h - 2, int(round(y1))))
    final_y2 = max(final_y1 + 10, min(img_h, int(round(y2))))

    return pil_img.crop((final_x1, final_y1, final_x2, final_y2))


def run_torso_preprocessing():
    print("=" * 65)
    print("DIGITALSCALE AI — PREPROCESAMIENTO ANATÓMICO (RECORTE DE TORSO)")
    print("=" * 65)

    input_csv = os.path.join(CURRENT_DIR, "unified_dataset.csv")
    if not os.path.exists(input_csv):
        print(f"Error: No se encontró {input_csv}.")
        return

    df = pd.read_csv(input_csv)
    print(f"Total de imágenes en el dataset original: {len(df)}")

    crops_dir = os.path.join(CURRENT_DIR, "Image Dataset", "Torso_Crops")
    os.makedirs(crops_dir, exist_ok=True)

    output_csv = os.path.join(CURRENT_DIR, "torso_unified_dataset.csv")

    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )

    valid_records = []
    skipped_count = 0
    t0 = time.time()

    print("\nIniciando extracción anatómica y recorte de torso con MediaPipe...")
    for idx, row in df.iterrows():
        photo_path = row["photo_path"]

        # Si la ruta guardada no existe (por cambio de ruta absoluta), buscar en local
        if not os.path.exists(photo_path):
            filename = str(row["image_id"]).strip()
            folder = "Image_train" if row["partition"] in ["train", "val"] else "Image_test"
            candidate = os.path.join(CURRENT_DIR, "Image Dataset", folder, filename)
            if not os.path.exists(candidate):
                base_name, _ = os.path.splitext(filename)
                for ext in [".jpg", ".png"]:
                    cand_ext = os.path.join(CURRENT_DIR, "Image Dataset", folder, base_name + ext)
                    if os.path.exists(cand_ext):
                        candidate = cand_ext
                        break
            photo_path = candidate

        if not os.path.exists(photo_path):
            skipped_count += 1
            continue

        # Lectura de imagen
        bgr = cv2.imread(photo_path)
        if bgr is None:
            skipped_count += 1
            continue

        h, w, _ = bgr.shape
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)

        if not results or not results.pose_landmarks:
            skipped_count += 1
            continue

        lms = results.pose_landmarks.landmark
        # Extraer keypoints necesarios para el recorte de torso
        # 11: left_shoulder, 12: right_shoulder, 2: left_eye, 5: right_eye
        keypoints = {
            "left_shoulder_x": lms[11].x * w,
            "left_shoulder_y": lms[11].y * h,
            "right_shoulder_x": lms[12].x * w,
            "right_shoulder_y": lms[12].y * h,
            "left_eye_y": lms[2].y * h,
            "right_eye_y": lms[5].y * h,
        }

        pil_img = Image.fromarray(rgb)
        try:
            cropped_torso = get_clean_torso_crop(pil_img, keypoints, w, h)
        except Exception:
            skipped_count += 1
            continue

        if cropped_torso.width < 30 or cropped_torso.height < 30:
            skipped_count += 1
            continue

        # Guardar recorte optimizado en JPEG de alta calidad
        crop_filename = f"crop_{os.path.basename(photo_path)}"
        if not crop_filename.lower().endswith(('.jpg', '.jpeg')):
            crop_filename = os.path.splitext(crop_filename)[0] + ".jpg"
        crop_path = os.path.join(crops_dir, crop_filename)
        cropped_torso.save(crop_path, format="JPEG", quality=95)

        record = dict(row)
        record["cropped_photo_path"] = os.path.abspath(crop_path)
        valid_records.append(record)

        if (idx + 1) % 250 == 0 or (idx + 1) == len(df):
            elapsed = time.time() - t0
            print(f"   [{idx + 1:4d}/{len(df)}] Recortadas: {len(valid_records)} | Descartadas: {skipped_count} | {elapsed:.1f}s", flush=True)

    pose_detector.close()

    df_out = pd.DataFrame(valid_records)
    df_out.to_csv(output_csv, index=False)

    print("\n" + "=" * 65)
    print("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 65)
    print(f"Total imágenes válidas listas para entrenar: {len(df_out)}")
    print(f"Total imágenes descartadas (sin pose clara): {skipped_count}")
    print(f"Dataset guardado en: {output_csv}")
    print("\nDesglose por partición:")
    print(df_out["partition"].value_counts())
    print("\nSiguiente paso: Ejecutar `python train_rtx3050.py`")


if __name__ == "__main__":
    run_torso_preprocessing()
