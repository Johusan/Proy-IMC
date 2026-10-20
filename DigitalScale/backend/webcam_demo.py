import os
import sys
import time
import cv2
import pickle
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Forzar salida en consola sin bufer
sys.stdout.reconfigure(line_buffering=True)

# Anadir la ruta raiz del backend a sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Importar definicion del modelo y funcion de recorte del repositorio
from src.models.densenet.densenet import SEDensenet121
from src.helpers.torso_crop import crop_to_torso

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# Duracion de la cuenta regresiva en segundos para la captura automatica
AUTO_CAPTURE_SECONDS = 3.0

# Mapeo de 33 Puntos Anatomicos de MediaPipe Pose a los 17 Keypoints de COCO
MP_TO_COCO = {
    0: 0,   # Nariz
    2: 1,   # Ojo Izquierdo
    5: 2,   # Ojo Derecho
    7: 3,   # Oreja Izquierda
    8: 4,   # Oreja Derecha
    11: 5,  # Hombro Izquierdo
    12: 6,  # Hombro Derecho
    13: 7,  # Codo Izquierdo
    14: 8,  # Codo Derecho
    15: 9,  # Muneca Izquierda
    16: 10, # Muneca Derecha
    23: 11, # Cadera Izquierda
    24: 12, # Cadera Derecha
    25: 13, # Rodilla Izquierda
    26: 14, # Rodilla Derecha
    27: 15, # Tobillo Izquierdo
    28: 16  # Tobillo Derecho
}

def load_mediapipe():
    """Cargador robusto de MediaPipe Pose compatible entre diferentes versiones."""
    import mediapipe as mp
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
    except AttributeError:
        try:
            import mediapipe.python.solutions.pose as mp_pose
            import mediapipe.python.solutions.drawing_utils as mp_drawing
        except ImportError:
            import mediapipe.solutions.pose as mp_pose
            import mediapipe.solutions.drawing_utils as mp_drawing
    return mp_pose, mp_drawing

WINDOW_NAME = "DigitalScale - Demo de Estimacion de IMC con Camara"

def show_responsive_window(window_name, frame):
    """
    Renderiza el fotograma de forma adaptable dentro de una ventana de OpenCV.
    Mantiene la relacion de aspecto y centra el video sobre un fondo oscuro al redimensionar.
    """
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
        rect = cv2.getWindowImageRect(window_name)
    except Exception:
        cv2.imshow(window_name, frame)
        return True

    win_w, win_h = rect[2], rect[3]
    frame_h, frame_w = frame.shape[:2]

    if win_w > 100 and win_h > 100:
        scale = min(win_w / frame_w, win_h / frame_h)
        scaled_w = max(1, int(frame_w * scale))
        scaled_h = max(1, int(frame_h * scale))

        resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Crear lienzo oscuro con las dimensiones actuales de la ventana
        canvas = np.full((win_h, win_w, 3), 26, dtype=np.uint8)

        # Centrar el fotograma redimensionado en el lienzo
        y_offset = (win_h - scaled_h) // 2
        x_offset = (win_w - scaled_w) // 2
        canvas[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized

        cv2.imshow(window_name, canvas)
    else:
        cv2.imshow(window_name, frame)
    return True

def get_clean_torso_crop(pil_img, keypoints, img_w, img_h, margin_factor=1.4):
    """
    Extrae un recorte limpio del torso acotado estrictamente a las dimensiones de la imagen,
    evitando que PIL inserte bordes negros artificiales fuera de los limites de la camara.
    """
    shoulder_w = abs(keypoints['right_shoulder_x'] - keypoints['left_shoulder_x'])
    if shoulder_w < 10:
        shoulder_w = img_w * 0.25

    margin = shoulder_w * (margin_factor - 1.0)
    
    min_sx = min(keypoints['left_shoulder_x'], keypoints['right_shoulder_x'])
    max_sx = max(keypoints['left_shoulder_x'], keypoints['right_shoulder_x'])
    min_ey = min(keypoints['left_eye_y'], keypoints['right_eye_y'])
    max_sy = max(keypoints['left_shoulder_y'], keypoints['right_shoulder_y'])

    # Limites de la region del torso
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

    # Desplazar la caja si sobrepasa los bordes para capturar pixeles reales
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

    # Ajuste estricto para evitar rellenos fuera de rango
    final_x1 = max(0, min(img_w - 2, int(round(x1))))
    final_x2 = max(final_x1 + 10, min(img_w, int(round(x2))))
    final_y1 = max(0, min(img_h - 2, int(round(y1))))
    final_y2 = max(final_y1 + 10, min(img_h, int(round(y2))))

    return pil_img.crop((final_x1, final_y1, final_x2, final_y2))

def open_working_webcam():
    """Escanea indices de camara (1, 0, 2) con el backend DirectShow en Windows."""
    indices_to_try = [1, 0, 2]
    for idx in indices_to_try:
        print(f"[INFO] Probando camara indice {idx} con DirectShow...", flush=True)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[OK] Conexion exitosa con la camara en indice {idx}.", flush=True)
                    return cap
        cap.release()
    print("[INFO] Usando dispositivo de video por defecto VideoCapture(0)...", flush=True)
    return cv2.VideoCapture(0)

def main():
    print("=" * 60, flush=True)
    print("DIGITAL SCALE - ESTIMACION DE IMC POR VISION ARTIFICIAL", flush=True)
    print("=" * 60, flush=True)

    # 1. Configuracion del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo de inferencia: {device}", flush=True)

    # 2. Rutas de modelos
    weights_path = os.path.join(CURRENT_DIR, "weights", "best_model.ckpt")
    if not os.path.exists(weights_path):
        alt_candidates = [
            os.path.join(CURRENT_DIR, "trained_models", "model_5", "best_model.ckpt"),
            os.path.abspath(os.path.join(CURRENT_DIR, "..", "Entrenar_Data", "output", "best_model.ckpt"))
        ]
        for candidate in alt_candidates:
            if os.path.exists(candidate):
                weights_path = candidate
                break

    keypoint_models_dir = os.path.join(CURRENT_DIR, "trained_models", "keypoint_models")

    # 3. Cargar modelos de postura (Scaler, PCA, KMeans)
    scaler_path = os.path.join(keypoint_models_dir, "scaler.pkl")
    pca_path = os.path.join(keypoint_models_dir, "pca.pkl")
    kmeans_path = os.path.join(keypoint_models_dir, "kmeans.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(pca_path) and os.path.exists(kmeans_path)):
        print(f"[ERROR] Modelos de agrupamiento de postura no encontrados en: {keypoint_models_dir}", flush=True)
        return

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    print("[OK] Modelos de postura (PCA + KMeans) cargados correctamente.", flush=True)

    # 4. Cargar red neuronal de estimacion de IMC
    bmi_model = SEDensenet121()
    if os.path.exists(weights_path):
        print(f"[OK] Cargando pesos del modelo desde: {weights_path}", flush=True)
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        bmi_model.load_state_dict(state_dict)
        bmi_model.to(device)
        bmi_model.eval()
        print("[OK] Modelo SE-DenseNet-121 listo para inferencia.", flush=True)
    else:
        print(f"[ERROR] Archivo de pesos no encontrado en: {weights_path}", flush=True)
        print("   -> Por favor coloque 'best_model.ckpt' en 'DigitalScale/weights/'", flush=True)
        print("   -> Consulte el README.md para obtener el enlace de descarga.", flush=True)
        return

    # Desactivar bandera de memoria eficiente para inferencia estable
    for m in bmi_model.modules():
        if hasattr(m, 'memory_efficient'):
            m.memory_efficient = False

    # 5. Inicializar MediaPipe Pose
    mp_pose, mp_drawing = load_mediapipe()
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 6. Pipeline de transformaciones para PyTorch
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    # 7. Iniciar captura de video
    cap = open_working_webcam()
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir ningun dispositivo de camara.", flush=True)
        return

    # 8. Configurar ventana de visualizacion redimensionable
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    print("\n[INFO] Iniciando transmision de video en vivo...", flush=True)
    print("  - Mantenga una postura erguida por 3 segundos para CAPTURA AUTOMATICA.")
    print("  - Presione 's' para capturar manualmente.")
    print("  - Presione 'r' para reiniciar despues de una medicion.")
    print("  - Presione 'q' o cierre la ventana para salir.\n", flush=True)

    good_posture_start_time = None
    last_captured_bmi = None
    captured_result_image = None

    while cap.isOpened():
        # Comprobar si el usuario cerro la ventana
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Ventana cerrada por el usuario.", flush=True)
            break

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        status_text = "Buscando persona..."
        color_status = (0, 0, 255) # Rojo
        should_trigger = False
        remaining_sec = 0

        key = cv2.waitKey(1) & 0xFF

        # Si ya se tiene un resultado capturado, mostrar la pantalla de resultado
        if captured_result_image is not None:
            display_frame = frame.copy()

            # Superposicion oscura semitransparente
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (w - 220, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

            cv2.putText(display_frame, f"IMC ESTIMADO: {last_captured_bmi:.2f} kg/m2", (20, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "Presione 'r' para medir de nuevo | 'q' para salir", (20, 92), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

            # Dibujar miniatura del torso recortado en la esquina superior derecha (180x180)
            thumb = cv2.resize(captured_result_image, (180, 180))
            display_frame[20:200, w-200:w-20] = thumb
            cv2.rectangle(display_frame, (w-200, 20), (w-20, 200), (0, 255, 0), 2)

            show_responsive_window(WINDOW_NAME, display_frame)
            if key == ord('r'):
                captured_result_image = None
                last_captured_bmi = None
                good_posture_start_time = None
                print("[INFO] Reiniciando para una nueva medicion...", flush=True)
            elif key == ord('q'):
                break
            continue

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calcular caja delimitadora a partir de los landmarks
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            
            x1, x2 = int(max(0, min(x_coords))), int(min(w, max(x_coords)))
            y1, y2 = int(max(0, min(y_coords))), int(min(h, max(y_coords)))
            
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            bbox_ratio = (bbox_w * bbox_h) / (w * h)

            # Mapear MediaPipe -> COCO-17
            coco_landmarks = np.zeros((17, 2))
            for mp_idx, coco_idx in MP_TO_COCO.items():
                coco_landmarks[coco_idx] = [landmarks[mp_idx].x * w, landmarks[mp_idx].y * h]

            # Verificacion de encuadre y proximidad
            shoulder_span = abs(coco_landmarks[5][0] - coco_landmarks[6][0])
            touching_border = (x1 < 15) or (x2 > w - 15) or (y1 < 10)
            is_too_close = (bbox_ratio > 0.55) or (shoulder_span > w * 0.45) or touching_border

            # Dibujar caja de la persona
            color_bbox = (0, 165, 255) if is_too_close else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bbox, 2)

            if is_too_close:
                status_text = "DEMASIADO CERCA: Retroceda para enfocar el torso"
                color_status = (0, 165, 255) # Advertencia naranja
                good_posture_start_time = None
            elif bbox_ratio < 0.15:
                status_text = f"Muy lejos de la camara (Proporcion: {bbox_ratio:.2f})"
                color_status = (0, 0, 255) # Rojo
                good_posture_start_time = None
            else:
                # Normalizar puntos COCO respecto a la caja delimitadora
                normalized_kpts = []
                for px, py in coco_landmarks:
                    norm_x = (px - x1) / bbox_w
                    norm_y = (py - y1) / bbox_h
                    normalized_kpts.extend([norm_x, norm_y])

                # Clasificar cluster de postura con KMeans
                X_scaled = scaler.transform([normalized_kpts])
                X_pca = pca.transform(X_scaled)
                posture_cluster = kmeans.predict(X_pca)[0]

                # Clusters 0, 1, 2, 3 corresponden a posturas erguidas aceptables
                is_valid_pose = posture_cluster in [0, 1, 2, 3]

                if not is_valid_pose:
                    status_text = f"Postura Incorrecta (Cluster {posture_cluster})"
                    good_posture_start_time = None
                else:
                    color_status = (0, 255, 0) # Verde
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Actualizar temporizador de postura correcta
                    if good_posture_start_time is None:
                        good_posture_start_time = time.time()

                    elapsed_good_posture = time.time() - good_posture_start_time
                    remaining_sec = max(0, int(np.ceil(AUTO_CAPTURE_SECONDS - elapsed_good_posture)))

                    if remaining_sec > 0:
                        status_text = f"POSTURA CORRECTA. Mantenga la posicion... {remaining_sec}s"
                        cv2.putText(frame, f"CAPTURANDO EN {remaining_sec}...", (w//2 - 200, 100), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    else:
                        status_text = "CAPTURANDO AUTOMATICAMENTE"
                        should_trigger = True

        else:
            good_posture_start_time = None

        if key == ord('s'):
            should_trigger = True

        if should_trigger and results and results.pose_landmarks:
            print("[INFO] Captura activada. Estimando IMC...", flush=True)
            
            keypoint_dict = {
                'left_shoulder_x': coco_landmarks[5][0],
                'left_shoulder_y': coco_landmarks[5][1],
                'right_shoulder_x': coco_landmarks[6][0],
                'right_shoulder_y': coco_landmarks[6][1],
                'left_eye_y': coco_landmarks[1][1],
                'right_eye_y': coco_landmarks[2][1],
            }
            
            pil_img = Image.fromarray(rgb_frame)
            try:
                cropped_img = get_clean_torso_crop(pil_img, keypoint_dict, w, h)
            except Exception as e:
                print(f"[WARN] Recorte con fallback: {e}", flush=True)
                fx1, fy1 = max(0, x1), max(0, y1)
                fx2, fy2 = min(w, x2), min(h, y2)
                cropped_img = pil_img.crop((fx1, fy1, fx2, fy2))

            input_tensor = preprocess(cropped_img).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_bmi = bmi_model(input_tensor).item()

            print(f"[RESULTADO] IMC ESTIMADO: {predicted_bmi:.2f} kg/m2", flush=True)

            last_captured_bmi = predicted_bmi
            captured_result_image = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            good_posture_start_time = None

        # Dibujar estado con fondo oscuro para legibilidad
        (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (15, 15), (25 + text_w, 25 + text_h + 10), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (20, 20 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)

        show_responsive_window(WINDOW_NAME, frame)

        if key == ord('q'):
            print("[INFO] Saliendo de la aplicacion...", flush=True)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
