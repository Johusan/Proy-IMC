# Análisis Técnico de Ejecución: Adaptación de DigitalScale para Webcam de Laptop

Este documento analiza en profundidad el código del repositorio [DigitalScale](https://github.com/im-ethz/DigitalScale) y detalla qué componentes faltan, qué limitaciones de software/hardware existen actualmente, y cómo se puede implementar un script para capturar video en tiempo real desde la webcam de una laptop y realizar la estimación del IMC (BMI).

---

## 1. Análisis de Limitaciones en el Código Actual

El repositorio de `DigitalScale` está diseñado para dos ambientes muy específicos:
1. **Entrenamiento y Evaluación Batch (Python/Linux/CUDA)**: Scripts para procesar el dataset WayBED o VisualBodyToBMI de manera estática usando GPUs de alto rendimiento.
2. **Aplicación Móvil (Android/TFLite/ExecuTorch)**: Una app de celular escrita en Kotlin que gestiona la cámara con CameraX y ejecuta modelos portados a formatos embebidos.

Al intentar ejecutar el código de inferencia en Python de forma directa en una laptop con Windows/macOS y usando la webcam, nos encontramos con los siguientes **bloqueos técnicos**:

### A. Dependencia Obligatoria de CUDA
En el archivo [bounding_box_detection.py](file:///j:/Documentos/Proyectos/1.-Proyectos%20GDIAC/DigitalScale/src/models/bounding_box_detection.py) (líneas 104-106), el constructor de la clase `BoundingBoxModule` fuerza el uso de GPU:
```python
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available")
```
Esto provoca que cualquier máquina sin una GPU NVIDIA compatible configurada con CUDA falle inmediatamente al importar o inicializar la clase.

### B. Dependencia de Detectron2 (Compilación Compleja)
Las clases principales de procesamiento de imágenes (`BoundingBoxModule`, `KeyPointDetectionModule` y `CropToPerson`) dependen de **Detectron2** de Facebook. 
* **El Problema**: Detectron2 no tiene soporte nativo oficial para Windows. Instalarlo requiere descargar herramientas de compilación C++ de Visual Studio (MSVC), compilar extensiones de CUDA y configurar variables de entorno complejas. No es viable para una ejecución simple "out-of-the-box" en la mayoría de laptops.

### C. Ausencia de un Script de Control de Cámara / Captura
No existe ningún script en el repositorio que interactúe con la webcam de la laptop (`cv2.VideoCapture`). El script de inferencia rápido en [predict_bmi.py](file:///j:/Documentos/Proyectos/1.-Proyectos%20GDIAC/DigitalScale/get_started/predict_bmi.py) solo está estructurado para leer un directorio de archivos de imágenes estáticas en el disco (`data_dir`).

### D. Falta de Pesos del Modelo de IMC
El archivo de pesos `best_model.ckpt` para la arquitectura **SE-DenseNet-201** o **SE-DenseNet-121** no está incluido en el repositorio por razones de tamaño. De acuerdo con el [README.md](file:///j:/Documentos/Proyectos/1.-Proyectos%20GDIAC/DigitalScale/README.md), se debe solicitar por correo electrónico a los autores (`rmanichand@ethz.ch` o `planger@ethz.ch`). Sin estos pesos, el modelo estimador de IMC no se puede inicializar.

---

## 2. Componentes Necesarios para Ejecutar con Webcam

Para ejecutar la balanza digital usando la webcam de tu laptop sin requerir GPUs NVIDIA o compilaciones complejas, se deben resolver las siguientes tareas:

```
[Webcam (OpenCV)] ──> [MediaPipe Pose (CPU)] ──> [Normalización & Posture Clustering]
                                                          │
                                                    ¿Pose Aprobada?
                                                          │
                                                    (Sí)  ▼
[Predicción de IMC] <── [SE-DenseNet (PyTorch CPU)] <── [Recorte de Torso / Cuerpo]
```

1. **Captura de Cámara en Tiempo Real**: Uso de **OpenCV** para abrir la webcam (`cv2.VideoCapture(0)`) y procesar cuadro por cuadro en un ciclo de ejecución continuo.
2. **Reemplazo de Detectron2 por MediaPipe (Multiplataforma)**:
   * **MediaPipe Pose** (desarrollado por Google) es extremadamente ligero, corre en CPU de laptop a más de 30 FPS y es muy fácil de instalar en Windows con un simple `pip install mediapipe`.
   * Proporciona de forma simultánea la **caja delimitadora** de la persona detectada y los **puntos clave** del esqueleto.
3. **Mapeo de Puntos Clave a COCO-17**:
   * El estimador de postura KMeans del proyecto (`kmeans.pkl`) espera 17 puntos clave normalizados siguiendo la estructura COCO.
   * MediaPipe extrae 33 puntos clave. Debemos seleccionar y mapear los 17 puntos COCO (ojos, hombros, caderas, codos, muñecas, rodillas, tobillos) y normalizarlos con respecto a la caja delimitadora de MediaPipe antes de pasárselos a los estimadores `scaler.pkl` y `pca.pkl`.
4. **Carga del Modelo SE-DenseNet en CPU**:
   * Modificar el cargador del estimador de IMC para que no intente usar CUDA obligatoriamente y cargue el checkpoint en CPU (`map_location='cpu'`).

---

## 3. Propuesta de Implementación: Script `webcam_demo.py`

A continuación, se detalla un diseño completo del script que debes crear en tu laptop para ejecutar la predicción. Este script utiliza **MediaPipe** (para reemplazar Detectron2 en CPU) e integra los modelos de clustering de postura y el estimador de PyTorch:

```python
import os
import cv2
import torch
import pickle
import numpy as np
import mediapipe as mp
from PIL import Image

# Importar las clases de modelo definidas en el repositorio
from get_started.model import SEDensenet201
from src.helpers.torso_crop import crop_to_torso, CropMargins

# Configuración de rutas y constantes
MODEL_WEIGHTS_PATH = "weights/best_model.ckpt"  # Debes descargar/solicitar este archivo
KEYPOINT_MODELS_DIR = "trained_models/keypoint_models/"
IMG_SIZE = 224

# 1. Cargar Estimadores de Postura (PCA, Scaler, KMeans)
with open(os.path.join(KEYPOINT_MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(KEYPOINT_MODELS_DIR, "pca.pkl"), "rb") as f:
    pca = pickle.load(f)
with open(os.path.join(KEYPOINT_MODELS_DIR, "kmeans.pkl"), "rb") as f:
    kmeans = pickle.load(f)

# 2. Inicializar Estimador de IMC en CPU
device = torch.device("cpu")
bmi_model = SEDensenet201()
if os.path.exists(MODEL_WEIGHTS_PATH):
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    bmi_model.load_state_dict(state_dict)
    bmi_model.to(device)
    bmi_model.eval()
    print("Modelo SE-DenseNet de IMC cargado con éxito en CPU.")
else:
    print(f"ATENCIÓN: No se encontraron pesos en {MODEL_WEIGHTS_PATH}. El IMC no se estimará.")
    bmi_model = None

# Mapeo de índices de MediaPipe Pose (33 landmarks) a COCO-17
# MediaPipe landmarks: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MP_TO_COCO_MAPPING = {
    0: 0,   # Nariz
    2: 1,   # Ojo izquierdo
    5: 2,   # Ojo derecho
    7: 3,   # Oreja izquierda
    8: 4,   # Oreja derecha
    11: 5,  # Hombro izquierdo
    12: 6,  # Hombro derecho
    13: 7,  # Codo izquierdo
    14: 8,  # Codo derecho
    15: 9,  # Muñeca izquierda
    16: 10, # Muñeca derecha
    23: 11, # Cadera izquierda
    24: 12, # Cadera derecha
    25: 13, # Rodilla izquierda
    26: 14, # Rodilla derecha
    27: 15, # Tobillo izquierdo
    28: 16  # Tobillo derecho
}

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Iniciar captura de webcam
cap = cv2.VideoCapture(0)
print("Presiona 'q' para salir. Presiona 's' para realizar la predicción de IMC cuando estés posicionado.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, c = frame.shape
    # MediaPipe requiere formato RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    status_text = "Buscando persona..."
    color_status = (0, 0, 255) # Rojo
    
    if results.pose_landmarks:
        # 3. Calcular Bounding Box a partir de los puntos clave
        x_coords = [lm.x * w for lm in results.pose_landmarks.landmark]
        y_coords = [lm.y * h for lm in results.pose_landmarks.landmark]
        
        x1, x2 = int(min(x_coords)), int(max(x_coords))
        y1, y2 = int(min(y_coords)), int(max(y_coords))
        
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_ratio = (bbox_w * bbox_h) / (w * h)
        
        # Dibujar bounding box de la persona
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Validar tamaño relativo de la persona en el recuadro
        if bbox_ratio < 0.25:
            status_text = "Póngase más cerca (Muy lejos)"
        else:
            # 4. Extraer y Mapear Keypoints a COCO-17
            coco_landmarks = np.zeros((17, 2))
            landmark_list = results.pose_landmarks.landmark
            
            for mp_idx, coco_idx in MP_TO_COCO_MAPPING.items():
                coco_landmarks[coco_idx] = [landmark_list[mp_idx].x * w, landmark_list[mp_idx].y * h]
            
            # Normalizar los puntos de la pose con respecto al bounding box de la persona
            normalized_features = []
            for px, py in coco_landmarks:
                norm_x = (px - x1) / bbox_w if bbox_w > 0 else 0
                norm_y = (py - y1) / bbox_h if bbox_h > 0 else 0
                normalized_features.extend([norm_x, norm_y])
            
            # 5. Evaluar Postura usando el modelo K-Means del proyecto
            X_scaled = scaler.transform([normalized_features])
            X_pca = pca.transform(X_scaled)
            posture_cluster = kmeans.predict(X_pca)[0]
            
            # El pipeline del repositorio dictamina que los clusters 2 y 3 (0-indexed 1 y 2) son posturas válidas
            # Nota: Verifica si tu entrenamiento sigue esta convención de índices.
            is_good_posture = posture_cluster in [2, 3] 
            
            if not is_good_posture:
                status_text = f"Postura Incorrecta (Cluster {posture_cluster})"
            else:
                status_text = f"Postura Correcta! Listo para medir (Cluster {posture_cluster})"
                color_status = (0, 255, 0) # Verde
                
                # Dibujar esqueleto verde
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # 6. Al presionar 's', ejecutamos el crop y la estimación de IMC
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    if bmi_model is None:
                        print("Error: No se puede estimar. Faltan los pesos del modelo (best_model.ckpt).")
                    else:
                        print("Procesando imagen corporal...")
                        # Preparar diccionario de keypoints requeridos por crop_to_torso
                        keypoint_dict = {
                            'left_shoulder_x': coco_landmarks[5][0],
                            'left_shoulder_y': coco_landmarks[5][1],
                            'right_shoulder_x': coco_landmarks[6][0],
                            'right_shoulder_y': coco_landmarks[6][1],
                            'left_eye_y': coco_landmarks[1][1],
                            'right_eye_y': coco_landmarks[2][1],
                        }
                        
                        pil_img = Image.fromarray(rgb_frame)
                        # Recortar al área del torso
                        cropped_torso = crop_to_torso(pil_img, keypoint_dict)
                        
                        # Preprocesamiento para SE-DenseNet (Normalización ImageNet)
                        preprocess = transforms.Compose([
                            transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        
                        input_tensor = preprocess(cropped_torso).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            predicted_bmi = bmi_model(input_tensor).item()
                        
                        print(f"🚀 IMC Estimado: {predicted_bmi:.2f}")
                        # Mostrar el resultado final congelado en una ventana
                        cv2.imshow("Recorte de Inferencia", cv2.cvtColor(np.array(cropped_torso), cv2.COLOR_RGB2BGR))
                        cv2.waitKey(0)

    # Mostrar overlay del estado
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
    cv2.imshow("DigitalScale Webcam Demo", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 4. Plan de Acción para Ejecutar en tu Laptop

Para poner en marcha esta solución en tu entorno local, sigue estos pasos:

1. **Instalar Dependencias de Python**:
   Ejecuta lo siguiente en la terminal de tu laptop para instalar librerías ligeras multiplataforma:
   ```bash
   pip install opencv-python mediapipe torch torchvision scikit-learn pydantic
   ```
2. **Conseguir el Archivo de Pesos**:
   * Envía un correo a `rmanichand@ethz.ch` o `planger@ethz.ch` pidiendo el archivo `best_model.ckpt` (modelo SE-DenseNet entrenado en WayBED).
   * Almacénalo en una nueva carpeta dentro del repositorio llamada `weights/` (`DigitalScale/weights/best_model.ckpt`).
3. **Crear y Ejecutar el Script**:
   * Guarda el código anterior como `webcam_demo.py` en la raíz del repositorio (`j:\Documentos\Proyectos\1.-Proyectos GDIAC\DigitalScale\webcam_demo.py`).
   * Ejecútalo mediante:
     ```bash
     python webcam_demo.py
     ```
   * Párate frente a la cámara de tal forma que capte tu torso y cabeza. Cuando el texto superior cambie a color verde indicando **"Postura Correcta!"**, presiona la tecla **`s`** para ver el resultado de tu IMC.
