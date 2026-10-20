# Guía de Entrenamiento: Cómo Crear tu Propio Archivo de Pesos Verídicos para DigitalScale

Este documento describe paso a paso cómo recolectar tu propio dataset, procesarlo, estructurar los metadatos y entrenar el modelo de aprendizaje profundo desde cero o mediante ajuste fino (*Fine-Tuning*) para obtener un archivo de pesos verídico (`best_model.ckpt`) que funcione tanto en tu laptop (Python) como en tu smartphone (Android).

---

## 1. Planificación y Recolección de Datos

Para que el modelo aprenda a correlacionar características visuales con el Índice de Masa Corporal (IMC), necesitas construir un dataset estructurado. 

### Requisitos por Sujeto
Para cada participante en tu dataset, debes registrar:
1. **Fotografía**: Imagen de frente, con cuerpo completo o torso, idealmente con fondo neutro y ropa ajustada (la ropa muy holgada oculta la silueta y eleva el margen de error).
2. **Peso Real**: Registrado con precisión en kilogramos (kg) o libras (lb).
3. **Altura Real**: Registrada en metros (m) o pulgadas (in).
4. **Género**: Útil porque la distribución de grasa corporal y masa muscular varía de forma biológica entre hombres y mujeres.
5. **ID de Sujeto**: Identificador único de la persona (ej. `persona_001`). Esto es indispensable para agrupar fotos del mismo sujeto y evitar filtración de datos (*data leakage*) entre entrenamiento y prueba.

---

## 2. Preparación del Dataset y Estructura de Archivos

Para acoplar tu dataset con el cargador de datos del proyecto (`WaybetterDataset` en [densenet_dataloader.py](file:///j:/Documentos/Proyectos/1.-Proyectos%20GDIAC/DigitalScale/src/models/densenet/densenet_dataloader.py)), te sugerimos organizar los archivos de la siguiente manera:

```
mi_dataset/
├── imagenes/
│   ├── persona_001_foto1.jpg
│   ├── persona_001_foto2.jpg
│   ├── persona_002_foto1.jpg
│   └── ...
└── metadatos.csv
```

### Formato del Archivo `metadatos.csv`
El CSV debe contener, como mínimo, las siguientes columnas:
* `id`: Un identificador numérico o de texto único para cada imagen.
* `user_id`: El identificador de la persona (para control de splits).
* `photo`: La ruta relativa de la imagen (ej. `imagenes/persona_001_foto1.jpg`).
* `peso`: El peso en kg.
* `altura`: La altura en metros.
* `bmi`: El valor calculado del IMC:
  $$IMC = \frac{peso (kg)}{altura (m)^2}$$

---

## 3. Curación y Filtrado del Dataset Personalizado

Antes de entrenar, debes limpiar tus imágenes utilizando el mismo pipeline de calidad automática de DigitalScale para asegurar que el modelo no aprenda de fotos borrosas, lejanas o en poses incorrectas.

Puedes escribir un script de preparación (`preparar_dataset.py`) que ejecute estos pasos:

1. **Detección de Personas**: Filtra imágenes donde no se detecte un cuerpo con confianza $\ge 50\%$.
2. **Filtro de Ratio de Fondo**: Asegura que el cuerpo ocupe al menos el $25\%$ de la imagen (`bbox_area_ratio >= 0.25`).
3. **Filtro de Postura**:
   * Extrae los puntos clave COCO-17.
   * Carga los estimadores de postura (`scaler.pkl`, `pca.pkl`, `kmeans.pkl`).
   * Normaliza los puntos y predice el clúster. Conserva únicamente las fotos clasificadas en los clústeres válidos erguidos (clústeres 2 y 3).
4. **Recorte Corporal**: Recorta la imagen a cuerpo completo o torso (utilizando `crop_to_torso` definido en [torso_crop.py](file:///j:/Documentos/Proyectos/1.-Proyectos%20GDIAC/DigitalScale/src/helpers/torso_crop.py)).
5. **División de Datos sin Traslape de Usuarios**:
   * Divide tus datos en 3 particiones: Entrenamiento ($70\%$), Validación ($15\%$), y Prueba ($15\%$).
   * **Importante**: Asegúrate de que un usuario y sus fotos pertenezcan **únicamente a una sola partición** para evaluar la capacidad real de generalización del modelo ante personas desconocidas. Añade la columna `partition` (`train`, `val`, `test`) a tu CSV.

---

## 4. Script de Entrenamiento Personalizado (`entrenar_mi_modelo.py`)

Crea este script en la raíz de tu proyecto para iniciar el entrenamiento usando tu GPU o CPU. El script carga tu CSV filtrado, inicializa el dataloader, carga la arquitectura **SE-DenseNet-121** con transferencia de aprendizaje (pesos preentrenados de ImageNet) y realiza el ajuste fino de regresión del IMC.

```python
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.getcwd())

# Importar componentes de DigitalScale
from src.models.densenet.densenet_dataloader import get_dataloader
from src.models.densenet.densenet_trainer import Trainer
from src.models.densenet import densenet
from src.helpers.split_dataset import split_dataframe_without_user_overlap

def entrenar_modelo_veridico():
    csv_path = "mi_dataset/metadatos_filtrados.csv"
    photos_dir = "mi_dataset"
    save_dir = "trained_models/mi_modelo_veridico"
    epochs = 50
    batch_size = 16
    
    # 1. Cargar y Particionar Metadatos
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV en: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Agregar columna 'user_id' si no existe para agrupar splits
    if 'user_id' not in df.columns:
        df['user_id'] = df['id']
        
    df_splitted = split_dataframe_without_user_overlap(df, train_size=0.70, val_size=0.15)
    
    # 2. Inicializar Dataloaders
    print("Inicializando Dataloaders...")
    train_loader, val_loader, test_loader = get_dataloader(
        df_splitted,
        batch_size=batch_size,
        num_workers=2,
        photos_path=photos_dir
    )
    
    # 3. Inicializar Modelo con Aprendizaje por Transferencia (Transfer Learning)
    # Usamos SEDensenet121 por ser más rápido de entrenar en laptop
    print("Inicializando arquitectura SE-DenseNet-121 con pesos preentrenados de ImageNet...")
    model = densenet.model
    densenet.load_pretrained_densenet(model)
    
    # Reemplazar la capa lineal final para asegurar que prediga un solo valor de IMC
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1)
    )
    
    # 4. Configurar Dispositivo (GPU CUDA si está disponible, si no CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Entrenamiento configurado en dispositivo: {device}")
    
    # 5. Configurar Pérdida, Optimizador y Scheduler
    criterion = nn.MSELoss().to(device) # Mean Squared Error para regresión
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 6. Iniciar Bucle de Entrenamiento
    trainer = Trainer(model, device, optimizer, criterion, save_dir=save_dir, save_freq=10)
    print("Iniciando entrenamiento...")
    trainer.Loop(epochs, train_loader, val_loader, scheduler)
    
    print(f"✓ ¡Entrenamiento finalizado! Los pesos verídicos se han guardado en: {save_dir}/best_model.ckpt")

if __name__ == "__main__":
    entrenar_modelo_veridico()
```

---

## 5. Puntos Clave para Lograr una Alta Precisión

Para que tu archivo de pesos sea verdaderamente confiable y preciso, ten en cuenta los siguientes factores durante el entrenamiento:

### A. Tácticas de Data Augmentation
Dado que las imágenes caseras varían en luz y ángulo, aplica transformaciones aleatorias durante el entrenamiento para evitar el sobreajuste (*overfitting*):
* Rotaciones leves ($\pm 10^\circ$).
* Cambios aleatorios de brillo, contraste y saturación.
* Escalados y traslaciones pequeñas.
* *Nota*: Evita los volteos verticales (flip vertical) ya que la morfología humana tiene una orientación de gravedad fija.

### B. El Volumen del Dataset
* **Mínimo recomendado**: Al menos **1,000 imágenes** de diferentes individuos para obtener un modelo con un MAPE aceptable ($< 15\%$).
* Si entrenas con menos de 200 imágenes, el modelo memorizará las fotos de entrenamiento y fallará catastróficamente con nuevas caras en la webcam.

### C. Estrategia de Congelamiento de Capas (Fine-Tuning Gradual)
En lugar de entrenar todas las capas desde el principio:
1. **Fase 1 (Épocas 1-15)**: Congela todas las capas convolucionales base de DenseNet y entrena únicamente la capa lineal final (`classifier`). Esto estabilizará las predicciones iniciales del IMC.
2. **Fase 2 (Épocas 16-50)**: Descongela las capas convolucionales superiores (ej. el último DenseBlock) y entrena con una tasa de aprendizaje muy baja ($lr = 10^{-5}$) para adaptar el extractor visual a las sutilezas de las proporciones corporales humanas.

---

## 6. Integración Final de los Pesos Propios

Una vez que obtengas tu archivo `best_model.ckpt` verídico:

1. **Para usar en tu laptop (Webcam)**:
   * Coloca el archivo en la ruta `weights/best_model.ckpt`.
   * Ejecuta el script `webcam_demo.py` diseñado anteriormente.
2. **Para usar en Android (Celular)**:
   * Coloca el archivo en `trained_models/model_5/best_model.ckpt`.
   * Ejecuta el script de exportación: `python app/conversion/convert_model_executorch_fixed.py`.
   * Copia el archivo resultante `bmi_model_fixed.pte` al directorio de assets de tu aplicación Android.
