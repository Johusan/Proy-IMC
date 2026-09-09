# Análisis de Viabilidad: Generación y Estructura de Pesos Mock para el Modelo de IMC

Este documento aborda en profundidad la problemática del archivo de pesos del modelo de estimación del IMC (`best_model.ckpt`), analiza su estructura interna, y evalúa la viabilidad técnica de generar un archivo de pesos de prueba ("mock") para desbloquear la ejecución del pipeline tanto en Python (laptop) como en la aplicación móvil (Android).

---

## 1. Estructura y Anatomía del Archivo de Pesos `best_model.ckpt`

El archivo de pesos del proyecto es un archivo serializado utilizando el módulo pickle de Python a través de la función `torch.save()`. No contiene únicamente los pesos numéricos del modelo, sino un diccionario con metadatos del estado del entrenamiento.

### Anatomía del Checkpoint en DigitalScale
Al inspeccionar el código de guardado en [densenet_trainer.py](file:///j:/Documentos/Proyectos/1.-Proyectos%20GDIAC/DigitalScale/src/models/densenet/densenet_trainer.py) (líneas 141-148), vemos que el diccionario guardado tiene la siguiente estructura:

```python
state = {
    "epoch": epoch,                 # Entero (int): Época en la que se guardó el modelo
    "state_dict": MODEL.state_dict(), # OrderedDict: Pesos y sesgos de todas las capas
    "MAE": self.best_error,         # Float: Error Absoluto Medio en validación (ej. 2.56)
    "MAPE": mape.avg,               # Float: Error Porcentual Absoluto Medio (ej. 0.079)
    "optimizer": self.optimizer.state_dict() # Dict: Parámetros del optimizador Adam
}
```

### El Estado de las Capas (`state_dict`)
El componente crucial es el `"state_dict"`. Para la arquitectura **SE-DenseNet-121** o **SE-DenseNet-201** definida en el proyecto, este diccionario contiene tensores para:
1. **Capa Inicial (conv0)**: Filtro de convolución de entrada de 3 canales a 64 canales.
2. **DenseBlocks (1 a 4)**: Cientos de capas de Normalización por Lote (BatchNorm), ReLU y Convoluciones de $1\times1$ y $3\times3$.
3. **Módulos Squeeze-and-Excitation (SELayer)**: Capas lineales para la atención por canal (`fc.0.weight`, `fc.2.weight`).
4. **Clasificador Final**: Una sola capa lineal de regresión (`classifier.0.weight` y `classifier.0.bias`) que toma las características promediadas y genera una única salida escalar (el IMC estimado).

---

## 2. Análisis de Viabilidad de Pesos "Mock" (Falsos)

### ¿Es viable crear un archivo de pesos falso para ejecutar el programa?
**Sí, es 100% viable.** 

Sin embargo, se debe entender la diferencia entre la **viabilidad de ejecución** y la **viabilidad de precisión**:
* **Viabilidad de Ejecución (Alta)**: Crear un archivo con la misma estructura exacta de diccionarios y tensores inicializados de forma aleatoria o preentrenada (ImageNet) permitirá que el cargador de PyTorch inicialice el modelo y realice pasadas hacia adelante (*forward pass*) en la laptop y en Android sin arrojar excepciones.
* **Viabilidad de Precisión (Nula)**: El modelo predecirá valores de IMC absurdos o aleatorios (por ejemplo, valores negativos o extremadamente altos) porque la última capa lineal no ha sido entrenada con datos reales de la relación peso/altura de WayBED.

### ¿Para qué sirve generar pesos Mock?
1. **Desbloqueo de Integración**: Validar que la interfaz de la webcam capta los frames, extrae los keypoints, evalúa la postura, recorta la imagen y alimenta el tensor al modelo sin colapsar.
2. **Prueba de Exportación (Android/ExecuTorch)**: El script de conversión a ExecuTorch [convert_model_executorch_fixed.py](file:///j:/Documentos/Proyectos/1.-Proyectos%20GDIAC/DigitalScale/app/conversion/convert_model_executorch_fixed.py) requiere abrir un archivo `.ckpt` válido para trazar la gráfica del modelo (`torch.export`) y generar el archivo `.pte` para Android. Un archivo mock permite compilar y testear la aplicación Android de inmediato.
3. **Verificación de UI**: Permite verificar que los overlays de texto y los gráficos de resultados en la app móvil reaccionan tras una inferencia exitosa.

---

## 3. Script para Generar Pesos Mock (`generate_mock_weights.py`)

Puedes crear y ejecutar el siguiente script de Python en la raíz de tu workspace para autogenerar un archivo `best_model.ckpt` compatible. Este script utiliza la arquitectura del repositorio e inicializa las capas base con pesos preentrenados de ImageNet (para darle coherencia matemática a la extracción de características) e inicializa la capa final de regresión con valores razonables.

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Asegurar que el entorno de DigitalScale está accesible
import sys
sys.path.append(os.getcwd())

# Importar el modelo del repositorio
from src.models.densenet.densenet import SEDensenet121, load_pretrained_densenet

def generate_mock_weights():
    print("Inicializando arquitectura SE-DenseNet-121...")
    model = SEDensenet121()
    
    print("Cargando pesos preentrenados de ImageNet en la base convolucional...")
    # Esto inicializa la red con pesos reales de visión general (ayuda a evitar valores NaN)
    load_pretrained_densenet(model)
    
    # Inicializar la capa lineal final con pesos pequeños aleatorios para evitar IMCs extremos
    print("Inicializando regresor final de IMC...")
    nn.init.normal_(model.classifier[0].weight, mean=0.0, std=0.01)
    nn.init.constant_(model.classifier[0].bias, 22.0)  # Bias inicializado en 22.0 (IMC saludable promedio)
    
    # Crear un optimizador ficticio para poblar el diccionario del checkpoint
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Construir la estructura exacta del checkpoint que densenet_trainer espera
    checkpoint_state = {
        "epoch": 0,
        "state_dict": model.state_dict(),
        "MAE": 100.0,  # Valor de error dummy
        "MAPE": 1.0,   # Valor de error dummy
        "optimizer": optimizer.state_dict()
    }
    
    # Crear directorios si no existen
    os.makedirs("weights", exist_ok=True)
    os.makedirs("trained_models/model_5", exist_ok=True)
    
    # Guardar en las ubicaciones esperadas por el script de webcam y el de conversión de Android
    torch.save(checkpoint_state, "weights/best_model.ckpt")
    torch.save(checkpoint_state, "trained_models/model_5/best_model.ckpt")
    
    print("\n✓ ¡Archivo de pesos Mock generado con éxito!")
    print("  - weights/best_model.ckpt (Para demo de webcam)")
    print("  - trained_models/model_5/best_model.ckpt (Para conversión a ExecuTorch)")

if __name__ == "__main__":
    generate_mock_weights()
```

---

## 4. Portabilidad a Android (Conversión a ExecuTorch `.pte`)

Una vez que generas el archivo `.ckpt` de prueba, el pipeline de Android se puede alimentar siguiendo estos pasos:

1. **Compilar el archivo `.pte` para ExecuTorch**:
   Ejecuta el script de conversión proporcionado en el repositorio:
   ```bash
   cd app/conversion
   python convert_model_executorch_fixed.py
   ```
   *Este script leerá `trained_models/model_5/best_model.ckpt` (nuestro archivo mock), resolverá las dependencias y guardará `bmi_model_fixed.pte` en `android/app/src/main/assets/`*.

2. **Ejecutar en la App Móvil**:
   * Cuando la cámara detecte a la persona y confirme postura erguida mediante el JSON de posturas (`posture_model_params.json`), enviará la imagen procesada al runtime de ExecuTorch.
   * El modelo `.pte` mock procesará el tensor y devolverá un valor de IMC estable y no nulo (cercano a 22.0 debido al bias configurado en nuestro script), verificando el correcto acoplamiento de todo el ecosistema de hardware y software del celular.

---

## 5. Puntos Críticos y Advertencias a Tener en Cuenta

* **Sincronización de Arquitecturas**: El repositorio cuenta con dos arquitecturas: `SEDensenet121` (base) y `SEDensenet201` (grande). Si el script de la webcam en Python carga `SEDensenet201` pero los pesos mock se generaron con `SEDensenet121`, el programa fallará por incompatibilidad de llaves y formas en el `state_dict`. Asegúrate de usar la misma clase de modelo en la generación, en el pipeline de Python y en el script de conversión a ExecuTorch.
* **Trazado de ExecuTorch**: ExecuTorch utiliza `torch.export` que requiere flujos de control estáticos. Es indispensable que al generar los pesos mock no introduzcas valores nulos (`NaN` o `Inf`) que puedan corromper los trazadores lógicos durante la conversión.
* **El Bias del Regresor**: Para pruebas de interfaz de usuario más realistas en Android, el bias del regresor final en el script mock se puede inicializar en un valor estándar de IMC saludable (ej. `22.0`). Esto evitará que la interfaz de la aplicación móvil muestre errores de desbordamiento de UI o IMCs de cero.
