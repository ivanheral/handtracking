# 👁️ Vision Intelligence Suite (Hand Tracking & Face Swap)

Este repositorio contiene un conjunto de herramientas de visión artificial de alto rendimiento desarrolladas en Python, utilizando **MediaPipe** para el seguimiento multimodal y **InsightFace** para Deepfakes en tiempo real.

---

## 🚀 Módulos Principales

### 1. 🧤 Multimodal Gesture & Face Tracking (`hand_tracking.py`)
Un sistema ligero que monitoriza simultáneamente manos y expresiones faciales.
- **Detección de Manos:** Cuenta dedos y detecta gestos específicos (como el corazón coreano).
- **Expresiones Faciales:** Reconoce estados de ánimo (Feliz, Sorprendido, Triste, Ojos cerrados) mediante blendshapes.
- **Estética:** Renderizado en blanco y negro con nodos minimalistas para una visualización técnica limpia.

### 2. 🎭 Real-time Face Swap (`face_swap.py`)
Un motor de intercambio de caras (Deepfake) optimizado para **NVIDIA RTX**.
- **Inyección de Identidad:** Suplanta tu cara por la de cualquier imagen objetivo (`objetivo.jpg`).
- **Aceleración GPU:** Utiliza núcleos CUDA y TensorRT para lograr inferencia de baja latencia.
- **Poder NVIDIA:** Optimizado específicamente para sacar provecho de tarjetas como la **RTX 5070 Ti**.

---

## 🛠️ Instalación y Configuración

### 1. Preparar el entorno (Linux)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Soporte para GPU NVIDIA (Opcional pero recomendado)
Si tienes una tarjeta NVIDIA, el archivo `requirements.txt` ya incluye el soporte de CUDA. Para habilitarlo correctamente en Linux, asegúrate de añadir las librerías al PATH:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/venv/lib/python3.14/site-packages/nvidia/cudnn/lib:$(pwd)/venv/lib/python3.14/site-packages/nvidia/cublas/lib:$(pwd)/venv/lib/python3.14/site-packages/nvidia/cuda_runtime/lib:$(pwd)/venv/lib/python3.14/site-packages/nvidia/curand/lib:$(pwd)/venv/lib/python3.14/site-packages/nvidia/cufft/lib:$(pwd)/venv/lib/python3.14/site-packages/nvidia/cuda_nvrtc/lib:$(pwd)/venv/lib/python3.14/site-packages/nvidia/nvjitlink/lib:$(pwd)/venv/lib/python3.14/site-packages/nvidia/cusolver/lib:$(pwd)/venv/lib/python3.14/site-packages/nvidia/cusparse/lib
```

---

## 🎮 Ejecución

### Rastrear gestos y emociones:
```bash
python hand_tracking.py
```

### Realizar intercambio de caras:
1. Coloca una foto de la cara que quieres usar como `objetivo.jpg` en la raíz.
2. Asegúrate de tener el modelo `inswapper_128.onnx`.
3. Ejecuta:
```bash
python face_swap.py
```

---

## ⌨️ Controles
- **Q / ESC:** Cierra la ventana y libera la cámara de forma segura.
- **Modo Wayland:** Si la ventana no abre en Linux moderno, ejecuta con `QT_QPA_PLATFORM=xcb python <script>.py`.

---

## 📦 Dependencias Clave
- `opencv-contrib-python`: Visión y GUI.
- `insightface`: Motor de Deepfake.
- `onnxruntime-gpu`: Inferencia acelerada por hardware.
- `mediapipe`: Seguimiento de puntos clave.
