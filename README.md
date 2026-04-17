# 👁️ Vision Intelligence Suite

Una suite de visión artificial de alto rendimiento diseñada para la monitorización multimodal y el procesamiento de identidad en tiempo real.

![Status](https://img.shields.io/badge/Status-Optimized-brightgreen)
![NVIDIA](https://img.shields.io/badge/Hardware-NVIDIA_RTX_Ready-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

## ✨ Módulos Principales

### ✋ 1. Multimodal Gesture & Face Tracking (`hand_tracking.py`)
Monitorización avanzada de manos y expresiones faciales con una interfaz **HUD Premium**.
- **Seguimiento Hand-to-Gesture:** Conteo de dedos y detección de gestos especiales (como corazones).
- **Análisis de Blendshapes:** Reconoce 5 estados emocionales (Feliz, Sorprendido, Triste, etc.) en tiempo real.
- **Visualización Cyberpunk:** Renderizado técnico limpio con teselación facial minimalista.

### 🎭 2. Real-time Face Swap (`face_swap.py`)
Motor de intercambio de caras mediante Deepfakes optimizado para arquitecturas **NVIDIA (CUDA/TensorRT)**.
- **Identidad Fluida:** Inyecta la identidad de una imagen objetivo (`objetivo.jpg`) en la señal de vídeo.
- **Inferencia de Baja Latencia:** Ajustado para funcionar a altos FPS en tarjetas como la **RTX 5070 Ti**.

---

## 🛠️ Instalación

1. **Entorno Virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

2. **Dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Soporte GPU (Opcional)**:
   Si tienes una GPU NVIDIA, asegúrate de añadir las librerías CUDA al PATH (ver comentarios en `face_swap.py`).

## 🚀 Ejecución

### Para rastrear gestos y emociones:
```bash
python hand_tracking.py
```

### Para realizar intercambio de caras:
1. Coloca tu imagen de referencia como `objetivo.jpg`.
2. Asegúrate de tener el modelo `inswapper_128.onnx`.
3. Ejecuta:
   ```bash
   python face_swap.py
   ```

## 🎮 Controles de Ventana
- **`Q`** o **`ESC`**: Cierra la aplicación de forma segura.
- **Wayland (Linux)**: Usar `QT_QPA_PLATFORM=xcb` si la ventana falla al abrirse.

---
Desarrollado con ❤️ para la comunidad de Advanced AI Coding.
