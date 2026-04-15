# ✋ Hand Tracking (MediaPipe + OpenCV)

Programa en Python ultraligero (<30 líneas) para detectar y renderizar el esqueleto de tus manos en tiempo real mediante tu cámara web, apoyado en el motor de IA de Google.

## 🛠️ Preparación
Es necesario cargar las librerías encapsuladas (OpenCV y MediaPipe) activando el entorno:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Ejecución y Uso
Arranca el programa con el siguiente comando básico:
```bash
python hand_tracking.py
```

*(⚠ **Nota para usuarios de Linux modernos:** Si tu escritorio utiliza Wayland y la ventana falla al abrirse o ves avisos persistentes relacionados con tipografías y `qt.qpa.plugin`, fuerza al renderizador a usar el modo clásico de ventanas de Linux con la instrucción de abajo):*
```bash
QT_QPA_PLATFORM=xcb python hand_tracking.py
```

## 🎮 Controles
* Mueve tus manos frente a la cámara en directo para revelar los nodos.
* Presiona **`q`** o **`ESC`** sobre la ventana de captura para cerrar la sesión y liberar la cámara.
