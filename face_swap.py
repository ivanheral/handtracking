import cv2
import sys
import time
import os
import signal
import insightface
from insightface.app import FaceAnalysis
import numpy as np

# Parche de compatibilidad para NVIDIA en Windows (Soluciona DLLs faltantes)
if os.name == 'nt':
    import site
    possible_paths = []
    # Buscar en el sitio de paquetes del usuario y del venv
    for s in [site.getusersitepackages()] + site.getsitepackages():
        nvidia_base = os.path.join(s, 'nvidia')
        if os.path.exists(nvidia_base):
            for folder in ['cuda_runtime/bin', 'cublas/bin', 'cudnn/bin', 'curand/bin', 'cufft/bin']:
                full_path = os.path.join(nvidia_base, folder)
                if os.path.exists(full_path):
                    possible_paths.append(full_path)
    
    if possible_paths:
        print(f"[IA] Librerías NVIDIA encontradas: {len(possible_paths)} carpetas añadidas al sistema.")
        os.environ['PATH'] = os.pathsep.join(possible_paths) + os.pathsep + os.environ['PATH']
        for path in possible_paths:
            try: os.add_dll_directory(path)
            except: pass

# Configuración Estética (Tech Gold)
COLOR_TEXTO = (255, 255, 255)
COLOR_ACCENT = (0, 215, 255) # Dorado
COLOR_BG = (10, 10, 10)

def main():
    print(f"{'='*50}")
    print(" VISION INTELLIGENCE: FACE SWAP CORE v2.0 ")
    print(" Optimizado para NVIDIA RTX (5070 Ti Ready) ")
    print(f"{'='*50}")

    target_image_path = 'objetivo.jpg'
    model_path = 'inswapper_128.onnx'
    
    if not os.path.exists(target_image_path):
        print(f"[ERROR] Imagen '{target_image_path}' no encontrada.")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"[AVISO] Modelo '{model_path}' no encontrado.")
        print("[IA] Iniciando descarga automática desde Hugging Face (aprox. 554MB)...")
        import urllib.request
        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        try:
            def progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = (block_num * block_size * 100) / total_size
                    sys.stdout.write(f"\r[IA] Progreso de descarga: {percent:.1f}%")
                    sys.stdout.flush()
            urllib.request.urlretrieve(url, model_path, progress)
            print("\n[OK] Modelo descargado y listo.")
        except Exception as e:
            print(f"\n[ERROR] No se pudo descargar el modelo: {e}")
            sys.exit(1)

    # Configuración de aceleración de hardware NVIDIA
    # Eliminamos TensorRT para evitar conflictos de DLLs y usamos CUDA 100% puro.
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
        }),
        'CPUExecutionProvider'
    ]

    print("[IA] Cargando motores de inferencia en GPU (CUDA Only)...")
    try:
        face_analyzer = FaceAnalysis(name='buffalo_l', providers=providers)
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        swapper = insightface.model_zoo.get_model(model_path, providers=providers)
    except Exception as e:
        print(f"[ERROR] Error al inicializar GPU: {e}")
        return

    # Proceso de identidad objetivo
    print(f"[IA] Sincronizando identidad: {target_image_path}...")
    target_img = cv2.imread(target_image_path)
    target_faces = face_analyzer.get(target_img)
    
    if not target_faces:
        print("[ERROR] No se detectó rostro en la imagen objetivo.")
        sys.exit(1)
    
    target_identity = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    print("[IA] Identidad inyectada con éxito. RTX ON.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def clean(*_): 
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Desconectando redes neuronales. Sistema OFF.")
        sys.exit(0)

    signals = [signal.SIGINT]
    if hasattr(signal, "SIGTSTP"): signals.append(signal.SIGTSTP)
    for sig in signals: signal.signal(sig, clean)

    cv2.namedWindow("NVIDIA RTX Power: Face Swap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NVIDIA RTX Power: Face Swap", 1280, 720)
    
    t_prev = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        faces = face_analyzer.get(frame)
        
        if faces:
            main_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            try:
                frame = swapper.get(frame, main_face, target_identity, paste_back=True)
            except:
                pass

        t_curr = time.time()
        fps = 1/(t_curr - t_prev) if (t_curr - t_prev) > 0 else 0
        t_prev = t_curr

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (250, 60), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.line(frame, (0, 60), (250, 60), COLOR_ACCENT, 2)
        
        cv2.putText(frame, f"RTX ON | FPS: {fps:.1f}", (20, 38), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_ACCENT, 1, cv2.LINE_AA)

        cv2.imshow("NVIDIA RTX Power: Face Swap", frame)
        
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    clean()

if __name__ == "__main__":
    main()
