import cv2
import sys
import time
import os
import signal
import insightface
from insightface.app import FaceAnalysis
import numpy as np

def main():
    # El archivo de imagen objetivo y el modelo deben estar en la misma carpeta
    target_image_path = 'objetivo.jpg'
    model_path = 'inswapper_128.onnx'
    
    if not os.path.exists(target_image_path):
        print(f"[ERROR] No se ha encontrado la imagen '{target_image_path}'.")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"[ERROR] No se ha encontrado el modelo '{model_path}'.")
        sys.exit(1)

    print("Cargando los modelos de IA... Usando NVIDIA RTX GPU.")
    
    # Prioridad: TensorRT -> CUDA -> CPU
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

    # 1. Analizador facial: Buffalo_L para mayor precisión
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=providers)
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    # 2. Motor generativo InSwapper
    swapper = insightface.model_zoo.get_model(model_path, providers=providers)

    # 3. Extraer Embedding de la cara objetivo
    print(f"Analizando cara de: {target_image_path}...")
    target_img = cv2.imread(target_image_path)
    target_faces = face_analyzer.get(target_img)
    
    if len(target_faces) == 0:
        print("[ERROR] No se detectó ninguna cara en 'objetivo.jpg'. Asegúrate de que la cara sea visible.")
        sys.exit(1)
        
    target_identity = target_faces[0]

    # 4. Iniciar Webcam con reintento
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[AVISO] Intentando abrir la cámara de nuevo...")
        time.sleep(1)
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la webcam.")
        sys.exit(1)

    def cleanup(sig, frame):
        if cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Aplicación cerrada correctamente.")
        sys.exit(0)

    # Capturar señales de cierre
    signal.signal(signal.SIGINT, cleanup)

    cv2.namedWindow("Poder NVIDIA: Face Swap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Poder NVIDIA: Face Swap", 1024, 768)
    
    print("¡Sistema listo! Presiona 'Q' en la ventana para salir.")

    t_prev = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        # Buscar caras en el frame actual
        my_faces = face_analyzer.get(frame)
        
        if my_faces:
            # Ordenar por tamaño de caja (procesar solo la cara más grande/cercana)
            my_faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            
            try:
                # Aplicar el Swap sobre la cara real
                frame = swapper.get(frame, my_faces[0], target_identity, paste_back=True)
            except Exception as e:
                # Silenciar errores menores de inferencia
                pass
        
        # Calcular FPS
        t_curr = time.time()
        fps = 1/(t_curr-t_prev) if t_curr - t_prev > 0 else 0
        t_prev = t_curr
        
        # Overlay estético
        cv2.rectangle(frame, (10, 10), (280, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 127), 2)
        
        cv2.imshow("Poder NVIDIA: Face Swap", frame)
        
        # Tecla de salida
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cleanup(None, None)

if __name__ == "__main__":
    main()
