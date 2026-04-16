import cv2
import sys
import time
import os
import signal
import insightface
from insightface.app import FaceAnalysis

def main():
    # Asegúrate de colocar en esta misma carpeta una foto JPEG con la cara que vas a suplantar, 
    # y renómbrala a "objetivo.jpg"
    target_image_path = 'objetivo.jpg'
    model_path = 'inswapper_128.onnx'
    
    if not os.path.exists(target_image_path):
        print(f"[ERROR] No se ha encontrado la imagen '{target_image_path}'. Coloca una foto con tu cara objetivo en esta carpeta.")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"[ERROR] No se ha encontrado el modelo '{model_path}'.")
        print("Puedes descargarlo ejecutando: wget -O inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx")
        sys.exit(1)

    print("Cargando los modelos de IA... Esto puede tardar unos segundos.")
    
    # Proveedores de aceleración. Si no tienes NVIDIA (CUDA), usará la CPU automáticamente de forma muy pesada (y lenta).
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # 1. Analizador facial: RetinaFace + detección de keypoints geométricos
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=providers)
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    # 2. Motor generativo InSwapper
    swapper = insightface.model_zoo.get_model(model_path, providers=providers)

    # 3. Extraer Embedding de la cara que queremos colocar sobre nosotros
    print(f"Analizando identidad de la imagen objetivo: {target_image_path}...")
    target_img = cv2.imread(target_image_path)
    target_faces = face_analyzer.get(target_img)
    
    if len(target_faces) == 0:
        print("[ERROR] No se ha encontrado ninguna cara clara en la foto objetivo.")
        sys.exit(1)
        
    target_identity = target_faces[0] # Tomamos la primera cara detectada

    # 4. Iniciar Bucle Webcam
    cap = cv2.VideoCapture(0)
    
    def cleanup_and_exit(sig, frame):
        if cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTSTP, cleanup_and_exit)

    cv2.namedWindow("Real-time Face Swap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-time Face Swap", 1024, 768)

    print("¡Sistema listo! Mostrando cámara...")

    t_prev = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        # Buscar mi cara real en la cámara en este exacto milisegundo
        my_faces = face_analyzer.get(frame)
        
        # Si encuentra al menos una cara mía
        if my_faces:
            my_identity = my_faces[0]
            
            # Ejecutar inyección de Pixeles Generativos
            try:
                frame = swapper.get(frame, my_identity, target_identity, paste_back=True)
            except Exception as e:
                print(f"Error al intentar aplicar el Swap: {e}")
        
        # Calcular los FPS (en CPU serán muy bajos, el deepfake es costoso)
        t_curr = time.time()
        fps = int(1/(t_curr-t_prev) if t_curr - t_prev > 0 else 0)
        t_prev = t_curr
        
        # UI
        text_color = (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps} (Usa GPU para mejorar)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
        
        cv2.imshow("Real-time Face Swap", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cleanup_and_exit(None, None)

if __name__ == "__main__":
    main()
