import torch, os, cv2, sys, time, insightface
from insightface.app import FaceAnalysis
import numpy as np

class RTXFaceSwap:
    """Sistema de Face Swap optimizado para arquitecturas NVIDIA RTX Blackwell (Serie 50)."""
    
    def __init__(self, target_path='objetivo.jpg', model='inswapper_128.onnx'):
        self._setup_cuda_dlls()
        
        # Configuración de Inferencia GPU
        self.providers = [
            ('CUDAExecutionProvider', {
                'device_id': '0', # El número en CUDA tiene que ser texto.
                'cudnn_conv_algo_search': 'HEURISTIC', # Vital para saltar el bug de convolución en Blackwell
                'use_tf32': '0'
            }), 
            'CPUExecutionProvider'
        ]
        
        print("[IA] Inicializando motores neuronales...")
        # Cargamos detección, puntos clave y reconocimiento (necesario para el ADN facial)
        self.analyzer = FaceAnalysis(
            name='buffalo_l', 
            providers=self.providers, 
            allowed_modules=['detection', 'landmark_2d_106', 'recognition']
        )
        
        # Primera pasada en alta resolución para capturar la cara objetivo con precisión
        self.analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        if not os.path.exists(model):
            raise FileNotFoundError(f"Modelo {model} no encontrado. Descárgalo de Hugging Face.")
            
        self.swapper = insightface.model_zoo.get_model(model, providers=self.providers)
        
        # Extraer Identidad Base de la foto original
        img = cv2.imread(target_path)
        if img is None: 
            raise FileNotFoundError(f"No se pudo leer la imagen: {target_path}")
            
        faces = self.analyzer.get(img)
        if not faces: 
            raise ValueError(f"No se detectó ninguna cara en {target_path}")
            
        # Almacenamos el embedding (ADN) de la cara más grande de la foto
        self.target_id = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        
        # Re-ajustamos a baja resolución para el seguimiento de cámara ultra-fluido (FPS)
        self.analyzer.prepare(ctx_id=0, det_size=(320, 320))

    def _setup_cuda_dlls(self):
        """Inyecta las librerías CUDA de PyTorch en el sistema para corregir bugs de DLLs de NVIDIA."""
        t_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(t_path):
            os.environ['PATH'] = t_path + os.pathsep + os.environ.get('PATH', '')
            try: os.add_dll_directory(t_path)
            except: pass

    def _enhance_face(self, frame, bbox):
        """
        Filtro de Post-Producción Cinematográfico.
        Inyecta un Unsharpen Mask de Alto Rango Dinámico (HDR falso) guiado por las coordenadas
        puras de la neurona. Devuelve una textura cristalina a la piel "pegada".
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Geometría de seguridad (Evitar crasheos de memoria Out-Of-Bounds por los pelos)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        
        region = frame[y1:y2, x1:x2]
        if region.size == 0: return frame
        
        # Unsharpen Mask: Mezclar capa nítida y sustraer una borrosa, forzando los microporos a brillar
        blur = cv2.GaussianBlur(region, (0, 0), 2.5)
        sharpened = cv2.addWeighted(region, 1.4, blur, -0.4, 0)
        
        frame[y1:y2, x1:x2] = sharpened
        return frame

    def start(self):
        """Arranca el bucle de captura y procesamiento en tiempo real."""
        cap = cv2.VideoCapture(0)
        # Volvemos a levantar la resolución a HD nativo (1280x720) 
        # Ahora tu gráfica va suelta y puede comerse este tamaño sin despeinarse.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        p_time = time.time()
        frame_counter = 0
        last_faces = None

        print("[OK] Sistema RTX Vision activo. Pulsa 'Q' para salir.")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1) # Espejo
                
                # --- MEJORA 4: FRAME SKIPPING (INTERPOLACIÓN) ---
                # En lugar de usar la GPU entera para redetectar tu cara desde cero
                # en cada milísegundo, actualizamos la detección 1 de cada 2 frames.
                # El cerebro humano no nota la inercia, ¡y multiplicamos los FPS!
                if frame_counter % 2 == 0:
                    last_faces = self.analyzer.get(frame)
                
                frame_counter += 1

                # Proceso de Intercambio
                if last_faces:
                    # Seleccionamos la cara más dominante en pantalla
                    current_face = max(last_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    try:
                        frame = self.swapper.get(frame, current_face, self.target_id, paste_back=True)
                        # --- MEJORA AUDIOVISUAL: Inyección del Filtro de Fusión ---
                        frame = self._enhance_face(frame, current_face.bbox)
                    except:
                        pass

                # Cálculo de Rendimiento (FPS)
                c_time = time.time()
                fps = 1.0 / (c_time - p_time) if (c_time - p_time) > 0 else 0
                p_time = c_time
                
                # HUD Minimalista
                cv2.putText(frame, f"RTX CORE: {int(fps)} FPS", (25, 45), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 215, 255), 1, cv2.LINE_AA)
                
                cv2.imshow("NVIDIA RTX Vision: Face Swap Core", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Sistema desconectado de forma segura.")

if __name__ == "__main__":
    try:
        app = RTXFaceSwap()
        app.start()
    except Exception as e:
        print(f"\n[ERROR CRÍTICO]: {e}")
        input("Presiona Enter para salir...")
