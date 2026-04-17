import cv2
import time
import sys
import signal
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import numpy as np

# Paleta de colores Premium (Tech Gold)
COLOR_HAND_NODOS = (0, 255, 127)      # Esmeralda
COLOR_FACE_TESSELATION = (200, 200, 200) # Gris claro
COLOR_TEXTO = (255, 255, 255)
COLOR_HUD_BG = (10, 10, 10)          # Casi negro
COLOR_ACCENT = (0, 215, 255)         # Amarillo Oro / Dorado (BGR)

def d_sq(m, p1, p2): 
    return (m[p1].x - m[p2].x)**2 + (m[p1].y - m[p2].y)**2

def get_hand_status(m):
    """Detecta el número de dedos levantados o el gesto de corazón."""
    f = [d_sq(m, 4, 17) > d_sq(m, 2, 17)] + [d_sq(m, t, 0) > d_sq(m, p, 0) for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)]]
    if f[2:] == [0, 0, 0] and d_sq(m, 4, 8) < d_sq(m, 5, 9) * 4: 
        return "Corazón ❤️"
    return str(sum(f))

def get_emotion(shapes):
    """Clasifica la emoción basada en blendshapes."""
    s = {b.category_name: b.score for b in shapes}
    if s.get('mouthSmileLeft', 0) > 0.4 and s.get('mouthSmileRight', 0) > 0.4: return "Feliz 😊"
    if s.get('jawOpen', 0) > 0.4: return "Sorprendido 😲"
    if s.get('browInnerUp', 0) > 0.3 and s.get('mouthFrownLeft', 0) > 0.3: return "Triste 😢"
    if s.get('eyeBlinkLeft', 0) > 0.5 and s.get('eyeBlinkRight', 0) > 0.5: return "Ojos cerrados 😴"
    return "Neutral 😐"

def dibujar_interfaz(frame, txt_mano, txt_emocion):
    """Dibuja un HUD tecnológico sofisticado en tonos dorados sin superposición."""
    h, w, _ = frame.shape
    # Overlay superior semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Líneas decorativas doradas
    cv2.line(frame, (0, 58), (w, 58), COLOR_ACCENT, 2)

    # Info HUD - Extremo Izquierdo (Estado del Sistema)
    cv2.putText(frame, f"SISTEMA VISION IA: ACTIVO", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
    
    # Info central/derecha - (Resultados del Tracking)
    txt_status = f"HANDS: {txt_mano} | {txt_emocion}"
    font_scale = 0.6
    tw = cv2.getTextSize(txt_status, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0][0]
    
    # Ubicamos el texto a la derecha con un margen
    cv2.putText(frame, txt_status, (w - tw - 20, 38), cv2.FONT_HERSHEY_DUPLEX, font_scale, COLOR_TEXTO, 1, cv2.LINE_AA)

def main():
    print("Iniciando Vision Intelligence Suite...")
    
    Base, Run = BaseOptions, vision.RunningMode.VIDEO
    try:
        # Inicialización de detectores
        h_det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=Base('hand_landmarker.task'),
            running_mode=Run,
            num_hands=2,
            min_hand_detection_confidence=0.3, # Más sensible
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        ))
        f_det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=Base('face_landmarker.task'), running_mode=Run, num_faces=1, output_face_blendshapes=True))
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        return

    cap = cv2.VideoCapture(0)
    def clean(*_): 
        cap.release()
        cv2.destroyAllWindows()
        print("\nCámaras liberadas. Hasta pronto.")
        sys.exit(0)

    # Configuración de señales para salida limpia (Compatibilidad Windows/Linux)
    signals = [signal.SIGINT]
    if hasattr(signal, "SIGTSTP"): signals.append(signal.SIGTSTP) # Solo en Unix
    for sig in signals: signal.signal(sig, clean)

    cv2.namedWindow("Vision Intelligence Suite", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision Intelligence Suite", 1280, 720)

    t_prev = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        mp_img = mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = int(time.time() * 1000)
        
        # Inferencia
        h_res, f_res = h_det.detect_for_video(mp_img, ts), f_det.detect_for_video(mp_img, ts)
        
        # Eliminado efecto gris para máxima visibilidad de los nodos
        h, w = frame.shape[:2]
        
        mano_txt = get_hand_status(h_res.hand_landmarks[0]) if h_res.hand_landmarks else "-"
        emocion_txt = get_emotion(f_res.face_blendshapes[0]) if f_res.face_blendshapes else "Neutral"

        # Dibujar Hand Landmarks (Con mayor grosor y brillo)
        if h_res.hand_landmarks:
            for marks in h_res.hand_landmarks:
                # Dibujar conexiones
                for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    p1, p2 = marks[c.start], marks[c.end]
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (180, 105, 255), 2, cv2.LINE_AA)
                # Dibujar nodos
                for point in marks: 
                    # Nodos con borde para que resalten
                    pos = (int(point.x*w), int(point.y*h))
                    cv2.circle(frame, pos, 5, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, pos, 6, (255, 255, 255), 1, cv2.LINE_AA)

        # Dibujar Face Landmarks (Teselación minimalista)
        for f_marks in (f_res.face_landmarks or []):
            for i, c in enumerate(vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION):
                if i % 5 == 0: 
                    p1, p2 = f_marks[c.start], f_marks[c.end]
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (150, 150, 150), 1, cv2.LINE_AA)

        # Interfaz
        dibujar_interfaz(frame, mano_txt, emocion_txt)

        cv2.imshow("Vision Intelligence Suite", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    clean()

if __name__ == "__main__": 
    main()
