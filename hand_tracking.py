import cv2
import time
import sys
import signal
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import numpy as np

# Paleta de colores Premium (Cyberpunk/Tech)
COLOR_HAND_NODOS = (0, 255, 127)      # Esmeralda
COLOR_FACE_TESSELATION = (255, 255, 255) # Blanco (sutil)
COLOR_TEXTO = (255, 255, 255)
COLOR_HUD_BG = (20, 20, 20)          # Casi negro
COLOR_ACCENT = (180, 105, 255)       # Neón Púrpura

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

def dibujar_interfaz(frame, txt_mano, txt_emocion, fps):
    """Dibuja un HUD tecnológico sofisticado."""
    h, w, _ = frame.shape
    # Overlay superior semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Línea decorativa neón
    cv2.line(frame, (0, 60), (w, 60), COLOR_ACCENT, 2)

    # Info HUD
    cv2.putText(frame, f"SISTEMA IA: ACTIVO", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_ACCENT, 1)
    cv2.putText(frame, f"FPS: {fps}", (w - 120, 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_TEXTO, 1)
    
    # Info central (Gesto y Emoción)
    txt_status = f"MANOS: {txt_mano} | EMOCIÓN: {txt_emocion}"
    font_scale = 0.7
    tw = cv2.getTextSize(txt_status, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0][0]
    cv2.putText(frame, txt_status, ((w - tw)//2, 35), cv2.FONT_HERSHEY_DUPLEX, font_scale, COLOR_TEXTO, 1)

def main():
    print("Iniciando Vision Intelligence Suite...")
    
    Base, Run = BaseOptions, vision.RunningMode.VIDEO
    try:
        # Inicialización de detectores
        h_det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=Base('hand_landmarker.task'), running_mode=Run, num_hands=2))
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

    for sig in (signal.SIGINT, signal.SIGTSTP): signal.signal(sig, clean)

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
        
        # Efecto visual: Desaturar ligeramente para enfoque técnico
        frame = cv2.addWeighted(frame, 0.5, cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), 0.5, 0)
        
        h, w = frame.shape[:2]
        
        mano_txt = get_hand_status(h_res.hand_landmarks[0]) if h_res.hand_landmarks else "-"
        emocion_txt = get_emotion(f_res.face_blendshapes[0]) if f_res.face_blendshapes else "Neutral"

        # Dibujar Hand Landmarks
        for marks in (h_res.hand_landmarks or []):
            for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                p1, p2 = marks[c.start], marks[c.end]
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), COLOR_ACCENT, 1, cv2.LINE_AA)
            for point in marks: 
                cv2.circle(frame, (int(point.x*w), int(point.y*h)), 4, COLOR_HAND_NODOS, -1, cv2.LINE_AA)

        # Dibujar Face Landmarks (Teselación minimalista)
        for f_marks in (f_res.face_landmarks or []):
            for i, c in enumerate(vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION):
                if i % 4 == 0: # Dibujar solo una parte para no saturar la imagen
                    p1, p2 = f_marks[c.start], f_marks[c.end]
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), COLOR_FACE_TESSELATION, 1, cv2.LINE_AA)

        # Cálculo de FPS
        t_curr = time.time()
        fps = int(1 / (t_curr - t_prev)) if (t_curr - t_prev) > 0 else 0
        t_prev = t_curr

        # Interfaz
        dibujar_interfaz(frame, mano_txt, emocion_txt, fps)

        cv2.imshow("Vision Intelligence Suite", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    clean()

if __name__ == "__main__": 
    main()
