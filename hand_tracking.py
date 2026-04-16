import cv2, time, sys, signal
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

def d_sq(m, p1, p2): 
    return (m[p1].x - m[p2].x)**2 + (m[p1].y - m[p2].y)**2

def get_hand_status(m):
    f = [d_sq(m, 4, 17) > d_sq(m, 2, 17)] + [d_sq(m, t, 0) > d_sq(m, p, 0) for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)]]
    if f[2:] == [0, 0, 0] and d_sq(m, 4, 8) < d_sq(m, 5, 9) * 4: 
        return "Corazon"
    return str(sum(f))

def get_emotion(shapes):
    s = {b.category_name: b.score for b in shapes}
    if s.get('mouthSmileLeft', 0) > 0.4 and s.get('mouthSmileRight', 0) > 0.4: return "Feliz :)"
    if s.get('jawOpen', 0) > 0.4: return "Sorprendido :O"
    if s.get('browInnerUp', 0) > 0.3 and s.get('mouthFrownLeft', 0) > 0.3: return "Triste :("
    if s.get('eyeBlinkLeft', 0) > 0.5 and s.get('eyeBlinkRight', 0) > 0.5: return "Ojos cerrados -.-"
    return "Neutral"

def main():
    Base, Run = BaseOptions, vision.RunningMode.VIDEO
    h_det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=Base('hand_landmarker.task'), running_mode=Run, num_hands=2))
    f_det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=Base('face_landmarker.task'), running_mode=Run, num_faces=1, output_face_blendshapes=True))
    
    cap = cv2.VideoCapture(0)
    def clean(*_): cap.release(); cv2.destroyAllWindows(); sys.exit(0)
    for sig in (signal.SIGINT, signal.SIGTSTP): signal.signal(sig, clean)

    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracking", 1024, 768)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        mp_img = mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = int(time.time() * 1000)
        
        h_res, f_res = h_det.detect_for_video(mp_img, ts), f_det.detect_for_video(mp_img, ts)
        frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        
        hand_status = get_hand_status(h_res.hand_landmarks[0]) if h_res.hand_landmarks else "-"
        emotion = get_emotion(f_res.face_blendshapes[0]) if f_res.face_blendshapes else "Neutral"

        for marks in (h_res.hand_landmarks or []):
            for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                p1, p2 = marks[c.start], marks[c.end]
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 255), 1)
            for point in marks: cv2.circle(frame, (int(point.x*w), int(point.y*h)), 2, (0, 255, 0), -1)

        for f_marks in (f_res.face_landmarks or []):
            for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                p1, p2 = f_marks[c.start], f_marks[c.end]
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 255), 1)
            for point in f_marks: cv2.circle(frame, (int(point.x*w), int(point.y*h)), 2, (0, 255, 0), -1)

        txt = f"Mano: {hand_status} | Expresion: {emotion}"
        t_w = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
        cv2.putText(frame, txt, ((w - t_w)//2, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    clean()

if __name__ == "__main__": main()
