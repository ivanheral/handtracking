import cv2, time
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

def main():
    detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions('hand_landmarker.task'), running_mode=vision.RunningMode.VIDEO, num_hands=2))
    
    cap, t_prev = cv2.VideoCapture(0), 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        res = detector.detect_for_video(mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(time.time()*1000))

        h, w = frame.shape[:2]
        for marks in (res.hand_landmarks or []):
            for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                p1, p2 = marks[conn.start], marks[conn.end]
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0,0,255), 2)
            for m in marks:
                cv2.circle(frame, (int(m.x*w), int(m.y*h)), 5, (0,255,0), -1)

        t_curr = time.time()
        cv2.putText(frame, f'FPS: {int(1/(t_curr-t_prev) if t_prev else 0)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        t_prev = t_curr

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
