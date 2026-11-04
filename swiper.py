import time
import cv2
import pyautogui                                   
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

SWIPE_THRESHOLD = 1.2
COOLDOWN = 0.5
SCROLL_AMOUNT = 600                                

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)

prev_y = None
prev_t = None
last_swipe_t = 0
swipe_label = ""

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape
        now = time.time()

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            tip = hand[8]

            if prev_y is not None:
                vy = (tip.y - prev_y) / (now - prev_t)

                if now - last_swipe_t > COOLDOWN:
                    if vy < -SWIPE_THRESHOLD:
                        swipe_label = "SWIPE UP"
                        last_swipe_t = now
                        pyautogui.scroll(-SCROLL_AMOUNT)   
                    elif vy > SWIPE_THRESHOLD:
                        swipe_label = "SWIPE DOWN"
                        last_swipe_t = now
                        pyautogui.scroll(SCROLL_AMOUNT)    

            prev_y, prev_t = tip.y, now

            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
            cv2.circle(frame, (int(tip.x * w), int(tip.y * h)), 10, (0, 0, 255), -1)
        else:
            prev_y = None

        if now - last_swipe_t < 0.8:
            cv2.putText(frame, swipe_label, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow("Swipe Scroll", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()