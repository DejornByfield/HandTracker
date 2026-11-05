import csv
import os
import time

import cv2
import mediapipe as mp

from features import hand_features

LABELS = ["open", "fist", "pinch_index", "pinch_middle", "pinch_ring",
          "peace", "thumbs_up", "other"]       # edit freely; keys 1..N select a label
SAMPLES_PER_TAKE = 150
COUNTDOWN = 2.0
CSV_FILE = "gestures.csv"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

counts = {label: 0 for label in LABELS}
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, newline="") as f:
        for row in csv.reader(f):
            if row and row[0] in counts:
                counts[row[0]] += 1

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

countdown = None      # (label, time to start recording)
recording = None      # [label, samples left]

with HandLandmarker.create_from_options(options) as landmarker, \
        open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        h, w, _ = frame.shape
        now = time.time()
        hand = result.hand_landmarks[0] if result.hand_landmarks else None

        if hand:
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        if countdown and now >= countdown[1]:
            recording = [countdown[0], SAMPLES_PER_TAKE]
            countdown = None

        if recording and hand:
            writer.writerow([recording[0]] + hand_features(hand, w, h).tolist())
            counts[recording[0]] += 1
            recording[1] -= 1
            if recording[1] == 0:
                f.flush()
                recording = None

        if countdown:
            msg = "GET READY: %s (%.1f)" % (countdown[0], countdown[1] - now)
        elif recording:
            msg = "RECORDING %s  %d left" % (recording[0], recording[1])
        else:
            msg = "Press a number key to record a gesture"
        cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for i, label in enumerate(LABELS):
            cv2.putText(frame, "%d: %s (%d)" % (i + 1, label, counts[label]),
                        (10, 80 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Collect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if countdown is None and recording is None:
            for i, label in enumerate(LABELS):
                if key == ord(str(i + 1)):
                    countdown = (label, now + COUNTDOWN)

cap.release()
cv2.destroyAllWindows()