import math
import time

import cv2
import mediapipe as mp
import pyautogui

# ---------------- settings you can tune ----------------
CAM_W, CAM_H = 1280, 720      # camera resolution (higher = better tracking at distance)
BOX_X = (0.20, 0.80)          # part of the camera frame (left..right) that maps to the full screen
BOX_Y = (0.15, 0.65)          # part of the camera frame (top..bottom) that maps to the full screen
SMOOTHING = 4.0               # higher = smoother but laggier cursor
PINCH_ON = 0.30               # pinch counts as "closed" below this (relative to palm size)
PINCH_OFF = 0.50              # pinch counts as "released" above this (hysteresis stops flicker)
SCROLL_GAIN = 3000            # scroll speed while the scroll pinch is held
EDGE = 5                      # keeps the cursor out of the screen corners (see failsafe)

pyautogui.PAUSE = 0           # no built-in delay after each call
pyautogui.FAILSAFE = True     # slam the real mouse into the top-left corner to emergency-stop
screen_w, screen_h = pyautogui.size()

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


def dist(a, b, w, h):
    """Distance between two landmarks in pixels."""
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def remap(v, lo, hi, out_size):
    """Map v from the range lo..hi onto 0..out_size, clamped."""
    t = (v - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return t * out_size


def is_fist(hand, w, h):
    """Fist = every fingertip is closer to the wrist than its middle joint."""
    wrist = hand[0]
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if dist(hand[tip], wrist, w, h) >= dist(hand[pip], wrist, w, h):
            return False
    return True


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

cur_x = cur_y = None          # smoothed cursor position
mode = None                   # None, "left", "right" or "scroll"
left_down = False             # is the left button currently held?
scroll_prev_y = None
scroll_accum = 0.0

try:
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
            status = "NO HAND"
            ratio_text = ""

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                thumb, index, middle, ring = hand[4], hand[8], hand[12], hand[16]
                pointer = hand[9]                      # middle knuckle: stays steady during pinches
                palm = dist(hand[0], hand[9], w, h)    # hand size, so distance doesn't matter

                ratios = {
                    "left": dist(thumb, index, w, h) / palm,
                    "right": dist(thumb, middle, w, h) / palm,
                    "scroll": dist(thumb, ring, w, h) / palm,
                }
                ratio_text = "idx %.2f  mid %.2f  ring %.2f" % (
                    ratios["left"], ratios["right"], ratios["scroll"])

                # ---- decide the current mode ----
                paused = is_fist(hand, w, h)
                if paused:
                    new_mode = None
                elif mode is not None and ratios[mode] < PINCH_OFF:
                    new_mode = mode                    # keep holding the same pinch
                else:
                    best = min(ratios, key=ratios.get)
                    new_mode = best if ratios[best] < PINCH_ON else None

                # ---- act on mode changes ----
                if new_mode != mode:
                    if mode == "left" and left_down:
                        pyautogui.mouseUp()
                        left_down = False
                    if new_mode == "left":
                        pyautogui.mouseDown()
                        left_down = True
                    elif new_mode == "right":
                        pyautogui.rightClick()
                    elif new_mode == "scroll":
                        scroll_prev_y = pointer.y
                        scroll_accum = 0.0
                    mode = new_mode

                # ---- move or scroll ----
                if paused:
                    status = "PAUSED (fist)"
                elif mode == "scroll":
                    status = "SCROLL"
                    dy = pointer.y - scroll_prev_y
                    scroll_prev_y = pointer.y
                    scroll_accum += dy * SCROLL_GAIN   # hand up -> page down (flip the sign to reverse)
                    amount = int(scroll_accum)
                    if amount:
                        pyautogui.scroll(amount)
                        scroll_accum -= amount
                else:
                    status = {"left": "LEFT CLICK / DRAG", "right": "RIGHT CLICK"}.get(mode, "MOVE")
                    tx = remap(pointer.x, BOX_X[0], BOX_X[1], screen_w)
                    ty = remap(pointer.y, BOX_Y[0], BOX_Y[1], screen_h)
                    tx = max(EDGE, min(screen_w - 1 - EDGE, tx))
                    ty = max(EDGE, min(screen_h - 1 - EDGE, ty))
                    if cur_x is None:
                        cur_x, cur_y = tx, ty
                    else:
                        cur_x += (tx - cur_x) / SMOOTHING
                        cur_y += (ty - cur_y) / SMOOTHING
                    pyautogui.moveTo(int(cur_x), int(cur_y))

                # ---- draw the hand ----
                for lm in hand:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
                cv2.circle(frame, (int(pointer.x * w), int(pointer.y * h)), 10, (0, 0, 255), -1)
            else:
                # hand left the frame: let go of everything
                if left_down:
                    pyautogui.mouseUp()
                    left_down = False
                mode = None

            # control box and status text
            cv2.rectangle(frame,
                          (int(BOX_X[0] * w), int(BOX_Y[0] * h)),
                          (int(BOX_X[1] * w), int(BOX_Y[1] * h)),
                          (255, 255, 0), 2)
            cv2.putText(frame, status, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, ratio_text, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    if left_down:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
