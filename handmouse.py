import math
import time

import cv2
import mediapipe as mp
import pyautogui

import json 

import  joblib 
from collections import Counter, deque
from features import hand_features



CAM_W, CAM_H = 1280, 720      # camera resolution (higher = better tracking at distance)
BOX_X = (0.20, 0.80)          # part of the camera frame (left..right) that maps to the full screen
BOX_Y = (0.15, 0.65)          # part of the camera frame (top..bottom) that maps to the full screen
MIN_CUTOFF = 1                # lower = steader when still (more smoothing)
BETA = 8.0                    # higher = less lag when moving fast 
PINCH_ON = 0.30               # pinch counts as "closed" below this (relative to palm size)
PINCH_OFF = 0.50              # pinch counts as "released" above this (hysteresis stops flicker)
SCROLL_GAIN = 3000            # scroll speed while the scroll pinch is held
EDGE = 5                      # keeps the cursor out of the screen corners (see failsafe)
HOLD_TIME = 0.45              # how long to hold a pinch before it counts as a drag

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

class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=8.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt                       # this is how fast it's moving
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev      # smoothed speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)  # faster -> less smoothing
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


CALIB_FILE = "calibration.json"

def load_calibration():
    try:
        with open(CALIB_FILE) as f: 
            d = json.load(f)
        return tuple(d["x"]), tuple(d["y"])
    except Exception:
        return None

def save_calibration(bx, by):
    with open(CALIB_FILE, "w") as f:
        json.dump({"x": bx, "y": by}, f)

def calibrate(cap, landmarker):
    # calibrate the control box by having the user reach to two corners and press SPACE 
    points = []
    for name in ("TOP-LEFT", "BOTTOM-RIGHT"):
        samples = []
        capture_until = None
        while True:
            ok, frame = cap.read()
            if not ok:
                return None
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            h, w, _ = frame.shape
            now = time.time()
            pointer = result.hand_landmarks[0][9] if result.hand_landmarks else None
            if pointer:
                cv2.circle(frame, (int(pointer.x * w), int(pointer.y * h)), 10, (0, 0, 255), -1)

            if capture_until is not None:
                msg = "Hold still..."
                if pointer:
                    samples.append((pointer.x, pointer.y))
                if now >= capture_until:
                    if len(samples) >= 5:
                        break                      # got enough samples
                    capture_until, samples = None, []   # hand was lost, try again
            else:
                msg = "Reach to your %s comfortable point, press SPACE" % name

            cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, "q = cancel", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Hand Mouse", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None
            if key == 32 and capture_until is None and pointer:   # SPACE
                capture_until, samples = now + 1.0, []

        px = sum(s[0] for s in samples) / len(samples)
        py = sum(s[1] for s in samples) / len(samples)
        points.append((px, py))

    (x1, y1), (x2, y2) = points
    bx = (min(x1, x2), max(x1, x2))
    by = (min(y1, y2), max(y1, y2))
    if bx[1] - bx[0] < 0.1 or by[1] - by[0] < 0.1:      # corners too close together
        return None
    return bx, by
    

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

cur_x = cur_y = None          # smoothed cursor position
filt_x = OneEuro(MIN_CUTOFF, BETA)
filt_y = OneEuro(MIN_CUTOFF, BETA)
mode = None                   # None, "left", "right" or "scroll"
left_down = False             # is the left button currently held?
pinch_start = 0.0             # when the current pinch started 
scroll_prev_y = None
scroll_accum = 0.0

try:
    gesture_model = joblib.load("gesture_model.joblib")
except Exception:
    gesture_model = None
recent = deque(maxlen=5)             # recent gesture predictions, for smoothing

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        calib = load_calibration() or calibrate(cap, landmarker)
        if calib:
            BOX_X, BOX_Y = calib
            save_calibration(BOX_X, BOX_Y)
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
            gesture = ""

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                if gesture_model is not None: 
                    feats = hand_features(hand, w, h).reshape(1, -1)
                    probs = gesture_model.predict_proba(feats)[0]
                    best = probs.argmax()
                    recent.append(gesture_model.classes_[best] if probs[best] > 0.6 else "?")
                    gesture = Counter(recent).most_common(1)[0][0]
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
                now = time.time()
                if new_mode != mode:
                    if mode == "left":
                        if left_down:                           # was dragging: drop it
                            pyautogui.mouseUp()
                            left_down = False
                        elif new_mode is None and not paused:   # let go before HOLD_TIME: a click
                            pyautogui.click()
                    if new_mode == "left":
                        pinch_start = now                       # start timing the hold
                    elif new_mode == "right":
                        pyautogui.rightClick()
                    elif new_mode == "scroll":
                        scroll_prev_y = pointer.y
                        scroll_accum = 0.0
                    mode = new_mode

                # ---- a pinch held long enough becomes a drag ----
                if mode == "left" and not left_down and now - pinch_start >= HOLD_TIME:
                    pyautogui.mouseDown()
                    left_down = True

                # ---- move or scroll ----
                if paused:
                    status = "PAUSED (fist)"
                elif mode == "scroll":
                    status = "SCROLL"
                    dy = pointer.y - scroll_prev_y
                    scroll_prev_y = pointer.y
                    scroll_accum += dy * SCROLL_GAIN
                    amount = int(scroll_accum)
                    if amount:
                        pyautogui.scroll(amount)
                        scroll_accum -= amount
                elif mode in ("left", "right") and not left_down:
                    # cursor stays put so the click lands where you aimed
                    status = "PINCH - hold to drag" if mode == "left" else "RIGHT CLICK"
                else:
                    status = "DRAGGING" if left_down else "MOVE"
                    t = time.time()
                    fx = filt_x(pointer.x, t)
                    fy = filt_y(pointer.y, t)
                    tx = remap(fx, BOX_X[0], BOX_X[1], screen_w)
                    ty = remap(fy, BOX_Y[0], BOX_Y[1], screen_h)
                    cur_x = max(EDGE, min(screen_w - 1 - EDGE, tx))
                    cur_y = max(EDGE, min(screen_h - 1 - EDGE, ty))
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
                filt_x.reset()
                filt_y.reset()

            # control box and status text
            cv2.rectangle(frame,
                          (int(BOX_X[0] * w), int(BOX_Y[0] * h)),
                          (int(BOX_X[1] * w), int(BOX_Y[1] * h)),
                          (255, 255, 0), 2)
            cv2.putText(frame, status, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, ratio_text, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "gesture: " + gesture, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Hand Mouse", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                if left_down:
                    pyautogui.mouseUp()
                    left_down = False
                mode = None
                new_box = calibrate(cap, landmarker)
                if new_box:
                    BOX_X, BOX_Y = new_box
                    save_calibration(BOX_X, BOX_Y)
                filt_x.reset()
                filt_y.reset()
finally:
    if left_down:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
