# HandTracker

Control your computer's mouse with your hand, using nothing but a webcam. Move the cursor by moving your hand, click and drag by pinching, scroll by pinching and sliding, and flip pages in a PDF with gestures you train yourself.

Built for studying hands-free (reading a PDF from a distance while taking notes), but it works for any app.

Everything runs locally. No video is recorded, saved, or sent anywhere. The training tool stores only hand-landmark numbers, never images.

## Features

- **Cursor control** with MediaPipe hand tracking, smoothed by a One Euro filter (steady when you hold still, responsive when you move fast).
- **Click, drag, right-click and scroll** from pinch gestures. A quick pinch is a click, and a held pinch becomes a drag.
- **Trainable gestures.** Record your own gestures, train a small classifier, and map each one to an action. Peace sign and thumbs-up are included.
- **Calibration** so the control area fits your arm length and seating position.
- **Pause with a fist,** so you can scratch your nose without clicking things.
- **On-screen feedback** showing the current mode, your pinch distances, the recognized gesture and the last action fired.
- **Packages into a single Windows app** with PyInstaller.

## Gestures

| Gesture | What it does |
| --- | --- |
| Open hand, move | Moves the cursor (it follows your middle knuckle, which stays steady while you pinch) |
| Quick pinch, thumb + index | Left click (fires when you release) |
| Hold the pinch, thumb + index | Drag; release to drop |
| Pinch, thumb + middle | Right click |
| Pinch, thumb + ring, then move your hand up or down | Scroll (hand up moves the page down, like dragging on a touchscreen) |
| Fist | Pause: the mouse ignores your hand until you open it |
| Peace sign | Page Down |
| Thumbs up | Page Up |

Pinch distances are measured relative to your palm size, so everything works the same whether you are near the camera or far from it. A gesture the model recognizes always wins over the pinch and fist rules, which is why a thumbs-up isn't mistaken for a fist.

**Keys in the camera window:** `c` recalibrates, `q` quits.

**Emergency stop:** push your real mouse into a screen corner and the app stops (this is pyautogui's built-in failsafe).

## Requirements

- Windows 10 or 11 (developed and tested on Windows 11; other platforms are untested)
- Python 3.10 or newer (tested on 3.14)
- A webcam

## Setup

```
git clone https://github.com/DejornByfield/HandTracker.git
cd HandTracker
pip install opencv-python mediapipe pyautogui numpy scikit-learn
```

The repository includes `hand_landmarker.task` (MediaPipe's hand model) and a trained `gesture_model.joblib`. If you need to download the hand model again, get it from the [MediaPipe Hand Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

> **Note:** this project uses MediaPipe's current **Tasks API**. Recent MediaPipe releases removed the old `mp.solutions` interface, so older tutorials will fail with `module 'mediapipe' has no attribute 'solutions'`.

## Usage

```
python handmouse.py
```

On first run you will be asked to calibrate:

1. Sit and hold your arm the way you will actually use it.
2. Reach to the **top-left** of your comfortable range and press **SPACE**. Hold still for a second.
3. Do the same for the **bottom-right**.

The cyan rectangle in the camera window shows the area of the frame that maps to your whole screen. Your calibration is saved to `%APPDATA%\HandMouse\calibration.json`, so you only do this once. Press `c` any time to redo it.

Tips for the best experience:

- Use a well-lit room. Tracking gets noticeably worse in dim light.
- Click on your PDF viewer once so it has keyboard focus. Page Down and Page Up keystrokes go to the focused window, while scrolling goes to the window under the cursor.
- Choose comfortable calibration corners instead of your maximum stretch, so you are not straining to reach the screen edges.

## Training your own gestures

The recognition pipeline turns the 21 hand landmarks into numbers that ignore where your hand is and how big it looks (`features.py`), records labeled examples (`collect.py`), and trains a random forest (`train.py`).

1. **Choose your labels.** Edit the `LABELS` list at the top of `collect.py`. Keep an `other` class (relaxed hands, half-formed shapes) so the model has somewhere to put poses that aren't one of your gestures.
2. **Record examples.** Run `python collect.py`, press a number key to pick a gesture, and after a 2-second countdown it records 150 frames. Record each gesture 2 or 3 times.
3. **Train.** Run `python train.py`. It prints a report and saves `gesture_model.joblib`.
4. **Map it to an action.** Add the label to `GESTURE_ACTIONS` in `handmouse.py`, for example `"peace": lambda: pyautogui.press("pagedown")`.
5. **Run it** with `python handmouse.py` and try the gesture.

What makes good training data:

- Move your hand around the frame, tilt it, and change the distance while recording.
- Vary the lighting and distance between takes.
- Aim for roughly 300 to 450 samples per gesture.
- If two gestures get confused, record more varied examples of both rather than changing code.

The accuracy that `train.py` prints is optimistic, because neighboring video frames look almost identical. The real test is trying the gestures live.

## Settings

These are constants near the top of `handmouse.py`:

| Setting | What it controls |
| --- | --- |
| `CAM_W`, `CAM_H` | Camera resolution. Higher helps tracking at a distance. |
| `MIN_CUTOFF` | Cursor steadiness when still. Lower is steadier. |
| `BETA` | Cursor lag when moving fast. Higher means less lag. |
| `PINCH_ON`, `PINCH_OFF` | How close fingers must be to close and release a pinch (relative to palm size). |
| `HOLD_TIME` | Seconds a pinch must be held to become a drag instead of a click. |
| `SCROLL_GAIN` | Scroll speed. |
| `GESTURE_HOLD` | Seconds a gesture must be held before its action fires. |
| `GESTURE_COOLDOWN` | Minimum seconds between two gesture actions. |
| `GESTURE_ACTIONS` | Which gesture triggers which action. |

## Build a Windows app

```
pip install pyinstaller
pyinstaller --noconfirm --onedir --name HandMouse --add-data "hand_landmarker.task;." --add-data "gesture_model.joblib;." --collect-all mediapipe handmouse.py
```

The app is created at `dist\HandMouse\HandMouse.exe`. Add `--noconsole` once everything works to hide the terminal window.

Do not commit `dist/` or `build/` to Git. They contain files larger than GitHub's 100 MB limit, and you can rebuild them at any time. Share the built app by zipping `dist\HandMouse` and attaching it to a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

## Project layout

| File | Purpose |
| --- | --- |
| `handmouse.py` | The app: tracking, cursor control, pinches, gestures, calibration |
| `features.py` | Turns hand landmarks into position- and size-independent features |
| `collect.py` | Records labeled gesture examples to `gestures.csv` |
| `train.py` | Trains the classifier and saves `gesture_model.joblib` |
| `hand_landmarker.task` | MediaPipe's hand landmark model |
| `gesture_model.joblib` | The trained gesture classifier |
| `gestures.csv` | The recorded training data |

## Troubleshooting

- **`module 'mediapipe' has no attribute 'solutions'`:** you are running code written for the old API. This project uses the Tasks API, so make sure you are running `handmouse.py` from this repository.
- **No older MediaPipe version available:** on Python 3.14, only recent MediaPipe releases exist. That is fine, because this project is written for them.
- **Black camera window or camera not found:** change the camera index in `cv2.VideoCapture(0)` to `1`, and close other apps that might be using the camera.
- **A pinch clicks too easily or not enough:** read the three pinch ratios shown in the camera window while you pinch, then adjust `PINCH_ON`.
- **Gesture keeps getting confused with another:** record more varied samples of both and retrain.
- **Packaged app crashes with "Unable to open file":** a bundled file is being looked up by a relative path. Load bundled files through the `resource_path` helper so they resolve inside the package.

## Known issues

- The cursor can feel choppy while the camera window is minimized, which looks like Windows throttling background windows. Keeping the window visible avoids it.
- Windows can only send input to windows running with normal privileges. Apps running as administrator (such as Task Manager) will not respond to the cursor.
- Accuracy drops at a distance of more than a meter or two with a typical webcam, because your hand covers fewer pixels.

## Ideas for the future

- System tray icon with pause, recalibrate and quit
- More study actions: zoom, jump to top or bottom, a sticky "highlight" mode for selecting text
- Settings in a config file instead of constants in the code
- Two-hand control, per-app gesture profiles, and an installer

## Built with

[MediaPipe](https://ai.google.dev/edge/mediapipe) for hand tracking, [OpenCV](https://opencv.org/) for the camera, [pyautogui](https://pyautogui.readthedocs.io/) for mouse and keyboard control, [scikit-learn](https://scikit-learn.org/) for the gesture classifier, and [PyInstaller](https://pyinstaller.org/) for packaging.
