# HandTracker

HandTracker uses a webcam to track one hand and control the computer mouse with
hand movement and gestures. It is built with MediaPipe hand landmarks, OpenCV,
and PyAutoGUI. The repository includes the landmark task model, a trained
gesture classifier, and sample gesture data.

## Features

- Move the cursor by moving your hand.
- Pinch your thumb and index finger to click; hold the pinch to drag.
- Pinch your thumb and middle finger for a right-click.
- Pinch your thumb and ring finger to scroll.
- Hold a peace sign for Page Down or a thumbs-up for Page Up.
- Make a fist to pause mouse control.
- Calibrate the hand-movement area to your reach.

## Requirements

- Windows
- Python and pip
- A webcam

Install the packages used by the scripts:

```powershell
python -m pip install opencv-python mediapipe pyautogui joblib numpy scikit-learn
```

For building the Windows executable, also install PyInstaller:

```powershell
python -m pip install pyinstaller
```

## Run the hand mouse

From the project directory, run:

```powershell
python handmouse.py
```

The app opens the webcam and mirrors the camera view. On first run, it asks you
to calibrate two comfortable hand positions: reach to the indicated top-left
and bottom-right points, press **Space**, and hold still while samples are
collected. Press **C** in the main window to recalibrate, or **Q** to quit.

Calibration is saved under `%APPDATA%\HandMouse\calibration.json` (or in your
home directory if `APPDATA` is unavailable). The app also needs
`hand_landmarker.task` and `gesture_model.joblib` in the project directory.

Move your hand within the outlined area to move the cursor. Pinches select the
mouse action as described above; holding an index-finger pinch for about half
a second starts a drag. A fist pauses control while held. Gesture actions are
recognized using the included classifier. PyAutoGUI's failsafe is enabled:
move the physical mouse to the top-left corner of the screen to trigger it.

## Collect and train gestures

`collect.py` records labeled hand-landmark samples to `gestures.csv`. Start it
with:

```powershell
python collect.py
```

Press **1–8** to record the corresponding label shown in the camera window:
`open`, `fist`, `pinch_index`, `pinch_middle`, `pinch_ring`, `peace`,
`thumbs_up`, or `other`. Each take records 150 samples after a short countdown.
Press **Q** to quit. The collector appends to the existing CSV, so back it up
before collecting if you want to preserve the bundled dataset unchanged.

Train and evaluate a Random Forest classifier with:

```powershell
python train.py
```

Training reads `gestures.csv`, prints a classification report, and writes the
result to `gesture_model.joblib`. The model should be retrained after changing
the dataset or the feature representation in `features.py`.

## Swipe-to-scroll utility

Run the standalone swipe utility with:

```powershell
python swiper.py
```

It scrolls when the tracked index fingertip moves quickly up or down. Press
**Q** to quit.

## Build the Windows executable

With PyInstaller installed, build from the project directory:

```powershell
pyinstaller HandMouse.spec
```

The spec bundles `hand_landmarker.task`, `gesture_model.joblib`, and the
MediaPipe resources. The collected application is written to
`dist\HandMouse\`.

## Project files

- `handmouse.py` — webcam mouse control, gesture actions, and calibration.
- `swiper.py` — standalone swipe-to-scroll utility.
- `collect.py` — gesture data collection.
- `train.py` — gesture classifier training and evaluation.
- `features.py` — converts 21 hand landmarks into normalized features.
- `hand_landmarker.task` — MediaPipe hand-landmark model.
- `gesture_model.joblib` — trained gesture classifier.
- `gestures.csv` — labeled training samples.
