import numpy as np


def hand_features(hand, w, h):
    """21 landmarks -> 63 numbers that ignore where the hand is and how big it looks."""
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand], dtype=np.float32)
    pts -= pts[0]                              # wrist becomes the origin
    scale = np.linalg.norm(pts[9][:2])         # wrist -> middle knuckle = hand size
    pts /= max(scale, 1e-6)                    # now distance from the camera doesn't matter
    return pts.flatten()