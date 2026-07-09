from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import PIL.ImageOps
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .cache import content_key, get_eyes, put_eyes

_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
_landmarker = None

# MediaPipe eye landmarks per eye
_EYES = [
    ((33, 133), [(159, 145), (158, 153)]),
    ((362, 263), [(386, 374), (385, 380)]),
]


def sharpness(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()  # focus measure


def exposure(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    dark = (img < 16).mean()     # fraction crushed to near-black
    bright = (img > 240).mean()  # fraction blown to near-white
    return 1.0 - (dark + bright)


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL)),
                num_faces=10,
            )
        )
    return _landmarker


def _tile_boxes(w, h, frac =0.5, step = 0.25):
    def starts(size):
        win, stride = int(size * frac), max(int(size * step), 1)
        xs = list(range(0, size - win + 1, stride))
        if xs[-1] != size - win:
            xs.append(size - win)
        return xs, win
    xs, tw = starts(w)
    ys, th = starts(h)
    return [(x, y, x + tw, y + th) for x in xs for y in ys]


def _ear(landmarks, w, h):
    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])  # normalized
    eyes = []
    for (c1, c2), verts in _EYES:
        horiz = np.linalg.norm(pt(c1) - pt(c2))
        if horiz == 0:
            continue
        vert = np.mean([np.linalg.norm(pt(t) - pt(b)) for t, b in verts])
        eyes.append(vert / horiz)
    return min(eyes) if eyes else None  


def eyes_open(path):
    key = content_key(path)
    hit, value = get_eyes(key)
    if hit:
        return value
    value = _compute_eyes(path)
    put_eyes(key, value)
    return value


def _compute_eyes(path):
    img = PIL.ImageOps.exif_transpose(PIL.Image.open(path)).convert("RGB")  # honor rotation
    w, h = img.size
    lm = _get_landmarker()
    ears = []
    for box in [(0, 0, w, h)] + _tile_boxes(w, h):  # catch small faces
        crop = img.crop(box)
        cw, ch = crop.size
        arr = np.ascontiguousarray(np.asarray(crop))
        result = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        for face in result.face_landmarks:
            e = _ear(face, cw, ch)
            if e is not None:
                ears.append(e)
    return min(ears) if ears else None  # None = no face
