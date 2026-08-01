import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi
import cv2
import numpy as np
import PIL.Image
import PIL.ImageOps
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .cache import content_key, get_faces, put_faces

_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
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


# autodownloader 
def _ensure_model():
    if _MODEL.exists():
        return
    _MODEL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face model (~3.7 MB) -> {_MODEL}")
    part = _MODEL.with_suffix(".part")
    ctx = ssl.create_default_context(cafile=certifi.where())  # macOS python ships no CA bundle
    with urllib.request.urlopen(_MODEL_URL, context=ctx) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f)
    part.rename(_MODEL)  # rename last


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _ensure_model()
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
    return _analyze_faces(path)[0]  # min eye-openness in the photo, or None


def face_sharpness(path):
    return _analyze_faces(path)[1]  # Laplacian variance of the face region, or None


def _analyze_faces(path):
    key = content_key(path)
    row = get_faces(key)
    if row is not None:
        return row
    result = _detect(path)
    put_faces(key, *result)
    return result


def _detect(path):
    img = PIL.ImageOps.exif_transpose(PIL.Image.open(path)).convert("RGB")  # honor rotation
    w, h = img.size
    lm = _get_landmarker()
    ears, face_sharps = [], []
    for box in [(0, 0, w, h)] + _tile_boxes(w, h):  # catch small faces
        crop = img.crop(box)
        cw, ch = crop.size
        arr = np.ascontiguousarray(np.asarray(crop))
        result = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        gray = None
        for face in result.face_landmarks:
            e = _ear(face, cw, ch)
            if e is not None:
                ears.append(e)
            if gray is None:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            s = _face_lap(face, gray, cw, ch)
            if s is not None:
                face_sharps.append(s)
    return (min(ears) if ears else None,          # blinking 
            max(face_sharps) if face_sharps else None)  # sharpest face; None = no face


def _face_lap(landmarks, gray, w, h):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x0, x1 = max(int(min(xs) * w), 0), min(int(max(xs) * w), w)
    y0, y1 = max(int(min(ys) * h), 0), min(int(max(ys) * h), h)
    region = gray[y0:y1, x0:x1]
    if region.size < 400:  # face too small to measure focus reliably
        return None
    return cv2.Laplacian(region, cv2.CV_64F).var()  # focus of the face itself
