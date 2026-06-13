from pathlib import Path

import imagehash
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    return sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def hash_image(path):
    with Image.open(path) as im:
        return imagehash.phash(im)


def hash_folder(folder):
    return [(p, hash_image(p)) for p in list_images(folder)]
